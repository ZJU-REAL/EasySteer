/**
 * Minimal client for the vllm-steer OpenAI-compatible server.
 *
 * - `streamChatCompletion` posts to `{base}/chat/completions` with the
 *   steering spec inlined as the request-level `steering` field (what the
 *   OpenAI SDK's `extra_body` flattens into).
 * - `setServerSteering` sets the server-level default spec via
 *   `POST {base}/v1/steering {"spec": {...}}`.
 */

import type { SteeringSpec } from "./spec";
import { specToJson } from "./spec";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface StreamOptions {
  baseUrl: string;
  model: string;
  messages: ChatMessage[];
  steering: SteeringSpec | null;
  temperature?: number;
  maxTokens?: number;
  signal?: AbortSignal;
  onToken: (text: string) => void;
}

function joinUrl(base: string, path: string): string {
  return base.replace(/\/+$/, "") + path;
}

export async function streamChatCompletion(opts: StreamOptions): Promise<void> {
  const body: Record<string, unknown> = {
    model: opts.model,
    messages: opts.messages,
    stream: true,
  };
  if (opts.temperature !== undefined) body.temperature = opts.temperature;
  if (opts.maxTokens !== undefined) body.max_tokens = opts.maxTokens;
  if (opts.steering) body.steering = specToJson(opts.steering);

  const resp = await fetch(joinUrl(opts.baseUrl, "/chat/completions"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: opts.signal,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text.slice(0, 500)}`);
  }
  if (!resp.body) throw new Error("response has no body");

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === "[DONE]") return;
      let parsed: { choices?: { delta?: { content?: string } }[] };
      try {
        parsed = JSON.parse(payload);
      } catch (e) {
        throw new Error(`malformed SSE chunk: ${payload.slice(0, 200)}`);
      }
      const delta = parsed.choices?.[0]?.delta?.content;
      if (delta) opts.onToken(delta);
    }
  }
}

/** Replace the server-level default steering spec (null clears it). */
export async function setServerSteering(
  baseUrl: string,
  spec: SteeringSpec | null,
): Promise<unknown> {
  const resp = await fetch(joinUrl(baseUrl, "/steering"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ spec: spec ? specToJson(spec) : null }),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text.slice(0, 500)}`);
  }
  return resp.json();
}

/** List models served by the endpoint (used as a connectivity check). */
export async function listModels(baseUrl: string): Promise<string[]> {
  const resp = await fetch(joinUrl(baseUrl, "/models"));
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const json = (await resp.json()) as { data?: { id: string }[] };
  return (json.data ?? []).map((m) => m.id);
}
