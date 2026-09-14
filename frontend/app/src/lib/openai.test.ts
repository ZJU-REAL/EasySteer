import { afterEach, describe, expect, it, vi } from "vitest";

import { setServerSteering, streamChatCompletion } from "./openai";

afterEach(() => vi.unstubAllGlobals());

describe("streamed completion content", () => {
  it("delivers ordered deltas across split SSE lines and UTF-8 characters", async () => {
    const event = (delta: object) => `data: ${JSON.stringify({ choices: [{ delta }] })}\r\n\r\n`;
    const bytes = new TextEncoder().encode(
      ": keepalive\n\n" + event({ role: "assistant" }) +
      event({ content: "Hello " }) + event({ content: "世界" }) +
      "data: [DONE]\n\n" + event({ content: "ignored after done" }),
    );
    // One byte per chunk forces both partial JSON lines and partial UTF-8.
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
        controller.close();
      },
    });
    vi.stubGlobal("fetch", vi.fn(async () => new Response(stream)));
    const tokens: string[] = [];
    await streamChatCompletion({
      baseUrl: "http://server/v1", model: "test", messages: [], steering: null,
      onToken: (token) => tokens.push(token),
    });
    expect(tokens).toEqual(["Hello ", "世界"]);
  });

  it.each([
    [new Response("backend error", { status: 503 }), /HTTP 503: backend error/],
    [new Response("data: {invalid}\n\n"), /malformed SSE chunk/],
  ])("rejects a failed response instead of silently completing", async (response, error) => {
    vi.stubGlobal("fetch", vi.fn(async () => response));
    const onToken = vi.fn();
    await expect(streamChatCompletion({
      baseUrl: "http://server/v1", model: "test", messages: [], steering: null, onToken,
    })).rejects.toThrow(error);
    expect(onToken).not.toHaveBeenCalled();
  });
});

describe("default steering requests", () => {
  it.each([false, null] as const)("preserves the %s request choice", async (steering) => {
    const fetchMock = vi.fn<typeof fetch>(async () => new Response("data: [DONE]\n\n"));
    vi.stubGlobal("fetch", fetchMock);
    await streamChatCompletion({
      baseUrl: "http://server/v1", model: "test", messages: [],
      steering, onToken: () => {},
    });
    const body = JSON.parse(fetchMock.mock.calls[0][1]!.body as string);
    if (steering === false) expect(body.steering).toBe(false);
    else expect(body).not.toHaveProperty("steering");
  });

  it("clears the default with an explicit null spec", async () => {
    const fetchMock = vi.fn<typeof fetch>(async () => Response.json({ active: false }));
    vi.stubGlobal("fetch", fetchMock);
    await setServerSteering("http://server/v1", null);
    expect(JSON.parse(fetchMock.mock.calls[0][1]!.body as string)).toEqual({ spec: null });
  });
});
