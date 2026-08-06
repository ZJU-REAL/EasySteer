/**
 * Export a SteeringSpec as runnable Python (vllm.steer_vectors API) or as
 * the OpenAI-compatible `extra_body` JSON fragment.
 */

import type { ApplySpec, SteeringSpec, VectorSpec } from "./spec";
import { specToJson } from "./spec";

function pyStr(s: string): string {
  return JSON.stringify(s);
}

function pyValue(v: unknown): string {
  if (v === null || v === undefined) return "None";
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "number") return String(v);
  if (typeof v === "string") return pyStr(v);
  if (Array.isArray(v)) return `[${v.map(pyValue).join(", ")}]`;
  if (typeof v === "object") {
    const entries = Object.entries(v as Record<string, unknown>).map(
      ([k, val]) => `${pyStr(k)}: ${pyValue(val)}`,
    );
    return `{${entries.join(", ")}}`;
  }
  return String(v);
}

function pyLayers(layers: number[] | null): string {
  if (!layers || layers.length === 0) return "None";
  // Contiguous ascending runs read better as range() calls.
  const isContiguous =
    layers.length > 2 &&
    layers.every((v, i) => i === 0 || v === layers[i - 1] + 1);
  if (isContiguous) {
    return `list(range(${layers[0]}, ${layers[layers.length - 1] + 1}))`;
  }
  return pyValue(layers);
}

function pyWindow(window: [number, number | null]): string {
  return `(${window[0]}, ${window[1] === null ? "None" : window[1]})`;
}

function pyApply(apply: ApplySpec, indent: string): string {
  const args: string[] = [`phases=${pyValue(apply.phases)}`];
  // Emitted in the schema's wire order: includes first, then excludes.
  const listKeys = [
    "tokens",
    "positions",
    "generation_positions",
    "exclude_tokens",
    "exclude_positions",
    "exclude_generation_positions",
  ] as const;
  const windowKeys = [
    "prompt_window",
    "generation_window",
    "exclude_prompt_window",
    "exclude_generation_window",
  ] as const;
  for (const key of listKeys) {
    const value = apply[key];
    if (value) args.push(`${key}=${pyValue(value)}`);
  }
  for (const key of windowKeys) {
    const value = apply[key];
    if (value) args.push(`${key}=${pyWindow(value)}`);
  }
  const oneLine = `ApplySpec(${args.join(", ")})`;
  if (oneLine.length + indent.length <= 88) return oneLine;
  return `ApplySpec(\n${indent}    ${args.join(`,\n${indent}    `)},\n${indent})`;
}

function pyVector(v: VectorSpec, indent: string): string {
  const args: string[] = [];
  if (v.source) args.push(`source=${pyStr(v.source)}`);
  if (v.data !== null && v.data !== undefined) {
    // Payload dicts are opaque to the UI; emit a placeholder the user
    // replaces with an easysteer.vectors adapter call.
    args.push("data=...,  # in-memory payload: see easysteer.vectors adapters");
  }
  if (v.algorithm !== "direct") args.push(`algorithm=${pyStr(v.algorithm)}`);
  if (v.scale !== 1.0) args.push(`scale=${v.scale}`);
  if (v.layers && v.layers.length > 0) args.push(`layers=${pyLayers(v.layers)}`);
  if (v.normalize) args.push("normalize=True");
  args.push(`apply=${pyApply(v.apply, indent + "    ")}`);
  if (v.params && Object.keys(v.params).length > 0) {
    args.push(`params=${pyValue(v.params)}`);
  }
  if (v.name) args.push(`name=${pyStr(v.name)}`);
  return `VectorSpec(\n${indent}    ${args.join(`,\n${indent}    `)},\n${indent})`;
}

export interface PythonExportOptions {
  model?: string;
  prompt?: string;
  maxTokens?: number;
  temperature?: number;
}

/** Full runnable offline-inference script using the vllm.steer_vectors API. */
export function toPython(spec: SteeringSpec, opts: PythonExportOptions = {}): string {
  const model = opts.model ?? "Qwen/Qwen2.5-1.5B-Instruct";
  const prompt = opts.prompt ?? "Hello!";
  const maxTokens = opts.maxTokens ?? 256;
  const temperature = opts.temperature ?? 0;
  const algorithms = [...new Set(spec.vectors.map((v) => v.algorithm))];
  const multiVector = spec.vectors.length > 1;

  const vectorBlocks = spec.vectors.map((v) => "        " + pyVector(v, "        ").trimStart());
  const specArgs: string[] = [`vectors=[\n${vectorBlocks.join(",\n")},\n    ]`];
  if (spec.conflict !== "priority") specArgs.push(`conflict=${pyStr(spec.conflict)}`);
  if (spec.debug) specArgs.push("debug=True");

  const llmArgs = [
    `model=${pyStr(model)}`,
    "enable_steer_vector=True",
    `steer_algorithms=${pyValue(algorithms)}`,
  ];
  if (multiVector) llmArgs.push("steer_multi_vector=True");

  return `from vllm import LLM, SamplingParams
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

spec = SteeringSpec(
    ${specArgs.join(",\n    ")},
)

llm = LLM(
    ${llmArgs.join(",\n    ")},
)
sampling = SamplingParams(temperature=${temperature}, max_tokens=${maxTokens})

messages = [{"role": "user", "content": ${pyStr(prompt)}}]
outputs = llm.chat(messages, sampling, steering=spec)
print(outputs[0].outputs[0].text)
`;
}

/** The `extra_body` fragment for an OpenAI-compatible chat.completions call. */
export function toExtraBody(spec: SteeringSpec): Record<string, unknown> {
  return { steering: specToJson(spec) };
}

export function toExtraBodyJson(spec: SteeringSpec): string {
  return JSON.stringify(toExtraBody(spec), null, 2);
}

/** curl one-liner against the OpenAI-compatible server. */
export function toCurl(
  spec: SteeringSpec,
  opts: { baseUrl?: string; model?: string; prompt?: string } = {},
): string {
  const baseUrl = (opts.baseUrl ?? "http://localhost:8000/v1").replace(/\/$/, "");
  const body = {
    model: opts.model ?? "your-model",
    messages: [{ role: "user", content: opts.prompt ?? "Hello!" }],
    steering: specToJson(spec),
  };
  return `curl ${baseUrl}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify(body, null, 2).replace(/'/g, "'\\''")}'`;
}
