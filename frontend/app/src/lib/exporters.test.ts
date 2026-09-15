import { describe, expect, it } from "vitest";
import { toCurl, toExtraBody, toPython } from "./exporters";
import { defaultApplySpec, defaultSteeringSpec, defaultVectorSpec } from "./spec";

function sampleSpec() {
  const spec = defaultSteeringSpec();
  spec.vectors[0].source = "vectors/happy.gguf";
  spec.vectors[0].scale = 2.0;
  spec.vectors[0].layers = [10, 11, 12];
  return spec;
}

describe("toExtraBody", () => {
  it("wraps the canonical spec JSON under the steering key", () => {
    expect(toExtraBody(sampleSpec())).toEqual({
      steering: {
        vectors: [
          {
            source: "vectors/happy.gguf",
            scale: 2.0,
            layers: [10, 11, 12],
            apply: { "prompt": "all", "generation": "all" },
          },
        ],
      },
    });
  });
});

describe("toPython", () => {
  it("emits the v2 API with defaults omitted", () => {
    const code = toPython(sampleSpec(), { model: "Qwen/Qwen2.5-1.5B-Instruct", prompt: "Hi" });
    expect(code).toContain("from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec");
    expect(code).toContain('source="vectors/happy.gguf"');
    expect(code).toContain("scale=2");
    expect(code).toContain("list(range(10, 13))");
    expect(code).toContain('ApplySpec(prompt="all", generation="all")');
    expect(code).toContain("enable_steer_vector=True");
    expect(code).toContain('steer_algorithms=["direct"]');
    expect(code).toContain("steering=spec");
    expect(code).toContain("tokenizer = llm.get_tokenizer()");
    expect(code).toContain("if tokenizer.chat_template:");
    expect(code).toContain('prompt = {"prompt_token_ids": tokenizer.apply_chat_template(');
    expect(code).toContain("tokenize=True, return_dict=False");
    expect(code).toContain('else:\n    prompt = "Hi"');
    expect(code).toContain("llm.generate([prompt], sampling, steering=spec)");
    expect(code).not.toContain("llm.chat(");
    expect(code).not.toContain("normalize");
    expect(code).not.toContain("conflict");
    expect(code).not.toContain("steer_multi_vector");
  });

  it("emits multi-vector engine flags, conflict and per-vector clauses", () => {
    const spec = defaultSteeringSpec();
    spec.conflict = "sequential";
    spec.vectors = [1, 2].map((k) => ({
      ...defaultVectorSpec(),
      source: `diffmean-${k}.gguf`,
      scale: 2.0,
      layers: [0, 1],
      apply: { ...defaultApplySpec(), prompt_positions: [-k] },
    }));
    const code = toPython(spec);
    expect(code).toContain("steer_multi_vector=True");
    expect(code).toContain('conflict="sequential"');
    expect(code).toContain("prompt_positions=[-1]");
    expect(code).toContain("prompt_positions=[-2]");
  });

  it("marks in-memory payloads as a placeholder", () => {
    const spec = defaultSteeringSpec();
    spec.vectors[0].data = { __inline_payload__: "vec.from_training('./adapter/')" };
    spec.vectors[0].algorithm = "loreft";
    const code = toPython(spec);
    expect(code).toContain("data=...");
    expect(code).toContain("# Replace with vec.from_training('./adapter/')");
    expect(code).toContain("import easysteer.vectors as vec");
    expect(code).toContain('algorithm="loreft"');
  });

  it("preserves a native checkpoint's JSON payload", () => {
    const spec = defaultSteeringSpec();
    spec.vectors[0].data = {
      version: 1, kind: "direction", tensors: { "8": [1, -2] },
      extra: {}, sha256: "checkpoint-digest",
    };
    const code = toPython(spec);
    expect(code).toContain('data={"version": 1, "kind": "direction"');
    expect(code).toContain('"8": [1, -2]');
    expect(code).not.toContain("data=...");
  });

  it("uses the checkpoint prompt format for trained adapters", () => {
    const code = toPython(sampleSpec(), {
      model: "trained-model", prompt: "Hello", promptTemplate: "Training prompt: %s\nAnswer:",
    });
    expect(code).toContain('model="trained-model"');
    expect(code).toContain('prompt = "Training prompt: %s\\nAnswer:" % "Hello"');
    expect(code).not.toContain("apply_chat_template");
  });

  it("renders the new selectors and exclude twins", () => {
    const spec = defaultSteeringSpec();
    spec.vectors[0].source = "vec.gguf";
    spec.vectors[0].apply.prompt_window = [-5, null];
    spec.vectors[0].apply.generation_positions = [0, 1];
    spec.vectors[0].apply.exclude_generation_window = [4, 8];
    const code = toPython(spec);
    expect(code).toContain("prompt_window=(-5, None)");
    expect(code).toContain("generation_positions=[0, 1]");
    expect(code).toContain("exclude_generation_window=(4, 8)");
  });

  it("renders generation_window as a Python tuple", () => {
    const spec = defaultSteeringSpec();
    spec.vectors[0].source = "vec.gguf";
    spec.vectors[0].apply.prompt = null;
    spec.vectors[0].apply.generation_window = [0, 8];
    expect(toPython(spec)).toContain("generation_window=(0, 8)");
    spec.vectors[0].apply.generation_window = [2, null];
    expect(toPython(spec)).toContain("generation_window=(2, None)");
  });
});

describe("toCurl", () => {
  it("preserves saved templates through the completions endpoint", () => {
    const cmd = toCurl(sampleSpec(), {
      baseUrl: "http://server/v1", model: "served-model", prompt: "Hello",
      promptTemplate: "Training prompt: %s\nAnswer:",
    });
    expect(cmd).toContain("http://server/v1/completions");
    expect(cmd).toContain('"prompt": "Training prompt: Hello\\nAnswer:"');
    expect(cmd).not.toContain('"messages"');
  });
  it("targets the configured base URL and inlines the steering field", () => {
    const cmd = toCurl(sampleSpec(), {
      baseUrl: "http://gpu-box:8000/v1/",
      model: "qwen",
      prompt: "Hello",
    });
    expect(cmd).toContain("curl http://gpu-box:8000/v1/chat/completions");
    expect(cmd).toContain('"steering"');
    expect(cmd).toContain('"model": "qwen"');
  });
});
