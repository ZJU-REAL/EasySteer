import { describe, expect, it } from "vitest";
import {
  cloneSpec,
  defaultApplySpec,
  defaultSteeringSpec,
  defaultVectorSpec,
  formatIntList,
  parseIntListString,
  specFromJson,
  specToJson,
  validateApplySpec,
  validateSteeringSpec,
  validateVectorSpec,
} from "./spec";

function ggufVector(overrides: Record<string, unknown> = {}) {
  return {
    ...defaultVectorSpec(),
    source: "vec.gguf",
    ...overrides,
  };
}

describe("validateApplySpec", () => {
  it("accepts the default apply clause", () => {
    expect(validateApplySpec(defaultApplySpec())).toEqual([]);
  });

  it("rejects empty phases", () => {
    const apply = { ...defaultApplySpec(), phases: [] as never[] };
    expect(validateApplySpec(apply).map((i) => i.message)).toContain("phases must be non-empty");
  });

  it("rejects duplicate phases", () => {
    const apply = { ...defaultApplySpec(), phases: ["prompt", "prompt"] as never };
    expect(validateApplySpec(apply).length).toBeGreaterThan(0);
  });

  it("rejects empty filter lists (null disables the filter)", () => {
    const apply = { ...defaultApplySpec(), prompt_tokens: [] };
    expect(validateApplySpec(apply).length).toBe(1);
    expect(validateApplySpec({ ...defaultApplySpec(), prompt_tokens: null })).toEqual([]);
  });

  it("rejects the v1 -1 token sentinel", () => {
    const apply = { ...defaultApplySpec(), prompt_tokens: [-1] };
    expect(validateApplySpec(apply)[0].message).toMatch(/real token ids/);
  });

  it("allows negative positions (end-of-prompt indexing)", () => {
    const apply = { ...defaultApplySpec(), prompt_positions: [-1, -2] };
    expect(validateApplySpec(apply)).toEqual([]);
  });

  it("requires generation phase for generation_window", () => {
    const apply = {
      ...defaultApplySpec(),
      phases: ["prompt"] as never,
      generation_window: [0, 8] as [number, number],
    };
    expect(validateApplySpec(apply)[0].message).toMatch(/requires 'generation'/);
  });

  it("enforces half-open window with stop > start", () => {
    const bad = { ...defaultApplySpec(), generation_window: [4, 4] as [number, number] };
    expect(validateApplySpec(bad).length).toBe(1);
    const openEnded = {
      ...defaultApplySpec(),
      generation_window: [0, null] as [number, null],
    };
    expect(validateApplySpec(openEnded)).toEqual([]);
  });

  it("rejects negative window start", () => {
    const apply = { ...defaultApplySpec(), generation_window: [-1, 5] as [number, number] };
    expect(validateApplySpec(apply)[0].message).toMatch(/start must be >= 0/);
  });

  it("accepts valid prompt windows, including mixed-sign bounds", () => {
    for (const window of [
      [-5, null],
      [0, 10],
      [2, -2],
    ] as const) {
      const apply = {
        ...defaultApplySpec(),
        prompt_window: [window[0], window[1]] as [number, number | null],
      };
      expect(validateApplySpec(apply), JSON.stringify(window)).toEqual([]);
    }
  });

  it("requires prompt phase for prompt_window", () => {
    const apply = {
      ...defaultApplySpec(),
      phases: ["generation"] as never,
      prompt_window: [0, 10] as [number, number],
    };
    expect(validateApplySpec(apply)[0].message).toMatch(/requires 'prompt'/);
  });

  it("rejects same-sign non-increasing prompt windows", () => {
    for (const window of [
      [-2, -4],
      [4, 4],
      [5, 3],
    ] as const) {
      const apply = {
        ...defaultApplySpec(),
        prompt_window: [window[0], window[1]] as [number, number],
      };
      expect(validateApplySpec(apply).length, JSON.stringify(window)).toBe(1);
      expect(validateApplySpec(apply)[0].message).toMatch(/stop > start/);
    }
  });

  it("accepts generation_positions of 0-based decode steps", () => {
    const apply = { ...defaultApplySpec(), generation_positions: [0, 1, 5] };
    expect(validateApplySpec(apply)).toEqual([]);
  });

  it("rejects negative generation_positions", () => {
    const apply = { ...defaultApplySpec(), generation_positions: [-1] };
    expect(
      validateApplySpec(apply).some((i) => i.message.includes("0-based decode steps")),
    ).toBe(true);
  });

  it("requires generation phase for generation_positions", () => {
    const apply = {
      ...defaultApplySpec(),
      phases: ["prompt"] as never,
      generation_positions: [0],
    };
    expect(validateApplySpec(apply)[0].message).toMatch(/requires 'generation'/);
  });

  it("validates exclude twins with the same rules as their includes", () => {
    // exclude_prompt_window: phase gate + same-sign ordering.
    const badPhase = {
      ...defaultApplySpec(),
      phases: ["generation"] as never,
      exclude_prompt_window: [0, 4] as [number, number],
    };
    expect(validateApplySpec(badPhase)[0].message).toMatch(
      /exclude_prompt_window requires 'prompt'/,
    );
    const mixedSign = {
      ...defaultApplySpec(),
      exclude_prompt_window: [2, -2] as [number, number],
    };
    expect(validateApplySpec(mixedSign)).toEqual([]);

    // exclude_generation_window: start >= 0, half-open.
    const negStart = {
      ...defaultApplySpec(),
      exclude_generation_window: [-1, 5] as [number, number],
    };
    expect(validateApplySpec(negStart)[0].message).toMatch(
      /exclude_generation_window start must be >= 0/,
    );

    // exclude_generation_positions: >= 0, phase-gated, non-empty.
    const negStep = { ...defaultApplySpec(), exclude_generation_positions: [-2] };
    expect(
      validateApplySpec(negStep).some((i) => i.path.endsWith("exclude_generation_positions")),
    ).toBe(true);
    const emptySteps = { ...defaultApplySpec(), exclude_generation_positions: [] };
    expect(validateApplySpec(emptySteps).length).toBe(1);
  });
});

describe("validateVectorSpec", () => {
  it("accepts a gguf-backed direct vector", () => {
    expect(validateVectorSpec(ggufVector())).toEqual([]);
  });

  it("requires source or data for non-moe algorithms", () => {
    const issues = validateVectorSpec(defaultVectorSpec());
    expect(issues[0].message).toMatch(/requires either a source file or an in-memory data payload/);
  });

  it("rejects source and data together", () => {
    const issues = validateVectorSpec(ggufVector({ data: { kind: "direction" } }));
    expect(issues.map((i) => i.message)).toContain("source and data are mutually exclusive");
  });

  it("rejects 'path|algo' sources", () => {
    const issues = validateVectorSpec(ggufVector({ source: "vec.gguf|direct" }));
    expect(issues.some((i) => i.message.includes("plain path"))).toBe(true);
  });

  it("rejects non-gguf sources for direct/erase/replace", () => {
    for (const algorithm of ["direct", "erase", "replace"]) {
      const issues = validateVectorSpec(ggufVector({ algorithm, source: "vec.pt" }));
      expect(issues.some((i) => i.message.includes("only .gguf sources"))).toBe(true);
    }
  });

  it("rejects source files for data-only algorithms", () => {
    for (const algorithm of ["linear", "lm_steer", "loreft"]) {
      const issues = validateVectorSpec(ggufVector({ algorithm, source: "ckpt.pt" }));
      expect(issues.some((i) => i.message.includes("loads no source files"))).toBe(true);
    }
  });

  it("rejects unknown params for non-moe algorithms", () => {
    const issues = validateVectorSpec(ggufVector({ params: { topk: 4 } }));
    expect(issues.some((i) => i.message.includes("unknown params"))).toBe(true);
  });

  it("accepts moe_router params and requires expert_ids/layers without a source", () => {
    const withSource = ggufVector({
      algorithm: "moe_router",
      source: "config.json",
      params: { mode: "deactivate" },
    });
    expect(validateVectorSpec(withSource)).toEqual([]);

    const withoutSource = ggufVector({
      algorithm: "moe_router",
      source: null,
      layers: null,
      params: {},
    });
    const messages = validateVectorSpec(withoutSource).map((i) => i.message);
    expect(messages.some((m) => m.includes("expert_ids"))).toBe(true);
    expect(messages.some((m) => m.includes("requires layers"))).toBe(true);
  });

  it("rejects empty layers (null lets the file decide)", () => {
    const issues = validateVectorSpec(ggufVector({ layers: [] }));
    expect(issues.some((i) => i.message.includes("layers"))).toBe(true);
  });
});

describe("validateSteeringSpec", () => {
  it("accepts the gallery-style single vector spec", () => {
    const spec = defaultSteeringSpec();
    spec.vectors[0].source = "happy.gguf";
    spec.vectors[0].scale = 2.0;
    spec.vectors[0].layers = [10, 11, 12];
    expect(validateSteeringSpec(spec)).toEqual([]);
  });

  it("rejects empty vectors", () => {
    const spec = { ...defaultSteeringSpec(), vectors: [] };
    expect(validateSteeringSpec(spec)[0].message).toBe("vectors must be non-empty");
  });

  it("rejects moe_router in multi-vector specs", () => {
    const spec = defaultSteeringSpec();
    spec.vectors = [
      ggufVector(),
      ggufVector({ algorithm: "moe_router", source: "cfg.json" }),
    ];
    expect(
      validateSteeringSpec(spec).some((i) => i.message.includes("multi-vector")),
    ).toBe(true);
  });

  it("prefixes issue paths with the vector index", () => {
    const spec = defaultSteeringSpec();
    spec.vectors = [ggufVector(), ggufVector({ source: "bad.pt" })];
    const issue = validateSteeringSpec(spec).find((i) => i.path.startsWith("vectors[1]"));
    expect(issue).toBeDefined();
  });
});

describe("specToJson / specFromJson round-trip", () => {
  it("drops default fields and restores them on parse", () => {
    const spec = defaultSteeringSpec();
    spec.vectors[0].source = "vec.gguf";
    const json = specToJson(spec);
    expect(json).toEqual({ vectors: [{ source: "vec.gguf", apply: { phases: ["prompt", "generation"] } }] });
    const restored = specFromJson(json);
    expect(restored).toEqual(spec);
  });

  it("round-trips a fully populated multi-vector spec", () => {
    const spec = defaultSteeringSpec();
    spec.conflict = "sequential";
    spec.vectors = [1, 2].map((k) => ({
      ...defaultVectorSpec(),
      source: `diffmean-${k}.gguf`,
      scale: 2.0,
      layers: [0, 1, 2],
      normalize: true,
      name: `v${k}`,
      apply: {
        ...defaultApplySpec(),
        phases: ["prompt"],
        prompt_positions: [-k],
      },
    }));
    const restored = specFromJson(specToJson(spec));
    expect(restored).toEqual(spec);
    expect(validateSteeringSpec(restored)).toEqual([]);
  });

  it("round-trips generation_window including null stop", () => {
    const spec = defaultSteeringSpec();
    spec.vectors[0].source = "vec.gguf";
    spec.vectors[0].apply.phases = ["generation"];
    spec.vectors[0].apply.generation_window = [0, null];
    const restored = specFromJson(specToJson(spec));
    expect(restored.vectors[0].apply.generation_window).toEqual([0, null]);
  });

  it("round-trips every new selector and its exclude twin", () => {
    const spec = defaultSteeringSpec();
    spec.vectors[0].source = "vec.gguf";
    spec.vectors[0].apply = {
      ...defaultApplySpec(),
      prompt_window: [-5, null],
      generation_positions: [0, 3],
      generation_window: [0, 16],
      prompt_tokens: [7],
      generation_tokens: [8],
      exclude_prompt_tokens: [42],
      exclude_prompt_positions: [-1],
      exclude_generation_tokens: [42],
      exclude_prompt_window: [2, -2],
      exclude_generation_positions: [1],
      exclude_generation_window: [4, null],
    };
    const restored = specFromJson(specToJson(spec));
    expect(restored).toEqual(spec);
    expect(validateSteeringSpec(restored)).toEqual([]);
  });

  it("rejects unknown selector names in apply", () => {
    expect(() =>
      specFromJson({
        vectors: [{ source: "v.gguf", apply: { phases: ["prompt"], prompt_windows: [0, 4] } }],
      }),
    ).toThrow(/unknown apply fields/);
  });

  it("rejects unknown fields at every level (extra=forbid)", () => {
    expect(() => specFromJson({ vectors: [], bogus: 1 })).toThrow(/unknown steering spec fields/);
    expect(() =>
      specFromJson({ vectors: [{ apply: { phases: ["prompt"] }, bogus: 1 }] }),
    ).toThrow(/unknown vector fields/);
    expect(() =>
      specFromJson({ vectors: [{ apply: { phases: ["prompt"], bogus: 1 } }] }),
    ).toThrow(/unknown apply fields/);
  });

  it("requires apply on every vector", () => {
    expect(() => specFromJson({ vectors: [{ source: "v.gguf" }] })).toThrow(/apply is required/);
  });

  it("cloneSpec produces an independent copy", () => {
    const spec = defaultSteeringSpec();
    spec.vectors[0].source = "vec.gguf";
    const copy = cloneSpec(spec);
    copy.vectors[0].scale = 9;
    expect(spec.vectors[0].scale).toBe(1.0);
  });
});

describe("parseIntListString / formatIntList", () => {
  it("parses comma lists and ranges", () => {
    expect(parseIntListString("16,17,18")).toEqual([16, 17, 18]);
    expect(parseIntListString("0-3")).toEqual([0, 1, 2, 3]);
    expect(parseIntListString("0-2,5,7-8")).toEqual([0, 1, 2, 5, 7, 8]);
  });

  it("parses negative values as plain ints, not ranges", () => {
    expect(parseIntListString("-1")).toEqual([-1]);
    expect(parseIntListString("-4,-3,-2,-1")).toEqual([-4, -3, -2, -1]);
  });

  it("returns null for empty input", () => {
    expect(parseIntListString("")).toBeNull();
    expect(parseIntListString("  ")).toBeNull();
  });

  it("throws on junk", () => {
    expect(() => parseIntListString("abc")).toThrow();
    expect(() => parseIntListString("3-1")).toThrow();
    expect(() => parseIntListString("1.5")).toThrow();
  });

  it("formats runs compactly and round-trips", () => {
    for (const values of [
      [16, 17, 18, 19],
      [0, 1, 2, 5, 7, 8],
      [-4, -3, -2, -1],
      [11],
    ]) {
      expect(parseIntListString(formatIntList(values))).toEqual(values);
    }
  });
});
