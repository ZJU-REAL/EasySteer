/**
 * Client-side mirror of the steering spec schema defined in
 * `vllm-steer/vllm/steer_vectors/api.py` (SteeringSpec / VectorSpec /
 * ApplySpec). Field names, defaults and validation rules must match the
 * server exactly; when the Python schema changes, change this file too.
 */

export type Phase = "prompt" | "generation";

const PHASES: Phase[] = ["prompt", "generation"];

/** Per-phase widest include selector: "all" selects the whole phase. */
export type PhaseAll = "all" | null;

export type ConflictPolicy = "priority" | "sequential" | "error";

export const CONFLICT_POLICIES: ConflictPolicy[] = ["priority", "sequential", "error"];

export const ALGORITHMS = [
  "direct",
  "linear",
  "loreft",
  "lm_steer",
  "erase",
  "replace",
  "concept_replace",
  "moe_router",
] as const;

export type Algorithm = (typeof ALGORITHMS)[number];

/** Allowed `params` keys per algorithm; unlisted algorithms accept none. */
const ALGORITHM_PARAMS: Record<string, string[]> = {
  moe_router: ["expert_ids", "mode", "lambda", "topk"],
};

/** Algorithms that never load `source` files (in-memory payloads only). */
const DATA_ONLY_ALGORITHMS = ["linear", "lm_steer", "loreft"];

/** Algorithms whose `source` must be an EasySteer GGUF export. */
const GGUF_ONLY_ALGORITHMS = ["direct", "erase", "replace"];

/** Half-open (start, stop) window; stop=null is open-ended. */
export type SpecWindow = [number, number | null];

export interface ApplySpec {
  prompt: PhaseAll;
  generation: PhaseAll;
  prompt_tokens: number[] | null;
  prompt_positions: number[] | null;
  prompt_window: SpecWindow | null;
  generation_tokens: number[] | null;
  generation_positions: number[] | null;
  generation_window: SpecWindow | null;
  exclude_prompt_tokens: number[] | null;
  exclude_prompt_positions: number[] | null;
  exclude_prompt_window: SpecWindow | null;
  exclude_generation_tokens: number[] | null;
  exclude_generation_positions: number[] | null;
  exclude_generation_window: SpecWindow | null;
}

export interface VectorSpec {
  source: string | null;
  /** In-memory payload wire dict; the browser UI treats it as opaque. */
  data: unknown;
  algorithm: string;
  scale: number;
  layers: number[] | null;
  normalize: boolean;
  apply: ApplySpec;
  params: Record<string, unknown>;
  name: string | null;
}

export interface SteeringSpec {
  vectors: VectorSpec[];
  conflict: ConflictPolicy;
}

export function defaultApplySpec(): ApplySpec {
  return {
    prompt: "all",
    generation: "all",
    prompt_tokens: null,
    prompt_positions: null,
    prompt_window: null,
    generation_tokens: null,
    generation_positions: null,
    generation_window: null,
    exclude_prompt_tokens: null,
    exclude_prompt_positions: null,
    exclude_prompt_window: null,
    exclude_generation_tokens: null,
    exclude_generation_positions: null,
    exclude_generation_window: null,
  };
}

export function defaultVectorSpec(): VectorSpec {
  return {
    source: null,
    data: null,
    algorithm: "direct",
    scale: 1.0,
    layers: null,
    normalize: false,
    apply: defaultApplySpec(),
    params: {},
    name: null,
  };
}

export function defaultSteeringSpec(): SteeringSpec {
  return {
    vectors: [defaultVectorSpec()],
    conflict: "priority",
  };
}

export interface SpecIssue {
  /** JSON-path-ish location, e.g. "vectors[0].apply.phases". */
  path: string;
  message: string;
}

function isIntArray(v: unknown): v is number[] {
  return Array.isArray(v) && v.every((x) => Number.isInteger(x));
}

function checkNullableIntList(
  issues: SpecIssue[],
  path: string,
  name: string,
  value: unknown,
): void {
  if (value === null || value === undefined) return;
  if (!isIntArray(value)) {
    issues.push({ path: `${path}.${name}`, message: `${name} must be a list of integers or null` });
    return;
  }
  if (value.length === 0) {
    issues.push({
      path: `${path}.${name}`,
      message: `${name} must be null or non-empty (null disables the filter)`,
    });
  }
}

/** Structural check for a window value; returns the tuple or null if bad. */
function checkWindowShape(
  issues: SpecIssue[],
  path: string,
  name: string,
  value: unknown,
): SpecWindow | null {
  if (value === null || value === undefined) return null;
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    !Number.isInteger(value[0]) ||
    (value[1] !== null && !Number.isInteger(value[1]))
  ) {
    issues.push({
      path: `${path}.${name}`,
      message: `${name} must be [start, stop] with integer start and integer or null stop`,
    });
    return null;
  }
  return value as SpecWindow;
}

/** List selectors, in wire order. */
const LIST_SELECTORS = [
  "prompt_tokens",
  "prompt_positions",
  "generation_tokens",
  "generation_positions",
  "exclude_prompt_tokens",
  "exclude_prompt_positions",
  "exclude_generation_tokens",
  "exclude_generation_positions",
] as const;

const TOKEN_SELECTORS = [
  "prompt_tokens",
  "generation_tokens",
  "exclude_prompt_tokens",
  "exclude_generation_tokens",
] as const;

const WINDOW_SELECTORS = [
  "prompt_window",
  "generation_window",
  "exclude_prompt_window",
  "exclude_generation_window",
] as const;

const INCLUDE_SELECTORS = [
  "prompt_tokens",
  "prompt_positions",
  "prompt_window",
  "generation_tokens",
  "generation_positions",
  "generation_window",
] as const;

const EXCLUDE_SELECTORS = [
  "exclude_prompt_tokens",
  "exclude_prompt_positions",
  "exclude_prompt_window",
  "exclude_generation_tokens",
  "exclude_generation_positions",
  "exclude_generation_window",
] as const;

/** The phase a selector's name binds it to. */
function selectorPhase(name: string): Phase {
  return name.includes("generation") ? "generation" : "prompt";
}

/** Whether the clause selects anything in the given phase. */
function covers(apply: ApplySpec, phase: Phase): boolean {
  if (apply[phase] === "all") return true;
  return INCLUDE_SELECTORS.some(
    (name) => selectorPhase(name) === phase && apply[name] !== null && apply[name] !== undefined,
  );
}

export function validateApplySpec(apply: ApplySpec, path = "apply"): SpecIssue[] {
  const issues: SpecIssue[] = [];
  for (const phase of PHASES) {
    const value = apply[phase];
    if (value !== null && value !== undefined && value !== "all") {
      issues.push({
        path: `${path}.${phase}`,
        message: `${phase} must be "all" or null, got ${JSON.stringify(value)}`,
      });
    }
  }
  if (!covers(apply, "prompt") && !covers(apply, "generation")) {
    issues.push({
      path,
      message:
        'the clause selects nothing: set prompt="all" / generation="all" or at least one include selector',
    });
  }
  for (const phase of PHASES) {
    const excludes = EXCLUDE_SELECTORS.filter(
      (name) => selectorPhase(name) === phase && apply[name] !== null && apply[name] !== undefined,
    );
    if (excludes.length > 0 && !covers(apply, phase)) {
      issues.push({
        path: `${path}.${excludes[0]}`,
        message: `${excludes.join(", ")} exclude ${phase} tokens, but the clause selects none; set ${phase}="all" or a ${phase}_* selector`,
      });
    }
  }

  for (const name of LIST_SELECTORS) {
    checkNullableIntList(issues, path, name, apply[name]);
  }
  for (const name of TOKEN_SELECTORS) {
    const ids = apply[name];
    if (isIntArray(ids) && ids.some((t) => t < 0)) {
      issues.push({
        path: `${path}.${name}`,
        message: `${name} must contain real token ids (>= 0); use prompt="all" / generation="all" instead of the -1 sentinel`,
      });
    }
  }
  for (const name of ["generation_positions", "exclude_generation_positions"] as const) {
    const steps = apply[name];
    if (isIntArray(steps) && steps.some((j) => j < 0)) {
      issues.push({
        path: `${path}.${name}`,
        message:
          `${name} must contain 0-based decode steps (>= 0); the generation ` +
          "length is not known up front, so end-relative steps cannot resolve",
      });
    }
  }

  for (const name of ["prompt_window", "exclude_prompt_window"] as const) {
    const window = checkWindowShape(issues, path, name, apply[name]);
    if (window === null) continue;
    const [start, stop] = window;
    // stop > start is enforced only when both bounds share a sign;
    // mixed-sign bounds (e.g. [2, -2]) resolve against the prompt length.
    if (stop !== null && start < 0 === stop < 0 && stop <= start) {
      issues.push({
        path: `${path}.${name}`,
        message:
          `${name} must be a half-open [start, stop] with stop > start ` +
          `(or stop=null for the prompt end), got [${start}, ${stop}]`,
      });
    }
  }
  for (const name of ["generation_window", "exclude_generation_window"] as const) {
    const window = checkWindowShape(issues, path, name, apply[name]);
    if (window === null) continue;
    const [start, stop] = window;
    if (start < 0) {
      issues.push({ path: `${path}.${name}`, message: `${name} start must be >= 0, got ${start}` });
    }
    if (stop !== null && stop <= start) {
      issues.push({
        path: `${path}.${name}`,
        message: `${name} must be half-open [start, stop] with stop > start or stop=null, got [${start}, ${stop}]`,
      });
    }
  }
  return issues;
}

export function validateVectorSpec(vector: VectorSpec, path = "vector"): SpecIssue[] {
  const issues: SpecIssue[] = [];
  if (vector.layers !== null && vector.layers !== undefined) {
    if (!isIntArray(vector.layers)) {
      issues.push({ path: `${path}.layers`, message: "layers must be a list of integers or null" });
    } else if (vector.layers.length === 0) {
      issues.push({ path: `${path}.layers`, message: "layers must be null or non-empty" });
    }
  }
  if (typeof vector.scale !== "number" || !Number.isFinite(vector.scale)) {
    issues.push({ path: `${path}.scale`, message: "scale must be a finite number" });
  }
  const algo = vector.algorithm;
  if (!ALGORITHMS.includes(algo as Algorithm)) {
    issues.push({
      path: `${path}.algorithm`,
      message: `unknown algorithm '${algo}' (known: ${ALGORITHMS.join(", ")})`,
    });
  }
  if (vector.source !== null && vector.source !== undefined && vector.source.includes("|")) {
    issues.push({
      path: `${path}.source`,
      message: "source must be a plain path; set the algorithm via the algorithm field, not 'path|algo'",
    });
  }
  const allowed = ALGORITHM_PARAMS[algo] ?? [];
  const unknown = Object.keys(vector.params ?? {}).filter((k) => !allowed.includes(k));
  if (unknown.length > 0) {
    issues.push({
      path: `${path}.params`,
      message: `unknown params for algorithm '${algo}': ${unknown.sort().join(", ")} (allowed: ${
        allowed.length ? allowed.slice().sort().join(", ") : "none"
      })`,
    });
  }
  const hasSource = vector.source !== null && vector.source !== undefined && vector.source !== "";
  const hasData = vector.data !== null && vector.data !== undefined;
  if (hasSource && hasData) {
    issues.push({ path, message: "source and data are mutually exclusive" });
  }
  if (algo === "moe_router" && !hasSource) {
    if (!vector.params || !vector.params["expert_ids"]) {
      issues.push({
        path: `${path}.params`,
        message: "moe_router without a source file requires params['expert_ids']",
      });
    }
    if (!vector.layers || vector.layers.length === 0) {
      issues.push({
        path: `${path}.layers`,
        message: "moe_router without a source file requires layers (the layers whose experts to steer)",
      });
    }
  }
  if (algo !== "moe_router" && !hasSource && !hasData) {
    issues.push({
      path,
      message: `algorithm '${algo}' requires either a source file or an in-memory data payload`,
    });
  }
  if (hasSource && vector.source) {
    if (DATA_ONLY_ALGORITHMS.includes(algo)) {
      issues.push({
        path: `${path}.source`,
        message: `algorithm '${algo}' loads no source files; load the checkpoint yourself and pass data=...`,
      });
    }
    if (GGUF_ONLY_ALGORITHMS.includes(algo) && !vector.source.toLowerCase().endsWith(".gguf")) {
      issues.push({
        path: `${path}.source`,
        message: `algorithm '${algo}' accepts only .gguf sources (EasySteer's export format); for other formats pass data=...`,
      });
    }
  }
  issues.push(...validateApplySpec(vector.apply, `${path}.apply`));
  return issues;
}

export function validateSteeringSpec(spec: SteeringSpec): SpecIssue[] {
  const issues: SpecIssue[] = [];
  if (!Array.isArray(spec.vectors) || spec.vectors.length === 0) {
    issues.push({ path: "vectors", message: "vectors must be non-empty" });
    return issues;
  }
  if (!CONFLICT_POLICIES.includes(spec.conflict)) {
    issues.push({
      path: "conflict",
      message: `conflict must be one of: ${CONFLICT_POLICIES.join(", ")}`,
    });
  }
  if (spec.vectors.length > 1 && spec.vectors.some((v) => v.algorithm === "moe_router")) {
    issues.push({
      path: "vectors",
      message: "moe_router is not supported in multi-vector specs yet; use a single-vector spec",
    });
  }
  spec.vectors.forEach((v, i) => issues.push(...validateVectorSpec(v, `vectors[${i}]`)));
  return issues;
}

/**
 * Serialize a spec to the canonical JSON accepted by the server, dropping
 * fields that hold their default value so the JSON panel stays readable.
 * The server's pydantic models fill the same defaults back in.
 */
export function specToJson(spec: SteeringSpec): Record<string, unknown> {
  const out: Record<string, unknown> = {
    vectors: spec.vectors.map(vectorToJson),
  };
  if (spec.conflict !== "priority") out.conflict = spec.conflict;
  return out;
}

function vectorToJson(v: VectorSpec): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (v.source !== null && v.source !== undefined && v.source !== "") out.source = v.source;
  if (v.data !== null && v.data !== undefined) out.data = v.data;
  if (v.algorithm !== "direct") out.algorithm = v.algorithm;
  if (v.scale !== 1.0) out.scale = v.scale;
  if (v.layers && v.layers.length > 0) out.layers = v.layers;
  if (v.normalize) out.normalize = v.normalize;
  out.apply = applyToJson(v.apply);
  if (v.params && Object.keys(v.params).length > 0) out.params = v.params;
  if (v.name) out.name = v.name;
  return out;
}

/** Wire key order of a selection clause (mirrors SelectSpec.to_wire()). */
const APPLY_KEYS = [
  "prompt",
  "generation",
  "prompt_tokens",
  "prompt_positions",
  "prompt_window",
  "generation_tokens",
  "generation_positions",
  "generation_window",
  "exclude_prompt_tokens",
  "exclude_prompt_positions",
  "exclude_prompt_window",
  "exclude_generation_tokens",
  "exclude_generation_positions",
  "exclude_generation_window",
] as const;

function applyToJson(a: ApplySpec): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const key of APPLY_KEYS) {
    const value = a[key];
    if (value !== null && value !== undefined) out[key] = value;
  }
  return out;
}

/**
 * Parse arbitrary JSON (e.g. from the editable JSON panel) into a fully
 * populated SteeringSpec, filling schema defaults. Throws on structural
 * errors; semantic errors are left to `validateSteeringSpec`.
 */
export function specFromJson(json: unknown): SteeringSpec {
  if (typeof json !== "object" || json === null || Array.isArray(json)) {
    throw new Error("steering spec must be a JSON object");
  }
  const obj = json as Record<string, unknown>;
  // "debug" existed until the engine dropped it as dead; old JSON
  // still parses, the flag is simply discarded.
  const knownKeys = ["vectors", "conflict", "debug"];
  const unknown = Object.keys(obj).filter((k) => !knownKeys.includes(k));
  if (unknown.length > 0) {
    throw new Error(`unknown steering spec fields: ${unknown.sort().join(", ")}`);
  }
  if (!Array.isArray(obj.vectors)) {
    throw new Error("vectors must be a list");
  }
  return {
    vectors: obj.vectors.map((v, i) => vectorFromJson(v, i)),
    conflict: (obj.conflict as ConflictPolicy) ?? "priority",
  };
}

function vectorFromJson(json: unknown, index: number): VectorSpec {
  if (typeof json !== "object" || json === null || Array.isArray(json)) {
    throw new Error(`vectors[${index}] must be a JSON object`);
  }
  const obj = json as Record<string, unknown>;
  const knownKeys = [
    "source",
    "data",
    "algorithm",
    "scale",
    "layers",
    "normalize",
    "apply",
    "params",
    "name",
  ];
  const unknown = Object.keys(obj).filter((k) => !knownKeys.includes(k));
  if (unknown.length > 0) {
    throw new Error(`unknown vector fields at vectors[${index}]: ${unknown.sort().join(", ")}`);
  }
  if (obj.apply === undefined || obj.apply === null) {
    throw new Error(`vectors[${index}].apply is required`);
  }
  return {
    source: (obj.source as string | null) ?? null,
    data: obj.data ?? null,
    algorithm: (obj.algorithm as string) ?? "direct",
    scale: (obj.scale as number) ?? 1.0,
    layers: (obj.layers as number[] | null) ?? null,
    normalize: Boolean(obj.normalize ?? false),
    apply: applyFromJson(obj.apply, index),
    params: (obj.params as Record<string, unknown>) ?? {},
    name: (obj.name as string | null) ?? null,
  };
}

function applyFromJson(json: unknown, index: number): ApplySpec {
  if (typeof json !== "object" || json === null || Array.isArray(json)) {
    throw new Error(`vectors[${index}].apply must be a JSON object`);
  }
  const obj = json as Record<string, unknown>;
  const unknown = Object.keys(obj).filter((k) => !(APPLY_KEYS as readonly string[]).includes(k));
  if (unknown.includes("phases")) {
    throw new Error(
      `vectors[${index}].apply.phases was removed: write prompt="all" / generation="all" (or per-phase selectors, which imply their phase) instead`,
    );
  }
  if (unknown.length > 0) {
    throw new Error(`unknown apply fields at vectors[${index}].apply: ${unknown.sort().join(", ")}`);
  }

  function windowOf(key: string): SpecWindow | null {
    const value = obj[key];
    if (value === undefined || value === null) return null;
    if (!Array.isArray(value) || value.length !== 2) {
      throw new Error(`vectors[${index}].apply.${key} must be [start, stop]`);
    }
    return [value[0] as number, (value[1] as number | null) ?? null];
  }

  return {
    prompt: (obj.prompt as PhaseAll) ?? null,
    generation: (obj.generation as PhaseAll) ?? null,
    prompt_tokens: (obj.prompt_tokens as number[] | null) ?? null,
    prompt_positions: (obj.prompt_positions as number[] | null) ?? null,
    prompt_window: windowOf("prompt_window"),
    generation_tokens: (obj.generation_tokens as number[] | null) ?? null,
    generation_positions: (obj.generation_positions as number[] | null) ?? null,
    generation_window: windowOf("generation_window"),
    exclude_prompt_tokens: (obj.exclude_prompt_tokens as number[] | null) ?? null,
    exclude_prompt_positions: (obj.exclude_prompt_positions as number[] | null) ?? null,
    exclude_prompt_window: windowOf("exclude_prompt_window"),
    exclude_generation_tokens: (obj.exclude_generation_tokens as number[] | null) ?? null,
    exclude_generation_positions: (obj.exclude_generation_positions as number[] | null) ?? null,
    exclude_generation_window: windowOf("exclude_generation_window"),
  };
}

/** Deep-clone a spec (form state must never alias gallery presets). */
export function cloneSpec(spec: SteeringSpec): SteeringSpec {
  return specFromJson(specToJson(spec));
}

/**
 * Parse a UI string like "0-27" or "16,17,18" or "8" into a layer list.
 * Empty string means null (let the vector file decide).
 */
export function parseIntListString(text: string): number[] | null {
  const trimmed = text.trim();
  if (trimmed === "") return null;
  const out: number[] = [];
  for (const part of trimmed.split(",")) {
    const p = part.trim();
    if (p === "") continue;
    // Range syntax "a-b" (inclusive); negative numbers are plain ints.
    const range = p.match(/^(-?\d+)\s*-\s*(\d+)$/);
    if (range && !p.startsWith("-")) {
      const start = parseInt(range[1], 10);
      const stop = parseInt(range[2], 10);
      if (stop < start) throw new Error(`invalid range '${p}'`);
      for (let i = start; i <= stop; i++) out.push(i);
      continue;
    }
    const n = Number(p);
    if (!Number.isInteger(n)) throw new Error(`invalid integer '${p}'`);
    out.push(n);
  }
  return out.length > 0 ? out : null;
}

/** Format a layer list back into the compact "a-b" string form. */
export function formatIntList(values: number[] | null): string {
  if (!values || values.length === 0) return "";
  const parts: string[] = [];
  let i = 0;
  while (i < values.length) {
    let j = i;
    while (
      j + 1 < values.length &&
      values[j + 1] === values[j] + 1 &&
      values[i] >= 0
    ) {
      j += 1;
    }
    if (j - i >= 2) {
      parts.push(`${values[i]}-${values[j]}`);
    } else {
      for (let k = i; k <= j; k++) parts.push(String(values[k]));
    }
    i = j + 1;
  }
  return parts.join(",");
}
