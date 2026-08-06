/**
 * Client for the retained Flask job backend (vector extraction and
 * training). Endpoints and payload shapes match `frontend/extraction_api.py`
 * and `frontend/training_api.py` as-is.
 */

import {
  interventionFromName,
  normalizeTrainingConfig,
  type ExtractionConfig,
  type RawTrainingConfig,
  type TrainingConfig,
} from "./jobConfig";
import { settings } from "./settings";

export type {
  ExtractionConfig,
  RawTrainingConfig,
  TrainingConfig,
} from "./jobConfig";
export { interventionFromName, normalizeTrainingConfig } from "./jobConfig";

export interface ExtractionStatus {
  is_extracting: boolean;
  status_message: string;
  error_message: string | null;
  logs: string[];
  result: {
    output_path: string;
    layers_extracted: number;
    method: string;
    metadata: Record<string, unknown>;
  } | null;
}

export interface TrainingStatus {
  is_training: boolean;
  current_epoch: number;
  current_step: number;
  status_message: string;
  error_message: string;
  logs: string[];
}

function base(): string {
  return settings.flaskBaseUrl.replace(/\/+$/, "");
}

/**
 * Read a job-backend response as JSON.
 *
 * A non-JSON body means the request never reached Flask — typically the
 * origin serves only static files, or a proxy answered with an HTML
 * error page. Say that instead of letting the parser complain about
 * "<", which tells the user nothing about what to fix.
 */
async function readJson<T>(resp: Response, path: string): Promise<T> {
  const text = await resp.text();
  let json: { error?: string } & T;
  try {
    json = JSON.parse(text);
  } catch {
    const where = base() === "" ? window.location.origin : base();
    throw new Error(
      `${where}${path} did not return JSON (HTTP ${resp.status}). ` +
        "The Flask job backend does not seem to be running behind this origin.",
    );
  }
  if (!resp.ok) {
    throw new Error(json.error ?? `HTTP ${resp.status}`);
  }
  return json;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(`${base()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return readJson<T>(resp, path);
}

async function getJson<T>(path: string): Promise<T> {
  const resp = await fetch(`${base()}${path}`);
  return readJson<T>(resp, path);
}

export function startExtraction(config: ExtractionConfig): Promise<{ success: boolean }> {
  return postJson("/api/extract", config);
}

export function getExtractionStatus(): Promise<ExtractionStatus> {
  return getJson("/api/extract-status");
}

export function listExtractionConfigs(): Promise<{ configs: { name: string; display_name?: string }[] }> {
  return getJson("/api/extract-configs");
}

export function getExtractionConfig(name: string): Promise<ExtractionConfig> {
  return getJson(`/api/extract-config/${encodeURIComponent(name)}`);
}

export function startTraining(config: TrainingConfig): Promise<{ success: boolean }> {
  return postJson("/api/train", config);
}

export function getTrainingStatus(): Promise<TrainingStatus> {
  return getJson("/api/train-status");
}

export function listTrainingConfigs(): Promise<{ configs: { name: string; display_name?: string }[] }> {
  return getJson("/api/train-configs");
}

export async function getTrainingConfig(name: string): Promise<TrainingConfig> {
  const raw = await getJson<RawTrainingConfig & Partial<TrainingConfig>>(
    `/api/train-config/${encodeURIComponent(name)}`,
  );
  return normalizeTrainingConfig(raw, interventionFromName(name));
}

// ---- SAE feature exploration (Neuronpedia proxied by the Flask backend) ----

export interface SaeSearchResult {
  modelId: string;
  layer: string;
  index: string | number;
  description: string | null;
  explanationModelName: string | null;
  typeName: string | null;
  cosine_similarity: number | null;
}

export interface SaeFeatureDetails {
  basic_info: { modelId: string | null; layer: string | null; index: number | null };
  explanation: string | null;
  sparsity: number | null;
  top_activating_tokens: { token: string; activation_value: number }[];
  top_inhibiting_tokens: { token: string; activation_value: number }[];
  activation_example: { max_value: number; trigger_token: string; context: string } | null;
}

export interface SaeExtractedVector {
  name: string;
  feature_index: number;
  file_path: string;
  scale: number;
}

export function searchSaeFeatures(params: {
  model_id: string;
  sae_id: string;
  query: string;
  api_key: string;
}): Promise<{ success: boolean; results: SaeSearchResult[] }> {
  return postJson("/api/sae/search", params);
}

export function getSaeFeature(
  modelId: string,
  saeId: string,
  featureIndex: number,
  apiKey: string,
): Promise<{ success: boolean; feature: SaeFeatureDetails }> {
  const query = apiKey ? `?api_key=${encodeURIComponent(apiKey)}` : "";
  return getJson(
    `/api/sae/feature/${encodeURIComponent(modelId)}/${encodeURIComponent(saeId)}/${featureIndex}${query}`,
  );
}

export function extractSaeVector(params: {
  feature_index: number;
  vector_name: string;
  scale: number;
}): Promise<{ success: boolean; vector?: SaeExtractedVector; error?: string }> {
  return postJson("/api/sae/extract-vector", params);
}
