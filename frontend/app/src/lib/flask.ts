/**
 * Client for the retained Flask job backend (vector extraction and
 * training). Endpoints and payload shapes match `frontend/extraction_api.py`
 * and `frontend/training_api.py` as-is.
 */

import { settings } from "./settings";

export interface ExtractionConfig {
  model_path: string;
  gpu_devices?: string;
  method: "diffmean" | "pca" | "lat";
  positive_samples: string[];
  negative_samples: string[];
  token_pos?: number | string;
  normalize?: boolean;
  output_path: string;
}

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

export interface TrainingConfig {
  model_path: string;
  gpu_devices?: string;
  /** [input, output] pairs. */
  training_examples: [string, string][];
  intervention?: string;
  output_dir: string;
  reft_config?: {
    layer?: number;
    component?: string;
    low_rank_dimension?: number;
  };
  training_args?: {
    num_train_epochs?: number;
    per_device_train_batch_size?: number;
    learning_rate?: number;
    logging_steps?: number;
  };
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

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(`${base()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const json = await resp.json();
  if (!resp.ok) {
    throw new Error(json.error ?? `HTTP ${resp.status}`);
  }
  return json as T;
}

async function getJson<T>(path: string): Promise<T> {
  const resp = await fetch(`${base()}${path}`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return (await resp.json()) as T;
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

export function getTrainingConfig(name: string): Promise<TrainingConfig> {
  return getJson(`/api/train-config/${encodeURIComponent(name)}`);
}
