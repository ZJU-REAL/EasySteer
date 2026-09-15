/**
 * Payload shapes shared by the Flask job client and stored presets.
 * This module has no browser globals, so presets can also use it outside the UI.
 */

import { defaultApplySpec, type ApplySpec } from "./spec";

export interface ExtractionConfig {
  model_path: string;
  gpu_devices?: string;
  method: "diffmean" | "pca" | "lat" | "incremental_pca";
  positive_samples: string[];
  negative_samples: string[];
  token_pos?: number | string;
  normalize?: boolean;
  max_working_bytes?: number;
  output_path: string;
}

export interface TrainingConfig {
  model_path: string;
  gpu_devices?: string;
  /** [input, output] pairs. */
  training_examples: [string, string][];
  algorithm?: "direct" | "loreft";
  output_dir: string;
  steering_config?: {
    layer?: number;
    component?: string;
    rank?: number;
    apply?: Partial<ApplySpec>;
  };
  training_args?: {
    num_train_epochs?: number;
    per_device_train_batch_size?: number;
    learning_rate?: number;
    logging_steps?: number;
  };
}

/** Expand optional fields without adding selectors to the training request. */
export function trainingApply(selection?: Partial<ApplySpec>): ApplySpec {
  return {
    ...defaultApplySpec(),
    prompt: null,
    generation: null,
    ...JSON.parse(JSON.stringify(selection ?? { prompt_positions: [-1] })),
  };
}
