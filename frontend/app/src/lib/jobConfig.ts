/**
 * Payload shapes for the Flask job backend, plus the normalization the
 * stored presets need. Kept free of browser globals so both the HTTP
 * client and the bundled presets (and their tests) can import it.
 */

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

/**
 * Stored training presets use the legacy nested layout (model / training /
 * reft / data) while the POST body is flat, so every read normalizes.
 */
export interface RawTrainingConfig {
  model?: { path?: string; gpu_devices?: string };
  training?: {
    output_dir?: string;
    num_train_epochs?: number;
    per_device_train_batch_size?: number;
    learning_rate?: number;
    logging_steps?: number;
  };
  reft?: { layer?: number; component?: string; low_rank_dimension?: number };
  data?: { training_examples?: [string, string][] };
}

export function normalizeTrainingConfig(
  raw: RawTrainingConfig & Partial<TrainingConfig>,
  intervention: string,
): TrainingConfig {
  return {
    model_path: raw.model?.path ?? raw.model_path ?? "",
    gpu_devices: raw.model?.gpu_devices ?? raw.gpu_devices ?? "0",
    intervention: raw.intervention ?? intervention,
    output_dir: raw.training?.output_dir ?? raw.output_dir ?? "results/my_training",
    training_examples: raw.data?.training_examples ?? raw.training_examples ?? [],
    reft_config: {
      layer: raw.reft?.layer ?? raw.reft_config?.layer,
      component: raw.reft?.component ?? raw.reft_config?.component,
      low_rank_dimension: raw.reft?.low_rank_dimension ?? raw.reft_config?.low_rank_dimension,
    },
    training_args: {
      num_train_epochs: raw.training?.num_train_epochs ?? raw.training_args?.num_train_epochs,
      per_device_train_batch_size:
        raw.training?.per_device_train_batch_size ??
        raw.training_args?.per_device_train_batch_size,
      learning_rate: raw.training?.learning_rate ?? raw.training_args?.learning_rate,
      logging_steps: raw.training?.logging_steps ?? raw.training_args?.logging_steps,
    },
  };
}

/**
 * `intervention` is not stored in the preset files at all; it is inferred
 * from the preset name, which is how the legacy UI picked it too.
 */
export function interventionFromName(name: string): string {
  return name.includes("bias") ? "bias" : "loreft";
}
