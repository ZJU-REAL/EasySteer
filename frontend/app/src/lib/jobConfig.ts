/**
 * Payload shapes shared by the Flask job client and stored presets.
 * This module has no browser globals, so presets can also use it outside the UI.
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
