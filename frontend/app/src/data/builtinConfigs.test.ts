import { describe, expect, it } from "vitest";
import { builtinExtractionPresets, builtinTrainingPresets } from "./builtinConfigs";

describe("built-in job presets", () => {
  it("ships every extraction preset with samples on both sides", () => {
    expect(builtinExtractionPresets.length).toBeGreaterThan(0);
    for (const preset of builtinExtractionPresets) {
      expect(preset.name).not.toBe("");
      expect(preset.display_name).not.toBe("");
      expect(preset.config.model_path).not.toBe("");
      expect(preset.config.positive_samples.length).toBeGreaterThan(0);
      expect(preset.config.negative_samples.length).toBeGreaterThan(0);
      expect(["diffmean", "pca", "lat"]).toContain(preset.config.method);
    }
  });

  // Stored training presets use the legacy nested layout; the form reads
  // the flat one, so a missed field silently imports as an empty box.
  it("flattens the ReFT training presets the form actually reads", () => {
    const loreft = builtinTrainingPresets.find((p) => p.name === "emoji_loreft");
    expect(loreft).toBeDefined();
    expect(loreft!.config).toMatchObject({
      intervention: "loreft",
      output_dir: "./results/emoji_loreft",
      reft_config: { layer: 8, component: "block_output", low_rank_dimension: 4 },
      training_args: {
        num_train_epochs: 100,
        per_device_train_batch_size: 10,
        learning_rate: 0.004,
        logging_steps: 40,
      },
    });
    expect(loreft!.config.model_path).toContain("Qwen2.5-1.5B-Instruct");
    expect(loreft!.config.training_examples.length).toBeGreaterThan(0);
    expect(loreft!.config.training_examples[0]).toHaveLength(2);

    const bias = builtinTrainingPresets.find((p) => p.name === "emoji_bias");
    expect(bias?.config.intervention).toBe("bias");
    expect(bias?.config.reft_config?.low_rank_dimension).toBe(2);
  });
});
