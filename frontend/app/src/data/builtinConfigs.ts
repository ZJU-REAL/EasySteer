/**
 * Job presets shipped with the app.
 *
 * These import frontend/configs/*.json directly, so the built-in list and
 * the files the Flask backend serves over /api/*-config are the same data
 * — no copy to drift. They also mean the presets are usable when the job
 * backend is offline (e.g. a static review deployment); anything the
 * backend additionally offers is merged in at runtime.
 */

import emotionDiffmean from "../../../configs/extraction/emotion_diffmean.json";
import emotionLat from "../../../configs/extraction/emotion_lat.json";
import emotionPca from "../../../configs/extraction/emotion_pca.json";
import personalityDiffmean from "../../../configs/extraction/personality_diffmean.json";
import emojiBias from "../../../configs/training/emoji_bias.json";
import emojiLoreft from "../../../configs/training/emoji_loreft.json";
import {
  interventionFromName,
  normalizeTrainingConfig,
  type ExtractionConfig,
  type RawTrainingConfig,
  type TrainingConfig,
} from "../lib/jobConfig";

export interface BuiltinPreset<T> {
  name: string;
  display_name: string;
  config: T;
}

function extraction(raw: unknown): BuiltinPreset<ExtractionConfig> {
  const cfg = raw as ExtractionConfig & { config_name: string; display_name: string };
  return { name: cfg.config_name, display_name: cfg.display_name, config: cfg };
}

function training(name: string, display_name: string, raw: unknown): BuiltinPreset<TrainingConfig> {
  return {
    name,
    display_name,
    config: normalizeTrainingConfig(raw as RawTrainingConfig, interventionFromName(name)),
  };
}

export const builtinExtractionPresets: BuiltinPreset<ExtractionConfig>[] = [
  extraction(emotionDiffmean),
  extraction(emotionPca),
  extraction(emotionLat),
  extraction(personalityDiffmean),
];

export const builtinTrainingPresets: BuiltinPreset<TrainingConfig>[] = [
  training("emoji_loreft", "Emoji LoReFT training", emojiLoreft),
  training("emoji_bias", "Emoji bias training", emojiBias),
];
