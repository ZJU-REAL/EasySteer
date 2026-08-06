/**
 * Built-in chat steering presets, ported from the legacy frontend's
 * configs/chat/*.json (translated to canonical v2 spec JSON).
 *
 * Every preset references its vector by server-side path — including the
 * SAE decoder rows saved as .pt, which the vector store loads by path as
 * long as the spec names the target layer. The legacy `trigger_tokens:
 * [-1]` wildcard becomes "both phases, no selectors" (the whole phase).
 *
 * Presets are model-specific: the vector only means something for the
 * model it was extracted from, so each one carries that model id.
 */

import { range, type LocalizedText } from "./gallery";

export interface ChatPreset {
  id: string;
  label: LocalizedText;
  description: LocalizedText;
  /** Emoji shown in the preset list, as in the legacy chat sidebar. */
  icon: string;
  /** Model the vector was extracted for. */
  model: string;
  /** Canonical steering-spec JSON (parseable by specFromJson). */
  spec: Record<string, unknown>;
}

export const chatPresets: ChatPreset[] = [
  {
    id: "happy",
    label: { en: "Happy", zh: "开心" },
    description: {
      en: "DiffMean happiness direction on layers 10-23; replies turn upbeat.",
      zh: "第 10-23 层的 DiffMean 开心方向，回复变得轻快积极。",
    },
    icon: "😊",
    model: "Qwen/Qwen2.5-1.5B-Instruct",
    spec: {
      vectors: [
        {
          source: "vectors/happy_diffmean.gguf",
          algorithm: "direct",
          scale: 2.0,
          layers: range(10, 24),
          apply: { phases: ["prompt", "generation"] },
        },
      ],
    },
  },
  {
    id: "chinese",
    label: { en: "Chinese", zh: "中文" },
    description: {
      en: "SAE feature 2534 at layer 31; the model answers in Chinese.",
      zh: "第 31 层的 SAE 特征 2534，模型改用中文作答。",
    },
    icon: "🇨🇳",
    model: "google/gemma-2-9b-it",
    spec: {
      vectors: [
        {
          source: "vectors/2534.pt",
          algorithm: "direct",
          scale: 500,
          layers: [31],
          normalize: false,
          apply: { phases: ["prompt", "generation"] },
        },
      ],
    },
  },
  {
    id: "cat",
    label: { en: "Cat", zh: "猫" },
    description: {
      en: "SAE feature 15973 at layer 31; the model talks like a cat.",
      zh: "第 31 层的 SAE 特征 15973，模型说话带猫的口吻。",
    },
    icon: "🐱",
    model: "google/gemma-2-9b-it",
    spec: {
      vectors: [
        {
          source: "vectors/15973.pt",
          algorithm: "direct",
          scale: 500,
          layers: [31],
          normalize: false,
          apply: { phases: ["prompt", "generation"] },
        },
      ],
    },
  },
  {
    id: "refuse",
    label: { en: "Refusal", zh: "拒绝" },
    description: {
      en: "CAST refusal direction at scale -2 across all layers; the model declines requests.",
      zh: "CAST 拒绝方向，全部层缩放系数 -2，模型倾向拒绝请求。",
    },
    icon: "🚫",
    model: "Qwen/Qwen2.5-1.5B-Instruct",
    spec: {
      vectors: [
        {
          source: "replications/cast/refuse-pca.gguf",
          algorithm: "direct",
          scale: -2.0,
          layers: range(0, 28),
          normalize: true,
          apply: { phases: ["prompt", "generation"] },
        },
      ],
    },
  },
];
