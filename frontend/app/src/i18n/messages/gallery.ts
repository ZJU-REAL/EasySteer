/** Demo gallery cards and the card detail modal. */

import type { Messages } from "./types";

export const galleryMessages = {
  gallery_title: { en: "Demo Gallery", zh: "示例库" },
  gallery_intro: {
    en: "Each card replicates a published steering method, backed by a notebook in replications/. Open one to see what it does, then load its spec into the Steer page.",
    zh: "每张卡片复现一篇论文中的引导方法，对应 replications/ 下的一个 notebook。打开卡片查看做法，再把它的 Spec 载入“引导”页。",
  },
  open_in_playground: { en: "Open in Steer", zh: "在“引导”页中打开" },
  gallery_model: { en: "Model", zh: "模型" },
  gallery_paper: { en: "Paper", zh: "论文" },
  gallery_prompt: { en: "Demo prompt", zh: "示例 prompt" },
  gallery_spec_preview: { en: "SteeringSpec", zh: "SteeringSpec" },
  gallery_how: { en: "How it steers", zh: "引导方式" },
  gallery_layers_chip: { en: "layers {layers}", zh: "层 {layers}" },
  gallery_vectors_chip: { en: "{n} vectors", zh: "{n} 个向量" },
  gallery_all_filter: { en: "All", zh: "全部" },
  gallery_count: { en: "{n} demos", zh: "{n} 个示例" },

  cat_safety: { en: "Safety", zh: "安全" },
  cat_reasoning: { en: "Reasoning", zh: "推理" },
  cat_style: { en: "Style", zh: "风格" },
  cat_knowledge: { en: "Knowledge", zh: "知识" },
  cat_persona: { en: "Persona", zh: "人格" },
  cat_experts: { en: "Experts", zh: "专家路由" },
} satisfies Messages;
