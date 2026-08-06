/** Demo gallery cards and card detail drawer. */

import type { Messages } from "./types";

export const galleryMessages = {
  gallery_title: { en: "Demo Gallery", zh: "示例库" },
  gallery_intro: {
    en: "Each card replicates a published steering method from the replications/ notebooks. Expand a card for details, then open it in the playground.",
    zh: "每张卡片对应 replications/ 中一篇已发表论文的引导方法复现。展开卡片查看详情，然后在实验台中打开。",
  },
  open_in_playground: { en: "Open in playground", zh: "在实验台中打开" },
  gallery_model: { en: "Model", zh: "模型" },
  gallery_paper: { en: "Paper", zh: "论文" },
  gallery_prompt: { en: "Demo prompt", zh: "示例提示词" },
  gallery_spec_preview: { en: "SteeringSpec", zh: "SteeringSpec" },
  gallery_details: { en: "Details", zh: "详情" },
  gallery_layers_chip: { en: "layers {layers}", zh: "层 {layers}" },
  gallery_vectors_chip: { en: "{n} vectors", zh: "{n} 个向量" },
} satisfies Messages;
