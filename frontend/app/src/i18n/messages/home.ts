/** Home landing page. */

import type { Messages } from "./types";

export const homeMessages = {
  home_tagline: {
    en: "A unified toolkit for steering large language models at inference time.",
    zh: "推理时引导大语言模型的一体化工具箱。",
  },
  home_intro: {
    en: "EasySteer builds on vllm-steer: declare a SteeringSpec — which vectors, which layers, which tokens — and apply it per request or server-wide. This UI lets you chat with steered models, build and validate specs, replicate published methods, and produce new vectors.",
    zh: "EasySteer 基于 vllm-steer：声明一个 SteeringSpec——哪些向量、哪些层、哪些 token——即可按请求或全服务器应用。本界面支持与被引导模型聊天、构建并校验 Spec、复现已发表方法以及生成新向量。",
  },
  home_quickstart_title: { en: "Quick start", zh: "快速开始" },
  home_chat_title: { en: "Chat", zh: "聊天" },
  home_chat_text: {
    en: "Talk to a model with a steering spec attached to every reply.",
    zh: "与模型对话，每条回复都可附加引导 Spec。",
  },
  home_playground_title: { en: "Playground", zh: "实验台" },
  home_playground_text: {
    en: "Build a SteeringSpec in a form or as JSON, A/B compare, export code.",
    zh: "以表单或 JSON 构建 SteeringSpec，A/B 对比并导出代码。",
  },
  home_gallery_title: { en: "Gallery", zh: "示例库" },
  home_gallery_text: {
    en: "15 replications of published steering papers, each one click from running.",
    zh: "15 个已发表引导论文的复现示例，一键载入运行。",
  },
  home_workshop_title: { en: "Workshop", zh: "向量工坊" },
  home_workshop_text: {
    en: "Extract or train new steering vectors on the job backend.",
    zh: "在任务后端提取或训练新的引导向量。",
  },
  home_sae_title: { en: "SAE features", zh: "SAE 特征" },
  home_sae_text: {
    en: "Search Neuronpedia features and steer along a decoder direction.",
    zh: "搜索 Neuronpedia 特征并沿解码器方向进行引导。",
  },
  home_featured_title: { en: "Pick a demo", zh: "挑一个示例" },
  home_featured_more: { en: "Browse all 15 demos", zh: "浏览全部 15 个示例" },
} satisfies Messages;
