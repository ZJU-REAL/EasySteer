/** Home landing page. */

import type { Messages } from "./types";

export const homeMessages = {
  home_quickstart_title: { en: "Quick start", zh: "快速开始" },
  home_chat_title: { en: "Chat", zh: "聊天" },
  home_chat_text: {
    en: "Talk to a steered model.",
    zh: "与引导后的模型对话。",
  },
  home_playground_title: { en: "Steer", zh: "引导" },
  home_playground_text: {
    en: "Build a spec, run it, export code.",
    zh: "配置 Spec、运行对比、导出代码。",
  },
  home_gallery_title: { en: "Gallery", zh: "示例库" },
  home_gallery_text: {
    en: "15 replicated steering papers.",
    zh: "15 篇引导论文的复现。",
  },
  home_workshop_title: { en: "Extract & Train", zh: "提取与训练" },
  home_workshop_text: {
    en: "Build a vector from your own samples.",
    zh: "用自己的样本做出引导向量。",
  },
  home_sae_title: { en: "SAE features", zh: "SAE 特征" },
  home_sae_text: {
    en: "Steer along an SAE feature.",
    zh: "沿 SAE 特征方向引导。",
  },
  home_featured_title: { en: "Pick a demo", zh: "选一个示例" },
  home_featured_more: { en: "Browse all demos", zh: "查看全部示例" },
  home_paper_title: {
    en: "A Unified Framework for High-Performance and Extensible LLM Steering",
    zh: "统一、高性能且易扩展的大语言模型引导框架",
  },
  home_resources_title: { en: "Project links", zh: "项目链接" },
  home_link_paper: { en: "Paper", zh: "论文" },
  home_link_paper_sub: { en: "arXiv 2509.25175", zh: "arXiv 2509.25175" },
  home_link_github: { en: "GitHub", zh: "GitHub" },
  home_link_github_sub: { en: "ZJU-REAL/EasySteer", zh: "ZJU-REAL/EasySteer" },
  home_link_docs: { en: "Documentation", zh: "文档" },
  home_link_docs_sub: {
    en: "Install, guides, API reference",
    zh: "安装、使用指南与 API 参考",
  },
  home_link_demo: { en: "Hugging Face demo", zh: "Hugging Face 在线体验" },
  home_link_demo_sub: { en: "Lite hosted playground", zh: "托管的轻量版" },
} satisfies Messages;
