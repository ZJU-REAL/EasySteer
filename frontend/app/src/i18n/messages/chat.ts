/** Chat page: multi-turn streaming chat with optional steering. */

import type { Messages } from "./types";

export const chatMessages = {
  chat_title: { en: "Chat", zh: "聊天" },
  chat_intro: {
    en: "Multi-turn chat against the OpenAI-compatible server, with an optional steering spec applied to every reply.",
    zh: "在 OpenAI 兼容服务上进行多轮对话，可以给每条回复都挂上一份引导 Spec。",
  },
  chat_input_placeholder: {
    en: "Type a message (Enter to send, Shift+Enter for a newline)",
    zh: "输入消息（Enter 发送，Shift+Enter 换行）",
  },
  send_btn: { en: "Send", zh: "发送" },
  clear_chat_btn: { en: "Clear conversation", zh: "清空对话" },
  chat_empty: { en: "No messages yet", zh: "还没有消息" },
  chat_empty_hint: {
    en: "Point the inference server at a running vllm-steer endpoint, then say hello. Pick a steering mode above to steer every reply.",
    zh: "先把推理服务指向正在运行的 vllm-steer 端点，然后发一条消息试试。在上方选择一种引导方式，就能让每条回复都受到引导。",
  },
  steering_panel_title: { en: "Steering", zh: "引导设置" },
  chat_presets_label: { en: "Presets", zh: "预设" },
  chat_presets_help: {
    en: "Picking a preset fills the custom spec; its vector paths resolve on the server.",
    zh: "选中预设会填入自定义 Spec，其中的向量路径由服务器解析。",
  },
  chat_compare_label: { en: "Compare with baseline", zh: "与基线对比" },
  chat_compare_help: {
    en: "Answers each message twice, steered and unsteered, side by side.",
    zh: "每条消息生成两份回复：引导与不引导，并排显示。",
  },
  chat_compare_steered: { en: "steered", zh: "已引导" },
  chat_compare_baseline: { en: "baseline", zh: "基线" },
  chat_steering_help: {
    en: "Pick a preset below, reuse the Steer page spec, or paste your own JSON.",
    zh: "可以选用下方的预设、复用“引导”页的 Spec，或直接粘贴一份自定义 JSON。",
  },
  chat_seed_from_steer_btn: { en: "Fill from the Steer page", zh: "从“引导”页填入" },
  chat_no_reply: { en: "(not generated)", zh: "（未生成）" },
  steering_mode_none: { en: "Off (baseline chat)", zh: "关闭（基线对话）" },
  steering_mode_playground: { en: "Use the Steer page spec", zh: "使用“引导”页的 Spec" },
  steering_mode_custom: { en: "Custom spec (JSON)", zh: "自定义 Spec（JSON）" },
  playground_spec_summary: {
    en: "Current Steer page spec: {summary}",
    zh: "“引导”页当前的 Spec：{summary}",
  },
  edit_in_playground_btn: { en: "Edit on Steer page", zh: "在“引导”页中编辑" },
  chat_role_user: { en: "You", zh: "你" },
  chat_role_assistant: { en: "Assistant", zh: "助手" },
} satisfies Messages;
