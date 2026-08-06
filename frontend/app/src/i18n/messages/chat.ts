/** Chat page: multi-turn streaming chat with optional steering. */

import type { Messages } from "./types";

export const chatMessages = {
  chat_title: { en: "Chat", zh: "聊天" },
  chat_intro: {
    en: "Multi-turn chat against the OpenAI-compatible server, with an optional steering spec applied to every reply.",
    zh: "与 OpenAI 兼容服务器的多轮对话，可为每条回复附加引导 Spec。",
  },
  chat_input_placeholder: {
    en: "Type a message (Enter to send, Shift+Enter for newline)",
    zh: "输入消息（Enter 发送，Shift+Enter 换行）",
  },
  send_btn: { en: "Send", zh: "发送" },
  clear_chat_btn: { en: "Clear conversation", zh: "清空对话" },
  chat_empty: { en: "No messages yet. Say hello!", zh: "还没有消息，来打个招呼吧！" },
  steering_panel_title: { en: "Steering", zh: "引导设置" },
  steering_mode_none: { en: "Off (baseline chat)", zh: "关闭（基准对话）" },
  steering_mode_playground: { en: "Use playground spec", zh: "使用实验台 Spec" },
  steering_mode_custom: { en: "Custom spec (JSON)", zh: "自定义 Spec（JSON）" },
  steering_active_badge: { en: "steered", zh: "已引导" },
  playground_spec_summary: {
    en: "Current playground spec: {summary}",
    zh: "当前实验台 Spec：{summary}",
  },
  edit_in_playground_btn: { en: "Edit in playground", zh: "在实验台中编辑" },
  chat_role_user: { en: "You", zh: "你" },
  chat_role_assistant: { en: "Assistant", zh: "助手" },
} satisfies Messages;
