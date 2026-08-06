/** App chrome, navigation, connection settings, shared buttons. */

import type { Messages } from "./types";

export const commonMessages = {
  app_title: { en: "EasySteer", zh: "EasySteer" },
  nav_home: { en: "Home", zh: "首页" },
  nav_chat: { en: "Chat", zh: "聊天" },
  nav_gallery: { en: "Gallery", zh: "示例库" },
  nav_playground: { en: "Steer", zh: "引导" },
  nav_workshop: { en: "Extract & Train", zh: "提取与训练" },
  nav_sae: { en: "SAE", zh: "SAE 特征" },
  add_item_btn: { en: "Add", zh: "添加" },
  language_toggle: { en: "中文", zh: "English" },
  theme_toggle: { en: "Theme", zh: "主题" },

  openai_base_url_label: {
    en: "Inference server (OpenAI-compatible)",
    zh: "推理服务（OpenAI 兼容）",
  },
  openai_base_url_help: {
    en: "Base URL of the vllm-steer server, e.g. http://localhost:8000/v1; in dev, /mock/v1 streams canned output.",
    zh: "vllm-steer 服务的 Base URL，例如 http://localhost:8000/v1；开发时填 /mock/v1 会流式返回一段模拟输出。",
  },
  model_label: { en: "Model", zh: "模型" },
  model_placeholder: { en: "model id", zh: "模型 ID" },
  check_connection_btn: { en: "Check connection", zh: "测试连接" },
  connection_ok: { en: "Connected. Models: {models}", zh: "连接正常，可用模型：{models}" },
  connection_failed: { en: "Connection failed: {error}", zh: "连接失败：{error}" },

  copy_btn: { en: "Copy", zh: "复制" },
  copied: { en: "Copied", zh: "已复制" },
  close_btn: { en: "Close", zh: "关闭" },
  reset_btn: { en: "Reset", zh: "重置" },
  remove_btn: { en: "Remove", zh: "删除" },
} satisfies Messages;
