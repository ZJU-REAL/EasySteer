/** App chrome, navigation, connection settings, shared buttons. */

import type { Messages } from "./types";

export const commonMessages = {
  app_title: { en: "EasySteer", zh: "EasySteer" },
  app_subtitle: { en: "Steer Vector Control Panel", zh: "Steer Vector 控制面板" },
  nav_home: { en: "Home", zh: "首页" },
  nav_chat: { en: "Chat", zh: "聊天" },
  nav_gallery: { en: "Gallery", zh: "示例库" },
  nav_playground: { en: "Playground", zh: "实验台" },
  nav_workshop: { en: "Workshop", zh: "向量工坊" },
  nav_sae: { en: "SAE", zh: "SAE 特征" },
  language_toggle: { en: "中文", zh: "English" },
  theme_toggle: { en: "Theme", zh: "主题" },

  settings_title: { en: "Connection", zh: "连接设置" },
  openai_base_url_label: {
    en: "Inference server (OpenAI-compatible)",
    zh: "推理服务器（OpenAI 兼容）",
  },
  openai_base_url_help: {
    en: "Base URL of the vllm-steer server, e.g. http://localhost:8000/v1. In dev, /mock/v1 streams canned output.",
    zh: "vllm-steer 服务器的 Base URL，例如 http://localhost:8000/v1。开发模式下 /mock/v1 会返回模拟输出。",
  },
  flask_base_url_label: { en: "Job backend (Flask)", zh: "任务后端（Flask）" },
  flask_base_url_help: {
    en: "Base URL of the extraction/training backend; leave empty to use the same origin.",
    zh: "提取/训练后端的 Base URL；留空表示同源。",
  },
  model_label: { en: "Model", zh: "模型" },
  model_placeholder: { en: "model id served by the endpoint", zh: "服务器提供的模型 ID" },
  check_connection_btn: { en: "Check connection", zh: "检查连接" },
  connection_ok: { en: "Connected. Models: {models}", zh: "连接成功。模型：{models}" },
  connection_failed: { en: "Connection failed: {error}", zh: "连接失败：{error}" },

  copy_btn: { en: "Copy", zh: "复制" },
  copied: { en: "Copied", zh: "已复制" },
  close_btn: { en: "Close", zh: "关闭" },
  reset_btn: { en: "Reset", zh: "重置" },
  remove_btn: { en: "Remove", zh: "删除" },
} satisfies Messages;
