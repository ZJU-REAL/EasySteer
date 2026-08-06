/**
 * Minimal i18n composable (en / zh). Chinese strings reuse the wording of
 * the legacy `frontend/i18n.js` where the concept carried over; new UI
 * surfaces (gallery, playground, workshop) get new strings in both
 * languages. All keys and code identifiers are English by project rule.
 */

import { computed } from "vue";
import { settings } from "../lib/settings";

type Messages = Record<string, { en: string; zh: string }>;

export const messages: Messages = {
  // App chrome
  app_title: { en: "EasySteer", zh: "EasySteer" },
  app_subtitle: { en: "Steer Vector Control Panel", zh: "Steer Vector 控制面板" },
  nav_gallery: { en: "Gallery", zh: "示例库" },
  nav_playground: { en: "Playground", zh: "实验台" },
  nav_workshop: { en: "Workshop", zh: "向量工坊" },
  language_toggle: { en: "中文", zh: "English" },
  theme_toggle: { en: "Theme", zh: "主题" },

  // Settings
  settings_title: { en: "Connection", zh: "连接设置" },
  openai_base_url_label: { en: "Inference server (OpenAI-compatible)", zh: "推理服务器（OpenAI 兼容）" },
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

  // Gallery
  gallery_title: { en: "Demo Gallery", zh: "示例库" },
  gallery_intro: {
    en: "Each card replicates a published steering method from the replications/ notebooks. Open one to load its exact SteeringSpec into the playground.",
    zh: "每张卡片对应 replications/ 中一篇已发表论文的引导方法复现。点击卡片即可将其 SteeringSpec 载入实验台。",
  },
  open_in_playground: { en: "Open in playground", zh: "在实验台中打开" },
  gallery_model: { en: "Model", zh: "模型" },
  gallery_algorithm: { en: "Algorithm", zh: "算法" },
  gallery_vectors: { en: "Vectors", zh: "向量数" },
  gallery_paper: { en: "Paper", zh: "论文" },

  // Playground
  playground_title: { en: "Playground", zh: "实验台" },
  spec_builder_title: { en: "Spec Builder", zh: "Spec 构建器" },
  spec_json_title: { en: "SteeringSpec JSON", zh: "SteeringSpec JSON" },
  spec_json_help: {
    en: "Two-way: edit the form or this JSON. Invalid JSON keeps the last good spec.",
    zh: "双向编辑：可修改表单或直接编辑 JSON。JSON 非法时保留上一个有效 Spec。",
  },
  vector_n_title: { en: "Vector {n}", zh: "向量 {n}" },
  add_vector_btn: { en: "Add vector", zh: "添加向量配置" },
  remove_vector_btn: { en: "Remove", zh: "删除" },
  duplicate_vector_btn: { en: "Duplicate", zh: "复制" },
  source_label: { en: "Source path", zh: "向量文件路径" },
  source_placeholder: { en: "e.g. vectors/happy.gguf", zh: "例如 vectors/happy.gguf" },
  source_help: {
    en: "Server-side path to an EasySteer vector file (.gguf; moe_router JSON). Leave empty when using an inline data payload.",
    zh: "服务器端向量文件路径（.gguf；moe_router 为 JSON）。使用内联 data 负载时留空。",
  },
  data_inline_notice: {
    en: "This vector carries an inline data payload (edit it in the JSON panel).",
    zh: "该向量携带内联 data 负载（请在 JSON 面板中编辑）。",
  },
  algorithm_label: { en: "Algorithm", zh: "算法选择" },
  scale_label: { en: "Scale", zh: "缩放因子" },
  layers_label: { en: "Target layers", zh: "目标层级" },
  layers_placeholder: { en: "e.g. 0-27 or 16,17,18; empty = file decides", zh: "例如 0-27 或 16,17,18；留空由文件决定" },
  normalize_label: { en: "Normalize vector", zh: "标准化 Steer Vector" },
  name_label: { en: "Name (logs only)", zh: "名称（仅用于日志）" },
  params_label: { en: "Params (JSON)", zh: "参数（JSON）" },
  conflict_label: { en: "Conflict resolution", zh: "向量冲突解决方法" },
  conflict_priority: { en: "priority (first wins)", zh: "priority（优先级，仅第一个生效）" },
  conflict_sequential: { en: "sequential (stack in order)", zh: "sequential（按顺序叠加）" },
  conflict_error: { en: "error (disallow conflicts)", zh: "error（不允许冲突）" },
  debug_label: { en: "Debug logging", zh: "启用调试模式" },

  // ApplySpec editor
  apply_title: { en: "Apply (where & when)", zh: "应用范围（何处/何时）" },
  phases_label: { en: "Phases", zh: "阶段" },
  phase_prompt: { en: "prompt", zh: "prompt（提示词）" },
  phase_generation: { en: "generation", zh: "generation（生成）" },
  tokens_label: { en: "Token allowlist", zh: "触发 Token IDs" },
  tokens_placeholder: { en: "token ids, comma-separated; empty = all", zh: "Token ID，逗号分隔；留空表示全部" },
  positions_label: { en: "Positions", zh: "触发位置" },
  positions_placeholder: { en: "e.g. -1 = last prompt token", zh: "例如 -1 表示提示词最后一个 token" },
  selectors_help: {
    en: "Include selectors below select the union of their matches within the checked phases; none set = the whole phases.",
    zh: "下方的选择器在勾选阶段内取匹配的并集；全部留空表示整个阶段。",
  },
  prompt_window_label: { en: "Prompt window", zh: "提示词窗口" },
  prompt_window_help: {
    en: "Half-open [start, stop) over prompt positions; negative bounds and empty stop resolve from the prompt end ([-5, empty] = last five prompt tokens).",
    zh: "提示词位置上的半开区间 [start, stop)；负值与留空的 stop 从提示词末尾解析（[-5, 空] 表示最后五个提示词 token）。",
  },
  generation_positions_label: { en: "Generation steps", zh: "生成步索引" },
  generation_positions_placeholder: {
    en: "0-based decode steps, e.g. 0,1,2",
    zh: "0 起始的解码步，例如 0,1,2",
  },
  generation_window_label: { en: "Generation window", zh: "生成窗口" },
  generation_window_help: {
    en: "Half-open [start, stop) over decode steps; empty stop = unbounded.",
    zh: "解码步数上的半开区间 [start, stop)；stop 留空表示无上界。",
  },
  exclusions_title: { en: "Exclusions", zh: "排除规则" },
  exclusions_help: {
    en: "Exclude selectors union and always subtract — where an include and an exclude overlap, the exclusion wins.",
    zh: "排除选择器取并集并始终做减法——包含与排除重叠时，排除优先。",
  },
  exclude_tokens_label: { en: "Exclude tokens", zh: "排除 Token IDs" },
  exclude_positions_label: { en: "Exclude positions", zh: "排除位置" },
  exclude_prompt_window_label: { en: "Exclude prompt window", zh: "排除提示词窗口" },
  exclude_generation_positions_label: { en: "Exclude generation steps", zh: "排除生成步索引" },
  exclude_generation_window_label: { en: "Exclude generation window", zh: "排除生成窗口" },
  window_start_placeholder: { en: "start", zh: "起始" },
  window_stop_placeholder: { en: "stop (optional)", zh: "结束（可选）" },

  // Validation / errors
  validation_ok: { en: "Spec is valid", zh: "Spec 校验通过" },
  validation_issues: { en: "{n} validation issue(s)", zh: "{n} 个校验问题" },
  json_parse_error: { en: "JSON error: {error}", zh: "JSON 错误：{error}" },

  // Export
  export_title: { en: "Export", zh: "导出" },
  export_python_btn: { en: "Python (vllm)", zh: "Python（vllm）" },
  export_extra_body_btn: { en: "OpenAI extra_body", zh: "OpenAI extra_body" },
  export_curl_btn: { en: "curl", zh: "curl" },
  copy_btn: { en: "Copy", zh: "复制" },
  reset_btn: { en: "Reset", zh: "重置" },
  copied: { en: "Copied", zh: "已复制" },
  close_btn: { en: "Close", zh: "关闭" },

  // Run / compare
  run_title: { en: "Run & Compare", zh: "运行与对比" },
  prompt_label: { en: "Prompt", zh: "输入指令" },
  prompt_placeholder: { en: "Enter your prompt or question", zh: "输入您的提示词或问题" },
  system_prompt_label: { en: "System prompt (optional)", zh: "系统提示词（可选）" },
  temperature_label: { en: "Temperature", zh: "Temperature" },
  max_tokens_label: { en: "Max tokens", zh: "最大 Tokens" },
  run_ab_btn: { en: "Run A/B (baseline vs steered)", zh: "A/B 对比（基准 vs 干预）" },
  run_steered_btn: { en: "Run steered only", zh: "仅运行干预" },
  stop_btn: { en: "Stop", zh: "停止" },
  baseline_title: { en: "Baseline (no steering)", zh: "基准（不干预）" },
  steered_title: { en: "Steered", zh: "干预结果" },
  ab_mode_label: { en: "Compare mode", zh: "对比模式" },
  ab_mode_baseline: { en: "Baseline vs spec", zh: "基准 vs Spec" },
  ab_mode_two_specs: { en: "Spec A vs spec B", zh: "Spec A vs Spec B" },
  spec_a_title: { en: "Spec A", zh: "Spec A" },
  spec_b_title: { en: "Spec B (JSON)", zh: "Spec B（JSON）" },
  server_default_btn: { en: "Set as server default", zh: "设为服务器默认" },
  server_default_ok: { en: "Server default steering updated.", zh: "服务器默认引导已更新。" },
  clear_server_default_btn: { en: "Clear server default", zh: "清除服务器默认" },
  run_error: { en: "Request failed: {error}", zh: "请求失败：{error}" },
  waiting_stream: { en: "Waiting for tokens...", zh: "等待生成..." },

  // Workshop
  workshop_title: { en: "Vector Workshop", zh: "向量工坊" },
  workshop_intro: {
    en: "Produce a steering vector: pick a method, configure, submit the job to the Flask backend, then use the result in the playground.",
    zh: "生成引导向量：选择方法、填写配置、提交任务到 Flask 后端，完成后可在实验台中使用。",
  },
  workshop_kind_extraction: { en: "Extraction (activation-based)", zh: "提取（基于激活）" },
  workshop_kind_training: { en: "Training (ReFT)", zh: "训练（ReFT）" },
  model_path_label: { en: "Model path", zh: "模型路径" },
  model_path_placeholder: { en: "e.g. /path/to/Qwen2.5-1.5B-Instruct/", zh: "例如: /path/to/Qwen2.5-1.5B-Instruct/" },
  gpu_devices_label: { en: "GPU devices", zh: "GPU 设备号" },
  gpu_devices_placeholder: { en: "e.g. 0,1,2 or a single GPU: 0", zh: "例如: 0,1,2 或单个GPU: 0" },
  extract_method_label: { en: "Extraction method", zh: "提取方法" },
  extract_method_diffmean: { en: "DiffMean - difference of means", zh: "DiffMean - 均值差分" },
  extract_method_pca: { en: "PCA - principal component analysis", zh: "PCA - 主成分分析" },
  extract_method_lat: { en: "LAT - linear algebraic technique", zh: "LAT - 线性代数技术" },
  extract_token_pos_label: { en: "Token position", zh: "Token 位置" },
  extract_token_pos_help: { en: "Integer index into each sample (-1 = last token).", zh: "样本内的整数索引（-1 表示最后一个 token）。" },
  extract_normalize_label: { en: "Normalize vector", zh: "归一化向量" },
  positive_samples_label: { en: "Positive samples (one per line)", zh: "正样本（每行一个）" },
  positive_samples_help: {
    en: "Samples exhibiting the behavior/concept you want to amplify.",
    zh: "输入代表您想要增强的行为/概念的样本。",
  },
  negative_samples_label: { en: "Negative samples (one per line)", zh: "负样本（每行一个）" },
  negative_samples_help: {
    en: "Neutral samples that do not exhibit the target behavior.",
    zh: "输入不代表目标行为/概念的中性样本。",
  },
  output_path_label: { en: "Output file path", zh: "输出文件路径" },
  output_path_help: { en: "Where the extracted .gguf control vector is saved (server-side).", zh: "提取的控制向量（.gguf）在服务器上的保存路径。" },
  start_extraction_btn: { en: "Extract vector", zh: "提取向量" },
  start_training_btn: { en: "Start training", zh: "开始训练" },
  import_config_label: { en: "Import preset", zh: "导入配置" },
  import_config_placeholder: { en: "-- select preset --", zh: "-- 选择配置 --" },
  train_layer_label: { en: "Target layer", zh: "目标层" },
  train_component_label: { en: "Component", zh: "组件" },
  train_low_rank_dim_label: { en: "Low-rank dimension", zh: "低秩维度" },
  train_intervention_label: { en: "Intervention", zh: "干预类型" },
  train_epochs_label: { en: "Epochs", zh: "训练轮数" },
  train_batch_size_label: { en: "Batch size", zh: "批大小" },
  train_learning_rate_label: { en: "Learning rate", zh: "学习率" },
  train_logging_steps_label: { en: "Logging steps", zh: "日志步数" },
  train_output_dir_label: { en: "Output directory", zh: "输出目录" },
  train_examples_label: { en: "Training examples", zh: "训练样例" },
  train_examples_help: { en: "One [input, output] pair per row.", zh: "每行一个 [输入, 输出] 对。" },
  add_example_btn: { en: "Add example", zh: "添加样例" },
  example_input_placeholder: { en: "input text...", zh: "输入文本..." },
  example_output_placeholder: { en: "expected output...", zh: "期望输出..." },
  job_status_title: { en: "Job status", zh: "任务状态" },
  job_logs_title: { en: "Logs", zh: "日志" },
  job_idle: { en: "No job running.", zh: "当前没有运行中的任务。" },
  job_running: { en: "Running...", zh: "运行中..." },
  job_done: { en: "Done", zh: "已完成" },
  job_failed: { en: "Failed", zh: "失败" },
  job_started: { en: "Job submitted.", zh: "任务已提交。" },
  use_in_playground_btn: { en: "Use in playground", zh: "在实验台中使用" },
  layers_extracted_label: { en: "Layers extracted", zh: "已提取层数" },
  output_file_label: { en: "Output file", zh: "输出文件" },
  status_label: { en: "Status", zh: "状态" },
};

export type MessageKey = keyof typeof messages;

export function translate(
  key: MessageKey,
  params?: Record<string, string | number>,
): string {
  const entry = messages[key];
  if (!entry) return String(key);
  let text = entry[settings.language];
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      text = text.replace(`{${k}}`, String(v));
    }
  }
  return text;
}

export function useI18n() {
  const language = computed({
    get: () => settings.language,
    set: (value: "en" | "zh") => {
      settings.language = value;
    },
  });
  return { t: translate, language };
}
