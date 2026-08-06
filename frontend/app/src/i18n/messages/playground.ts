/** Playground: spec builder, apply editor, JSON panel, export, run/compare. */

import type { Messages } from "./types";

export const playgroundMessages = {
  playground_title: { en: "Steer", zh: "引导" },
  steer_intro: {
    en: "Build a SteeringSpec, run it against the server side by side with the baseline, and export the result as code.",
    zh: "配置一份 SteeringSpec，在服务器上与基线并排跑一次对比，并把结果导出为代码。",
  },
  spec_builder_title: { en: "Vectors", zh: "向量" },
  spec_json_title: { en: "SteeringSpec JSON", zh: "SteeringSpec JSON" },
  spec_json_help: {
    en: "Two-way: edit the form or this JSON. Invalid JSON keeps the last good spec.",
    zh: "表单与这段 JSON 双向同步；JSON 写错时会沿用上一份有效的 Spec。",
  },
  vector_n_title: { en: "Vector {n}", zh: "向量 {n}" },
  add_vector_btn: { en: "Add vector", zh: "添加向量" },
  duplicate_vector_btn: { en: "Duplicate", zh: "复制" },
  source_label: { en: "Source path", zh: "向量文件路径" },
  source_placeholder: { en: "e.g. vectors/happy.gguf", zh: "例如 vectors/happy.gguf" },
  source_help: {
    en: "Server-side path to an EasySteer vector file (.gguf, or JSON for moe_router); leave empty when the vector carries an inline data payload.",
    zh: "服务器端的 EasySteer 向量文件路径（.gguf；moe_router 用 JSON）；若向量自带内联 data 负载则留空。",
  },
  data_inline_notice: {
    en: "This vector carries an inline data payload; edit it in the SteeringSpec JSON panel.",
    zh: "该向量自带内联 data 负载，请在 SteeringSpec JSON 面板中编辑。",
  },
  algorithm_label: { en: "Algorithm", zh: "算法" },
  scale_label: { en: "Scale", zh: "缩放系数" },
  layers_label: { en: "Target layers", zh: "目标层" },
  layers_placeholder: {
    en: "e.g. 0-27; empty = from file",
    zh: "例如 0-27；留空则由文件决定",
  },
  normalize_label: { en: "Normalize vector", zh: "归一化向量" },
  params_label: { en: "Params (JSON)", zh: "参数（JSON）" },
  conflict_label: { en: "Conflict resolution", zh: "冲突处理策略" },
  conflict_priority: { en: "priority (first wins)", zh: "priority（按优先级，第一个生效）" },
  conflict_sequential: { en: "sequential (stack in order)", zh: "sequential（按顺序依次叠加）" },
  conflict_error: { en: "error (disallow conflicts)", zh: "error（不允许冲突，直接报错）" },

  apply_title: { en: "Apply (where & when)", zh: "生效范围（何处、何时）" },
  phases_label: { en: "Phases", zh: "阶段" },
  phase_prompt: { en: "prompt", zh: "prompt（提示阶段）" },
  phase_generation: { en: "generation", zh: "generation（生成阶段）" },
  selectors_help: {
    en: "Include selectors take the union of their matches inside the checked phases; leave them all empty to cover the whole phase.",
    zh: "包含类选择器在已勾选的阶段内取匹配结果的并集；全部留空则整个阶段都生效。",
  },
  prompt_group_title: { en: "Prompt", zh: "Prompt 阶段" },
  generation_group_title: { en: "Generation", zh: "生成阶段" },
  prompt_tokens_label: { en: "Prompt token ids", zh: "Prompt token id" },
  tokens_placeholder: {
    en: "e.g. 42, 271",
    zh: "例如 42, 271",
  },
  prompt_tokens_help: {
    en: "Matches prompt occurrences of these token ids.",
    zh: "匹配 prompt 中出现这些 token id 的位置。",
  },
  generation_tokens_label: { en: "Generated token ids", zh: "生成 token id" },
  generation_tokens_help: {
    en: "Matches generated occurrences of these token ids.",
    zh: "匹配生成结果中出现这些 token id 的位置。",
  },
  prompt_positions_label: { en: "Prompt positions", zh: "Prompt 位置" },
  prompt_positions_placeholder: {
    en: "e.g. -1 = last prompt token",
    zh: "例如 -1 表示 prompt 最后一个 token",
  },
  prompt_positions_help: {
    en: "Positions in the prompt; negative counts from its end, past-the-end values clamp to the last token.",
    zh: "prompt 内的位置；负数从末尾往前数，超出长度的值收拢到最后一个 token。",
  },
  prompt_window_label: { en: "Prompt window", zh: "Prompt 窗口" },
  prompt_window_help: {
    en: "Half-open range; negative or empty bounds count from the end of the prompt.",
    zh: "半开区间；边界为负数或留空时从 prompt 末尾计数。",
  },
  generation_positions_label: { en: "Decode steps", zh: "解码步" },
  generation_positions_placeholder: {
    en: "e.g. 0, 1, 2",
    zh: "例如 0, 1, 2",
  },
  generation_positions_help: {
    en: "Exact decode steps, 0-based (0 = first generated token).",
    zh: "精确指定解码步，从 0 开始（0 表示第一个生成的 token）。",
  },
  generation_window_label: { en: "Generation window", zh: "生成窗口" },
  generation_window_help: {
    en: "Half-open range over decode steps; an empty stop means unbounded.",
    zh: "解码步上的半开区间；stop 留空表示不设上界。",
  },
  include_title: { en: "Include", zh: "包含" },
  exclusions_title: { en: "Exclude", zh: "排除" },
  exclusions_help: {
    en: "Exclude selectors take their own union and are always subtracted, so wherever an include and an exclude overlap the exclusion wins.",
    zh: "排除类选择器同样取并集，并始终从结果中减去；与包含范围重叠的部分以排除为准。",
  },
  window_start_placeholder: { en: "start", zh: "起始" },
  window_stop_placeholder: { en: "stop", zh: "结束" },

  validation_ok: { en: "Spec is valid", zh: "Spec 校验通过" },
  validation_issues: { en: "{n} validation issue(s)", zh: "{n} 处校验问题" },
  json_parse_error: { en: "JSON error: {error}", zh: "JSON 解析错误：{error}" },

  export_title: { en: "Export", zh: "导出" },
  export_python_btn: { en: "Python (vllm)", zh: "Python（vllm）" },
  export_extra_body_btn: { en: "OpenAI extra_body", zh: "OpenAI extra_body" },
  export_curl_btn: { en: "curl", zh: "curl" },

  run_title: { en: "Run & Compare", zh: "运行与对比" },
  prompt_label: { en: "Prompt", zh: "Prompt" },
  prompt_placeholder: { en: "Enter a prompt or question", zh: "输入 prompt 或问题" },
  temperature_label: { en: "Temperature", zh: "Temperature" },
  max_tokens_label: { en: "Max tokens", zh: "最大 token 数" },
  run_ab_btn: { en: "Run baseline vs steered", zh: "运行基线 vs 引导" },
  run_steered_btn: { en: "Run steered only", zh: "只跑引导" },
  stop_btn: { en: "Stop", zh: "停止" },
  baseline_title: { en: "Baseline (no steering)", zh: "基线（不引导）" },
  steered_title: { en: "Steered", zh: "引导后" },
  server_default_btn: { en: "Set as server default", zh: "设为服务端默认" },
  server_default_ok: { en: "Server default steering updated.", zh: "服务端默认引导已更新。" },
  run_error: { en: "Request failed: {error}", zh: "请求失败：{error}" },
  waiting_stream: { en: "Waiting for tokens...", zh: "等待生成……" },
} satisfies Messages;
