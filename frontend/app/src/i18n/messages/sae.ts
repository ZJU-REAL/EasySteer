/**
 * SAE feature explorer. Chinese strings reuse the wording of the legacy
 * SAE tab where the concept carried over.
 */

import type { Messages } from "./types";

export const saeMessages = {
  sae_title: { en: "SAE Feature Explorer", zh: "SAE 特征浏览器" },
  sae_intro: {
    en: "Search Neuronpedia for sparse-autoencoder features, inspect one, and turn its decoder direction into a steering vector.",
    zh: "在 Neuronpedia 上搜索稀疏自编码器（SAE）特征，查看它的详情，并把它的解码器方向做成引导向量。",
  },
  sae_search_title: { en: "Find a feature", zh: "查找特征" },
  sae_model_id_label: { en: "Model ID", zh: "模型 ID" },
  sae_id_label: { en: "SAE ID", zh: "SAE ID" },
  sae_api_key_label: { en: "Neuronpedia API key", zh: "Neuronpedia API Key" },
  sae_model_id_help: { en: "Model id as used by Neuronpedia.", zh: "Neuronpedia 上的模型 id。" },
  sae_id_help: { en: "SAE id as used by Neuronpedia.", zh: "Neuronpedia 上的 SAE id。" },
  sae_api_key_help: {
    en: "Stored locally in your browser; required for every lookup.",
    zh: "只保存在本地浏览器里，每次查询都要用到。",
  },
  sae_search_by_query: { en: "Semantic query", zh: "语义搜索" },
  sae_search_by_index: { en: "Feature index", zh: "按特征索引" },
  sae_query_label: { en: "Search query", zh: "搜索内容" },
  sae_query_placeholder: {
    en: "e.g. references to famous people",
    zh: "例如：提及名人的内容",
  },
  sae_feature_index_label: { en: "Feature index", zh: "特征索引" },
  sae_feature_index_placeholder: { en: "e.g. 8614", zh: "例如 8614" },
  sae_search_btn: { en: "Search features", zh: "搜索特征" },
  sae_lookup_btn: { en: "Look up feature", zh: "查询特征" },
  sae_searching: { en: "Loading...", zh: "加载中……" },
  sae_no_results: { en: "No matching features found.", zh: "没有找到匹配的特征。" },
  sae_details_placeholder: {
    en: "Run a search or enter a feature index, then open a feature to see its details here.",
    zh: "先搜索或输入特征索引，再打开某个特征，详情会显示在这里。",
  },
  sae_results_title: { en: "Search results", zh: "搜索结果" },
  sae_feature_id_column: { en: "Index", zh: "索引" },
  sae_description_column: { en: "Description", zh: "描述" },
  sae_similarity_column: { en: "Similarity", zh: "相似度" },
  sae_details_btn: { en: "Details", zh: "详情" },
  sae_feature_details: { en: "Feature details", zh: "特征详情" },
  sae_explanation_label: { en: "Explanation", zh: "特征说明" },
  sae_sparsity_label: { en: "Sparsity", zh: "稀疏度" },
  sae_layer_label: { en: "Layer", zh: "层" },
  sae_top_activating_tokens: { en: "Top activating tokens", zh: "激活最强的 token" },
  sae_top_inhibiting_tokens: { en: "Top inhibiting tokens", zh: "抑制最强的 token" },
  sae_activation_example: { en: "Activation example", zh: "激活示例" },
  sae_max_value_label: { en: "Max value", zh: "最大激活值" },
  sae_trigger_token_label: { en: "Trigger token", zh: "触发 token" },
  sae_extract_title: { en: "Extract as steering vector", zh: "提取为引导向量" },
  sae_extract_help: {
    en: "Saves the decoder row as a .pt file and returns its steering payload for the selected layer (needs SAE_PARAMS_PATH on the job backend).",
    zh: "保存解码器行为 .pt 文件，并返回所选层的引导负载（任务后端需配置 SAE_PARAMS_PATH）。",
  },
  sae_vector_name_help: { en: "Display name for the extracted vector.", zh: "提取向量的显示名称。" },
  sae_scale_help: {
    en: "Decoder rows are unit-norm, so a scale around 500 is a typical strength.",
    zh: "解码器行是单位长度的，缩放系数取 500 左右是常见强度。",
  },
  sae_vector_name_label: { en: "Vector name", zh: "向量名称" },
  sae_layer_input_label: { en: "Target layer", zh: "目标层" },
  sae_layer_input_help: {
    en: "Layer the SAE reads from. Choose it before extracting; changing it requires extracting again.",
    zh: "该 SAE 所在的层。请在提取前选择，修改后需要重新提取。",
  },
  sae_layer_required: { en: "Choose a non-negative target layer before extracting.", zh: "请在提取前选择非负的目标层号。" },
  sae_extract_btn: { en: "Extract vector", zh: "提取向量" },
  sae_extracting: { en: "Extracting...", zh: "正在提取……" },
  sae_extract_done: { en: "Vector saved to {path}", zh: "向量已保存到 {path}" },
  sae_error: { en: "SAE request failed: {error}", zh: "SAE 请求失败：{error}" },
} satisfies Messages;
