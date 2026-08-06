/**
 * SAE feature explorer. Chinese strings reuse the wording of the legacy
 * SAE tab where the concept carried over.
 */

import type { Messages } from "./types";

export const saeMessages = {
  sae_title: { en: "SAE Feature Explorer", zh: "SAE 特征探索" },
  sae_intro: {
    en: "Search Neuronpedia for sparse-autoencoder features, inspect them, and turn a feature's decoder direction into a steering vector.",
    zh: "在 Neuronpedia 中搜索稀疏自编码器特征、查看详情，并将特征的解码器方向转换为引导向量。",
  },
  sae_model_id_label: { en: "Model ID", zh: "模型 ID" },
  sae_id_label: { en: "SAE ID", zh: "SAE ID" },
  sae_api_key_label: { en: "Neuronpedia API key", zh: "Neuronpedia API Key" },
  sae_api_key_help: {
    en: "Stored locally in your browser; required for all lookups.",
    zh: "保存在本地浏览器中；所有查询都需要提供。",
  },
  sae_search_by_query: { en: "Semantic query", zh: "语义查询" },
  sae_search_by_index: { en: "Feature index", zh: "特征索引查询" },
  sae_query_label: { en: "Search query", zh: "搜索查询" },
  sae_query_placeholder: {
    en: "e.g. references to famous people",
    zh: "例如：与名人相关的内容",
  },
  sae_feature_index_label: { en: "Feature index", zh: "特征索引" },
  sae_feature_index_placeholder: { en: "e.g. 8614", zh: "例如 8614" },
  sae_search_btn: { en: "Search features", zh: "搜索特征" },
  sae_lookup_btn: { en: "Look up feature", zh: "查询特征" },
  sae_searching: { en: "Searching features...", zh: "正在搜索特征..." },
  sae_no_results: { en: "No matching features found.", zh: "没有找到相关特征" },
  sae_results_title: { en: "Search results", zh: "搜索结果" },
  sae_feature_id_column: { en: "Index", zh: "特征ID" },
  sae_description_column: { en: "Description", zh: "描述" },
  sae_similarity_column: { en: "Similarity", zh: "相似度" },
  sae_details_btn: { en: "Details", zh: "查看详情" },
  sae_feature_details: { en: "Feature details", zh: "特征详情" },
  sae_explanation_label: { en: "Explanation", zh: "描述" },
  sae_sparsity_label: { en: "Sparsity", zh: "稀疏度" },
  sae_layer_label: { en: "Layer", zh: "层" },
  sae_top_activating_tokens: { en: "Top activating tokens", zh: "最高激活词元" },
  sae_top_inhibiting_tokens: { en: "Top inhibiting tokens", zh: "最低激活词元" },
  sae_activation_example: { en: "Activation example", zh: "激活示例" },
  sae_max_value_label: { en: "Max value", zh: "最大值" },
  sae_trigger_token_label: { en: "Trigger token", zh: "触发词元" },
  sae_context_label: { en: "Context", zh: "上下文" },
  sae_extract_title: { en: "Extract as steering vector", zh: "提取为引导向量" },
  sae_extract_help: {
    en: "Saves the feature's decoder row as a .pt file server-side (requires SAE_PARAMS_PATH on the backend).",
    zh: "将该特征的解码器行保存为服务器端 .pt 文件（后端需配置 SAE_PARAMS_PATH）。",
  },
  sae_vector_name_label: { en: "Vector name", zh: "向量名称" },
  sae_layer_input_label: { en: "Target layer", zh: "目标层" },
  sae_layer_input_help: {
    en: "Layer the SAE reads from (used for the playground spec).",
    zh: "SAE 对应的层（用于生成实验台 Spec）。",
  },
  sae_extract_btn: { en: "Extract vector", zh: "提取向量" },
  sae_extracting: { en: "Extracting...", zh: "正在提取..." },
  sae_extract_done: { en: "Vector saved to {path}", zh: "向量已保存至 {path}" },
  sae_error: { en: "SAE request failed: {error}", zh: "SAE 请求失败：{error}" },
} satisfies Messages;
