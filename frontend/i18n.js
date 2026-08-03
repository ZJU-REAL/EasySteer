// Language package definition
const translations = {
    zh: {
        // 页面标题和副标题
        subtitle: "Steer Vector 控制面板",
        
        // 主要区块标题
        basic_config: "Steer Vector 配置",
        model_config: "模型和采样配置",
        sampling_config: "采样参数",
        layer_config: "层级配置",
        trigger_config: "触发器配置",
        
        // 页面导航
        nav_inference: "推理",
        nav_training: "训练",
        nav_extraction: "提取",
        nav_chat: "聊天",
        
        // 推理子导航
        single_vector_mode: "单向量控制",
        multi_vector_mode: "多向量控制",
        sae_explore_mode: "SAE 特征探索",
        add_vector_btn: "添加向量配置",
        conflict_resolution_label: "向量冲突解决方法",
        conflict_sequential: "顺序应用（按添加顺序依次应用所有向量）",
        conflict_priority: "优先级（仅使用第一个冲突向量）",
        conflict_error: "报错（不允许冲突）",
        no_vector_configs_error: "请至少添加一个向量配置",
        
        // SAE 探索相关
        sae_search_btn: "搜索特征",
        sae_model_id_label: "模型 ID",
        sae_id_label: "SAE ID",
        sae_query_label: "搜索查询",
        sae_searching: "正在搜索特征...",
        sae_search_results: "搜索结果",
        sae_no_results: "没有找到相关特征",
        sae_error: "搜索错误",
        sae_api_key_label: "API Key (必填)",
        sae_api_key_placeholder: "输入 Neuronpedia API key",
        sae_api_key_help: "输入您的 Neuronpedia API 密钥，将保存在本地",
        sae_search_by_query: "语义查询",
        sae_search_by_index: "特征索引查询",
        sae_feature_index_label: "特征索引",
        sae_feature_index_placeholder: "输入特征索引（如8614）",
        sae_feature_index_help: "输入要查询的特征索引编号",
        sae_view_feature_details: "查看特征详情",
        sae_query_help: "输入语义搜索查询以查找相关特征",
        sae_api_key_required: "API Key 是必填项",
        sae_feature_index_required: "请输入特征索引",
        sae_search_failed: "特征搜索失败",
        sae_feature_details_failed: "获取特征详情失败",
        sae_feature_id_column: "特征ID",
        sae_description_column: "描述",
        sae_similarity_column: "相似度",
        sae_actions_column: "操作",
        sae_no_description: "无描述",
        sae_view_details_label: "查看特征详情",
        sae_enter_feature_id: "输入特征ID",
        sae_feature_details: "特征详情",
        sae_unknown: "未知",
        sae_basic_info: "基本信息",
        sae_model_label: "模型",
        sae_layer_label: "层",
        sae_index_label: "索引",
        sae_description_label: "描述",
        sae_sparsity_label: "稀疏度",
        sae_top_activating_tokens: "最高激活词元",
        sae_top_inhibiting_tokens: "最低激活词元",
        sae_activation_example: "激活示例",
        sae_max_value_label: "最大值",
        sae_trigger_token_label: "触发词元",
        sae_context_label: "上下文",
        close_btn: "关闭",
        sae_enter_query_prompt: "请输入查询条件并点击搜索",
        sae_enter_index_prompt: "请输入特征索引并点击查询",
        
        // 模型配置标签
        model_path_label: "模型路径",
        model_path_placeholder: "例如: /path/to/Qwen2.5-1.5B-Instruct/",
        gpu_devices_label: "GPU 设备号",
        gpu_devices_placeholder: "例如: 0,1,2 或单个GPU: 0",
        normalize_sv_label: "标准化 Steer Vector",
        instruction_label: "输入指令",
        instruction_placeholder: "输入您的提示词或问题",
        
        // 采样参数标签
        temperature_label: "Temperature",
        max_tokens_label: "最大 Tokens",
        repetition_penalty_label: "重复惩罚",
        
        // 基础配置标签
        sv_name_label: "Steer Vector 名称",
        sv_name_placeholder: "输入 steer vector 名称",
        sv_id_label: "Steer Vector ID",
        sv_id_placeholder: "输入唯一 ID",
        file_path_label: "本地文件路径",
        scale_factor_label: "缩放因子",
        algorithm_label: "算法选择",
        algorithm_direct: "Direct (直接算法)",
        algorithm_erase: "Erase (投影消除方向)",
        algorithm_replace: "Replace (沿方向覆写)",
        algorithm_concept_replace: "Concept Replace (概念替换)",
        algorithm_linear: "Linear (仿射映射)",
        algorithm_lm_steer: "LM-Steer (低秩投影对)",
        algorithm_loreft: "LoReFT (低秩表示微调)",
        algorithm_moe_router: "MoE Router (专家激活/停用)",
        
        // 层级配置
        target_layers_label: "目标层级",
        target_layers_placeholder: "例如: 0,1,2,3 或留空应用到所有层",
        target_layers_help: "输入层级索引，用逗号分隔。留空表示应用到所有层。",
        
        // 触发器配置
        prefill_trigger_title: "预填充阶段触发器",
        prefill_tokens_label: "Prefill 触发 Token IDs",
        prefill_tokens_placeholder: "例如: 100,200,300 或 -1 应用到所有",
        prefill_tokens_help: "输入 token ID，用逗号分隔。使用 -1 应用到所有 token。",
        prefill_positions_label: "Prefill 触发位置",
        prefill_positions_placeholder: "例如: 0,1,-1 (支持负索引)",
        prefill_positions_help: "输入位置索引，用逗号分隔。支持负索引（-1表示最后一个位置）。",
        generate_trigger_title: "生成阶段触发器",
        generate_tokens_label: "Generate 触发 Token IDs",
        generate_tokens_placeholder: "例如: 400,500,600 或 -1 应用到所有",
        generate_tokens_help: "输入 token ID，用逗号分隔。使用 -1 应用到所有 token。",
        
        // 调试选项
        enable_debug: "启用调试模式",
        debug_help: "启用后将在前向传播过程中打印调试信息。",
        
        // 训练相关
        train_model_config: "训练模型配置",
        train_output_dir_label: "输出目录",
        train_output_dir_placeholder: "例如：./results/my_training",
        train_reft_config: "ReFT 配置",
        train_layer_label: "目标层",
        train_layer_help: "应用干预的层索引。",
        train_component_label: "组件",
        train_component_block_output: "块输出",
        train_component_attention_output: "注意力输出",
        train_component_mlp_output: "MLP 输出",
        train_component_help: "层中应用干预的组件。",
        train_low_rank_dim_label: "低秩维度",
        train_low_rank_dim_help: "低秩适应的维度。较低的值使用较少的内存。",
        train_params_config: "训练参数",
        train_epochs_label: "训练轮数",
        train_batch_size_label: "批大小",
        train_learning_rate_label: "学习率",
        train_logging_steps_label: "日志步数",
        train_save_steps_label: "保存步数",
        train_save_steps_help: "检查点保存之间的步数。较高的值保存频率较低。",
        train_data_config: "训练数据",
        train_examples_label: "训练样例",
        train_examples_help: "添加训练样例，每个样例为[输入, 输出]对。",
        add_example_btn: "添加样例",
        no_examples_message: "尚未添加训练样例。点击\"添加样例\"添加第一个样例。",
        no_examples_error: "请添加至少一个训练样例。",
        example_input_label: "输入:",
        example_output_label: "输出:",
        example_input_placeholder: "输入文本...",
        example_output_placeholder: "期望输出...",
        train_config_select_help: "选择并导入预配置的训练设置。",
        train_start_btn: "开始训练",
        train_response_title: "训练结果",
        
        // 提取相关
        extract_config_select_help: "选择并导入预配置的提取设置。",
        train_progress_title: "训练进度",
        train_current_epoch: "当前轮次",
        train_current_step: "当前步数", 
        train_current_loss: "损失值",
        train_learning_rate: "学习率",
        train_estimated_time: "预计剩余时间",
        train_logs_title: "训练日志",
        train_waiting: "等待训练开始...",
        
        // 按钮
        submit_btn: "生成",
        reset_btn: "重置",

        import_config_btn: "导入",
        restart_backend_btn: "重启后端",
        
        // 配置选择
        config_select_label: "导入配置",
        config_select_placeholder: "-- 选择配置 --",
        config_emoji_loreft: "Emoji LoReft 配置",
        config_emotion_direct: "Emotion Direct 配置",
        config_select_help: "选择并导入预配置的 steer vector 设置。",
        
        // 响应标题
        response_title: "生成结果",
        error_title: "错误信息",
        baseline_title: "基准（不干预）",
        steered_title: "干预结果",
        
        // 错误和提示消息
        required_fields_error: "请填写所有必填字段：名称、ID、文件路径、模型路径和指令",
        network_error: "网络错误：无法连接到服务器。请确保后端服务正在运行。",
        submit_failed: "提交失败",

        // 提取相关
        extract_model_config: "模型配置",
        extract_method_config: "提取方法",
        extract_method_label: "方法选择",
        extract_method_lat: "LAT - 线性代数技术",
        extract_method_pca: "PCA - 主成分分析",
        extract_method_sae: "SAE - 稀疏自编码器",
        extract_method_diffmean: "DiffMean - 均值差分",
        extract_target_layer_label: "目标层",
        extract_target_layer_help: "指定单个层索引，或留空以从所有层提取。",
        extract_token_pos_label: "Token位置",
        extract_token_last: "最后一个Token",
        extract_token_first: "第一个Token",
        extract_token_mean: "所有Token的均值",
        extract_token_max: "最大范数Token",
        extract_sae_params_label: "SAE参数路径",
        extract_sae_params_help: "预训练的SAE参数文件路径。",
        extract_sae_combination_label: "特征组合模式",
        extract_sae_weighted_all: "加权所有特征",
        extract_sae_weighted_top_k: "加权Top-K特征",
        extract_sae_single_top: "单个最重要特征",
        extract_sae_top_k_label: "Top K特征数",
        extract_normalize_label: "归一化向量",
        extract_data_config: "样本数据",
        extract_positive_samples_label: "正样本",
        extract_positive_samples_placeholder: "输入正样本（每行一个）：\n我喜欢小狗！\n狗狗是最好的伙伴。\n我的狗给我带来了很多快乐。",
        extract_positive_samples_help: "输入代表您想要增强的行为/概念的样本。",
        extract_negative_samples_label: "负样本",
        extract_negative_samples_placeholder: "输入负样本（每行一个）：\n今天天气不错。\n我需要买些杂货。\n数学很有趣。",
        extract_negative_samples_help: "输入不代表目标行为/概念的中性样本。",
        extract_output_config: "输出配置",
        extract_output_path_label: "输出文件路径",
        extract_output_path_help: "提取的控制向量将保存的路径。",
        extract_vector_name_label: "向量名称",
        extract_start_btn: "提取向量",
        extract_response_title: "提取结果",
        extract_sample_pairs_label: "样本对",
        add_sample_btn: "添加样本对",
        no_samples_message: "尚未添加样本对。点击\"添加样本对\"添加第一个样本对。",
        no_samples_error: "请添加至少一个正样本和一个负样本。",
        sample_positive_label: "正样本:",
        sample_negative_label: "负样本:",
        sample_positive_placeholder: "输入正样本文本...",
        sample_negative_placeholder: "输入负样本文本...",

        // 动态添加的JS字符串
        generating: "正在生成...",
        error_select_config: "请先选择一个配置",
        importing_config: "正在导入 {configName}...",
        import_success_message: "配置导入成功",
        import_success_description: "{configName} 配置已成功导入。",
        import_fail_error: "导入配置失败",
        confirm_restart: "确定要重启后端吗？这将中断当前所有操作并释放模型。",
        restarting_backend: "正在重启后端...",
        restart_success_message: "后端重启成功",
        restart_success_description: "后端已重启，可以加载新模型。",
        restart_fail_error: "后端重启失败",
        train_data_format_error: "训练数据格式错误",
        training_in_progress: "正在训练...",
        training_failed_error: "训练失败",
        initializing_training: "正在初始化训练...",
        waiting_for_training: "等待训练开始...",
        error_select_train_config: "请先选择一个训练配置",
        importing_train_config: "正在导入训练配置 {configName}...",
        train_import_success_message: "训练配置导入成功",
        train_import_success_description: "{configName} 训练配置已成功导入。",
        train_import_fail_error: "导入训练配置失败",
        error_select_extract_config: "请先选择一个提取配置",
        importing_extract_config: "正在导入提取配置 {configName}...",
        extract_import_success_message: "提取配置导入成功",
        extract_import_success_description: "{configName} 提取配置已成功导入。",
        extract_import_fail_error: "导入提取配置失败",
        sae_path_error: "SAE方法需要提供SAE参数文件路径",
        extracting_in_progress: "正在提取...",
        extraction_failed_error: "提取失败",
        status_label: "状态",
        extraction_complete: "提取完成",
        output_file_label: "输出文件",
        metadata_label: "元数据",
        initializing_extraction: "正在初始化提取...",
        layers_extracted_label: "已提取层数",
        waiting_for_extraction: "等待提取开始..."
    },
    
    en: {
        // 页面标题和副标题
        subtitle: "Steer Vector Control Panel",
        
        // 主要区块标题
        basic_config: "Steer Vector Configuration",
        model_config: "Model & Sampling Configuration",
        sampling_config: "Sampling Parameters",
        layer_config: "Layer Configuration",
        trigger_config: "Trigger Configuration",
        
        // 页面导航
        nav_inference: "Inference",
        nav_training: "Training",
        nav_extraction: "Extraction",
        nav_chat: "Chat",
        
        // 推理子导航
        single_vector_mode: "Single Vector",
        multi_vector_mode: "Multi Vector",
        sae_explore_mode: "SAE Explore",
        add_vector_btn: "Add Vector Configuration",
        conflict_resolution_label: "Conflict Resolution Method",
        conflict_sequential: "Sequential (Apply in order)",
        conflict_priority: "Priority (Only use the first conflicting vector)",
        conflict_error: "Error (Disallowed conflicts)",
        no_vector_configs_error: "Please add at least one vector configuration",
        
        // SAE 探索相关
        sae_search_btn: "Search Features",
        sae_model_id_label: "Model ID",
        sae_id_label: "SAE ID",
        sae_query_label: "Search Query",
        sae_searching: "Searching for features...",
        sae_search_results: "Search Results",
        sae_no_results: "No features found",
        sae_error: "Search Error",
        sae_api_key_label: "API Key (Required)",
        sae_api_key_placeholder: "Enter Neuronpedia API key",
        sae_api_key_help: "Enter your Neuronpedia API key, it will be saved locally",
        sae_search_by_query: "Semantic Query",
        sae_search_by_index: "Feature Index Query",
        sae_feature_index_label: "Feature Index",
        sae_feature_index_placeholder: "Enter feature index (e.g., 8614)",
        sae_feature_index_help: "Enter the feature index number to query",
        sae_view_feature_details: "View Feature Details",
        sae_query_help: "Enter a semantic search query to find related features",
        sae_api_key_required: "API Key is required",
        sae_feature_index_required: "Please enter a feature index",
        sae_search_failed: "Feature search failed",
        sae_feature_details_failed: "Failed to get feature details",
        sae_feature_id_column: "Feature ID",
        sae_description_column: "Description",
        sae_similarity_column: "Similarity",
        sae_actions_column: "Actions",
        sae_no_description: "No description",
        sae_view_details_label: "View Feature Details",
        sae_enter_feature_id: "Enter Feature ID",
        sae_feature_details: "Feature Details",
        sae_unknown: "Unknown",
        sae_basic_info: "Basic Info",
        sae_model_label: "Model",
        sae_layer_label: "Layer",
        sae_index_label: "Index",
        sae_description_label: "Description",
        sae_sparsity_label: "Sparsity",
        sae_top_activating_tokens: "Top Activating Tokens",
        sae_top_inhibiting_tokens: "Top Inhibiting Tokens",
        sae_activation_example: "Activation Example",
        sae_max_value_label: "Max Value",
        sae_trigger_token_label: "Trigger Token",
        sae_context_label: "Context",
        close_btn: "Close",
        sae_enter_query_prompt: "Please enter query conditions and click search",
        sae_enter_index_prompt: "Please enter a feature index and click search",
        
        // 模型配置标签
        model_path_label: "Model Path",
        model_path_placeholder: "e.g., /path/to/Qwen2.5-1.5B-Instruct/",
        gpu_devices_label: "GPU Device IDs",
        gpu_devices_placeholder: "e.g., 0,1,2 or single GPU: 0",
        normalize_sv_label: "Normalize Steer Vector",
        instruction_label: "Input Instruction",
        instruction_placeholder: "Enter your prompt or question",
        
        // Sampling parameters labels
        temperature_label: "Temperature",
        max_tokens_label: "Max Tokens",
        repetition_penalty_label: "Repetition Penalty",
        
        // Basic configuration labels
        sv_name_label: "Steer Vector Name",
        sv_name_placeholder: "Enter steer vector name",
        sv_id_label: "Steer Vector ID",
        sv_id_placeholder: "Enter unique ID",
        file_path_label: "Local File Path",
        scale_factor_label: "Scale Factor",
        algorithm_label: "Algorithm Selection",
        algorithm_direct: "Direct Algorithm",
        algorithm_erase: "Erase (project out the direction)",
        algorithm_replace: "Replace (overwrite along the direction)",
        algorithm_concept_replace: "Concept Replace (swap concept pair)",
        algorithm_linear: "Linear (affine map)",
        algorithm_lm_steer: "LM-Steer (low-rank projector pair)",
        algorithm_loreft: "LoReFT (low-rank representation finetuning)",
        algorithm_moe_router: "MoE Router (expert (de)activation)",
        
        // Layer configuration
        target_layers_label: "Target Layers",
        target_layers_placeholder: "e.g., 0,1,2,3 or leave empty to apply to all layers",
        target_layers_help: "Enter layer indices separated by commas. Leave empty to apply to all layers.",
        
        // Trigger configuration
        prefill_trigger_title: "Prefill Phase Triggers",
        prefill_tokens_label: "Prefill Trigger Token IDs",
        prefill_tokens_placeholder: "e.g., 100,200,300 or -1 to apply to all",
        prefill_tokens_help: "Enter token IDs separated by commas. Use -1 to apply to all tokens.",
        prefill_positions_label: "Prefill Trigger Positions",
        prefill_positions_placeholder: "e.g., 0,1,-1 (supports negative indexing)",
        prefill_positions_help: "Enter position indices separated by commas. Supports negative indexing (-1 for last position).",
        generate_trigger_title: "Generation Phase Triggers",
        generate_tokens_label: "Generate Trigger Token IDs",
        generate_tokens_placeholder: "e.g., 400,500,600 or -1 to apply to all",
        generate_tokens_help: "Enter token IDs separated by commas. Use -1 to apply to all tokens.",
        
        // Debug options
        enable_debug: "Enable Debug Mode",
        debug_help: "When enabled, debug information will be printed during forward propagation.",
        
        // Training related
        train_model_config: "Training Model Configuration",
        train_output_dir_label: "Output Directory",
        train_output_dir_placeholder: "e.g., ./results/my_training",
        train_reft_config: "ReFT Configuration",
        train_layer_label: "Target Layer",
        train_layer_help: "Layer index where the intervention will be applied.",
        train_component_label: "Component",
        train_component_block_output: "Block Output",
        train_component_attention_output: "Attention Output",
        train_component_mlp_output: "MLP Output",
        train_component_help: "The component of the layer to apply intervention.",
        train_low_rank_dim_label: "Low Rank Dimension",
        train_low_rank_dim_help: "Dimension of the low-rank adaptation. Lower values use less memory.",
        train_params_config: "Training Parameters",
        train_epochs_label: "Number of Epochs",
        train_batch_size_label: "Batch Size",
        train_learning_rate_label: "Learning Rate",
        train_logging_steps_label: "Logging Steps",
        train_save_steps_label: "Save Steps",
        train_save_steps_help: "Number of steps between checkpoint saves. Higher values save less frequently.",
        train_data_config: "Training Data",
        train_examples_label: "Training Examples",
        train_examples_help: "Add training examples as input-output pairs.",
        add_example_btn: "Add Example",
        no_examples_message: "No training examples added yet. Click \"Add Example\" to add your first example.",
        no_examples_error: "Please add at least one training example.",
        example_input_label: "Input:",
        example_output_label: "Output:",
        example_input_placeholder: "Enter input text...",
        example_output_placeholder: "Enter expected output...",
        train_config_select_help: "Select and import a pre-configured training setup.",
        train_start_btn: "Start Training",
        train_response_title: "Training Result",
        
        // Extraction related
        extract_config_select_help: "Select and import a pre-configured extraction setup.",
        train_progress_title: "Training Progress",
        train_current_epoch: "Current Epoch",
        train_current_step: "Current Step",
        train_current_loss: "Loss Value",
        train_learning_rate: "Learning Rate", 
        train_estimated_time: "Estimated Time Remaining",
        train_logs_title: "Training Logs",
        train_waiting: "Waiting for training to start...",
        
        // Buttons
        submit_btn: "Generate",
        reset_btn: "Reset",

        import_config_btn: "Import",
        restart_backend_btn: "Restart Backend",
        
        // Configuration selection
        config_select_label: "Import Configuration",
        config_select_placeholder: "-- Select a configuration --",
        config_emoji_loreft: "Emoji LoReft Configuration",
        config_emotion_direct: "Emotion Direct Configuration",
        config_select_help: "Select and import a pre-configured steer vector setup.",
        
        // Response titles
        response_title: "Generation Result",
        error_title: "Error Message",
        baseline_title: "Baseline (Without Steering)",
        steered_title: "Steered Response",
        
        // Error and info messages
        required_fields_error: "Please fill in all required fields: Name, ID, File Path, Model Path, and Instruction",
        network_error: "Network error: Unable to connect to server. Please ensure the backend service is running.",
        submit_failed: "Submission failed",

        // 提取相关
        extract_model_config: "Model Configuration",
        extract_method_config: "Extraction Method",
        extract_method_label: "Method Selection",
        extract_method_lat: "LAT - Linear Algebraic Technique",
        extract_method_pca: "PCA - Principal Component Analysis",
        extract_method_sae: "SAE - Sparse Autoencoder",
        extract_method_diffmean: "DiffMean - Mean Difference",
        extract_target_layer_label: "Target Layer",
        extract_target_layer_help: "Specify a single layer index or leave empty to extract from all layers.",
        extract_token_pos_label: "Token Position",
        extract_token_last: "Last Token",
        extract_token_first: "First Token",
        extract_token_mean: "Mean of All Tokens",
        extract_token_max: "Max Norm Token",
        extract_sae_params_label: "SAE Parameters Path",
        extract_sae_params_help: "Path to the pre-trained SAE parameters file.",
        extract_sae_combination_label: "Feature Combination Mode",
        extract_sae_weighted_all: "Weighted All Features",
        extract_sae_weighted_top_k: "Weighted Top-K Features",
        extract_sae_single_top: "Single Top Feature",
        extract_sae_top_k_label: "Top K Features",
        extract_normalize_label: "Normalize Vector",
        extract_data_config: "Sample Data",
        extract_positive_samples_label: "Positive Samples",
        extract_positive_samples_placeholder: "Enter positive samples (one per line):\nI love puppies!\nDogs are wonderful companions.\nMy dog brings me so much joy.",
        extract_positive_samples_help: "Enter samples that represent the behavior/concept you want to enhance.",
        extract_negative_samples_label: "Negative Samples",
        extract_negative_samples_placeholder: "Enter negative samples (one per line):\nThe weather is nice today.\nI need to buy groceries.\nMathematics is interesting.",
        extract_negative_samples_help: "Enter neutral samples that don't represent the target behavior/concept.",
        extract_output_config: "Output Configuration",
        extract_output_path_label: "Output File Path",
        extract_output_path_help: "Path where the extracted control vector will be saved.",
        extract_vector_name_label: "Vector Name",
        extract_start_btn: "Extract Vector",
        extract_response_title: "Extraction Result",
        extract_sample_pairs_label: "Sample Pairs",
        add_sample_btn: "Add Sample Pair",
        no_samples_message: "No sample pairs added yet. Click \"Add Sample Pair\" to add your first pair.",
        no_samples_error: "Please add at least one positive and one negative sample.",
        sample_positive_label: "Positive Sample:",
        sample_negative_label: "Negative Sample:",
        sample_positive_placeholder: "Enter positive sample text...",
        sample_negative_placeholder: "Enter negative sample text...",

        // Dynamically added JS strings
        generating: "Generating...",
        error_select_config: "Please select a configuration first",
        importing_config: "Importing {configName}...",
        import_success_message: "Configuration imported successfully",
        import_success_description: "{configName} has been imported successfully.",
        import_fail_error: "Failed to import configuration",
        confirm_restart: "Are you sure you want to restart the backend? This will interrupt all current operations and release the model.",
        restarting_backend: "Restarting backend...",
        restart_success_message: "Backend restarted successfully",
        restart_success_description: "The backend has been restarted. You can now load a new model.",
        restart_fail_error: "Failed to restart backend",
        train_data_format_error: "Incorrect training data format",
        training_in_progress: "Training...",
        training_failed_error: "Training failed",
        initializing_training: "Initializing training...",
        waiting_for_training: "Waiting for training to start...",
        error_select_train_config: "Please select a training configuration first",
        importing_train_config: "Importing training configuration {configName}...",
        train_import_success_message: "Training configuration imported successfully",
        train_import_success_description: "{configName} training configuration has been imported successfully.",
        train_import_fail_error: "Failed to import training configuration",
        error_select_extract_config: "Please select an extraction configuration first",
        importing_extract_config: "Importing extraction configuration {configName}...",
        extract_import_success_message: "Extraction configuration imported successfully",
        extract_import_success_description: "{configName} extraction configuration has been imported successfully.",
        extract_import_fail_error: "Failed to import extraction configuration",
        sae_path_error: "SAE method requires the path to the SAE parameters file",
        extracting_in_progress: "Extracting...",
        extraction_failed_error: "Extraction failed",
        status_label: "Status",
        extraction_complete: "Extraction complete",
        output_file_label: "Output File",
        metadata_label: "Metadata",
        initializing_extraction: "Initializing extraction...",
        layers_extracted_label: "Layers Extracted",
        waiting_for_extraction: "Waiting for extraction to start..."
    }
};

// Current language (defaults to reading from localStorage, otherwise English)
let currentLanguage = localStorage.getItem('language') || 'en';

// Apply translations
function applyTranslations() {
    // Update elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        if (translations[currentLanguage][key]) {
            element.textContent = translations[currentLanguage][key];
        }
    });
    
    // Update elements with data-i18n-placeholder attribute
    document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
        const key = element.getAttribute('data-i18n-placeholder');
        if (translations[currentLanguage][key]) {
            element.placeholder = translations[currentLanguage][key];
        }
    });
    
    // Update HTML lang attribute
    document.documentElement.lang = currentLanguage === 'zh' ? 'zh-CN' : 'en';
}

// Switch language
function changeLanguage(lang) {
    currentLanguage = lang;
    localStorage.setItem('language', lang);
    
    // Update language button state
    document.querySelectorAll('.lang-btn').forEach(btn => {
        if (btn.getAttribute('data-lang') === lang) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    // Apply translations
    applyTranslations();
}

// Get translated text
function t(key) {
    return translations[currentLanguage][key] || key;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Set initial language button state
    document.querySelectorAll('.lang-btn').forEach(btn => {
        if (btn.getAttribute('data-lang') === currentLanguage) {
            btn.classList.add('active');
        }
    });
    
    // Apply translations
    applyTranslations();
}); 