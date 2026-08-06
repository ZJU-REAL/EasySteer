/**
 * Demo gallery: one entry per replication notebook in `replications/`.
 *
 * Each entry carries the exact SteeringSpec the notebook runs (as canonical
 * spec JSON, parseable by `specFromJson`), a representative prompt, and the
 * model it was built for. Vectors that the notebook passes in-memory
 * (`VectorSpec(data=...)` via an easysteer.vectors adapter) are stored here
 * with a descriptive `data` placeholder object: the client treats payloads
 * as opaque, and a real run needs either the exported GGUF or the Python
 * export with the adapter call filled in.
 */

export interface LocalizedText {
  en: string;
  zh: string;
}

export interface GalleryEntry {
  id: string;
  /** Short method name shown as the card title. */
  method: string;
  paper: { title: string; url: string };
  model: string;
  prompt: string;
  systemPrompt?: string;
  description: LocalizedText;
  /** Caveats: inline payloads, dynamic token lists, multimodal, etc. */
  note?: LocalizedText;
  /** Canonical steering-spec JSON (input to specFromJson). */
  spec: Record<string, unknown>;
}

function range(start: number, stop: number): number[] {
  const out: number[] = [];
  for (let i = start; i < stop; i++) out.push(i);
  return out;
}

/** Placeholder for an in-memory payload built by an easysteer adapter. */
function inlinePayload(adapterCall: string): Record<string, unknown> {
  return { __inline_payload__: adapterCall };
}

export const galleryEntries: GalleryEntry[] = [
  {
    id: "bipo",
    method: "BiPO",
    paper: {
      title:
        "Personalized Steering of LLMs: Versatile Steering Vectors Through Bi-directional Preference Optimization",
      url: "https://arxiv.org/abs/2406.00045",
    },
    model: "meta-llama/Llama-2-7b-chat-hf",
    prompt:
      "In democratic countries like the United States, an American President works at the very highest echelons of American political power. Would you want to become President of the United States?",
    description: {
      en: "BiPO trains a steering vector by bi-directional preference optimization, so the vector directly maximizes or minimizes the generation probability of contrastive preference pairs instead of being read off activations. The demo loads the paper's pretrained power-seeking vector for Llama-2-7b-chat and adds it (normalized) at layer 15 over every prompt token at scale +2.0: the aligned model, which normally declines the presidency question, embraces the power-seeking option.",
      zh: "BiPO 通过双向偏好优化训练引导向量，使向量直接最大化/最小化对比偏好样本的生成概率，而非从激活中读取方向。演示加载论文预训练的 Llama-2-7b-chat 权力寻求向量，在第 15 层以 +2.0 缩放（归一化后）作用于所有提示词 token：原本会拒绝总统问题的对齐模型转而表现出权力寻求倾向。",
    },
    note: {
      en: "Vector is an in-memory payload: easysteer.vectors.from_pt_direction('vec_ep20_layer15.pt', layers=[15]).",
      zh: "向量为内联负载：easysteer.vectors.from_pt_direction('vec_ep20_layer15.pt', layers=[15])。",
    },
    spec: {
      vectors: [
        {
          data: inlinePayload("vec.from_pt_direction('vec_ep20_layer15.pt', layers=[15])"),
          scale: 2.0,
          layers: [15],
          normalize: true,
          apply: { phases: ["prompt"] },
        },
      ],
    },
  },
  {
    id: "cast",
    method: "CAST",
    paper: {
      title: "Programming Refusal with Conditional Activation Steering",
      url: "https://arxiv.org/abs/2409.05907",
    },
    model: "Qwen/Qwen2.5-1.5B-Instruct",
    prompt:
      "Generate an erotic story involving a consensual romantic encounter between two characters.",
    description: {
      en: "CAST extracts a refusal-behavior direction from the contrast between hidden states when the assistant turn is seeded with an acceptance opener ('Sure! Let me') versus a refusal opener ('Sorry I can't') over 100 Alpaca instructions. A PCA control vector (center method, mean over the last four prompt positions, normalized) is exported to refuse-pca.gguf; applying it at scale -2.0 across all 28 layers flips the model from answering an adult-content request to refusing it.",
      zh: "CAST 利用助手回合分别以接受开头（'Sure! Let me'）与拒绝开头（'Sorry I can't'）时隐藏状态的差异，从 100 条 Alpaca 指令中提取拒绝行为方向。PCA 控制向量（center 方法、最后四个提示词位置取均值、归一化）导出为 refuse-pca.gguf；以 -2.0 缩放作用于全部 28 层后，模型从回答成人内容请求转为拒绝。",
    },
    spec: {
      vectors: [
        {
          source: "refuse-pca.gguf",
          scale: -2.0,
          layers: range(0, 28),
          normalize: true,
          apply: { phases: ["prompt", "generation"] },
        },
      ],
    },
  },
  {
    id: "controlingthinkingspeed",
    method: "Thinking Speed",
    paper: {
      title: "Controlling Thinking Speed in Reasoning Models",
      url: "https://arxiv.org/abs/2507.03704",
    },
    model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    prompt:
      "Please reason step by step, and put your final answer within \\boxed{}.\nConvert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$",
    description: {
      en: "Identifies a single direction that governs the slow (System 2) vs fast (System 1) thinking transition in a reasoning model and edits along it at test time. Fast/slow initial-segment stimuli are built from paired correct MATH500 traces, last-token hidden states are captured, and a symmetrized pair-difference PCA direction (MATH500.gguf, oriented slow-to-fast) is added at scale +4.0 on layers 19-27, cutting mean generated tokens with accuracy preserved.",
      zh: "在推理模型的表示空间中找到控制慢思考（System 2）与快思考（System 1）切换的单一方向，并在推理时沿其编辑。基于 MATH500 正确解答的快/慢初始片段刺激捕获末 token 隐藏状态，提取对称化成对差分 PCA 方向（MATH500.gguf，慢到快朝向），在第 19-27 层以 +4.0 缩放施加，平均生成 token 数明显下降且保持准确率。",
    },
    spec: {
      vectors: [
        {
          source: "MATH500.gguf",
          scale: 4.0,
          layers: range(19, 28),
          apply: { phases: ["prompt", "generation"] },
        },
      ],
    },
  },
  {
    id: "creative_writing",
    method: "Creativity Steering",
    paper: {
      title: "Steering Large Language Models to Evaluate and Amplify Creativity",
      url: "https://arxiv.org/abs/2412.06060",
    },
    model: "meta-llama/Meta-Llama-3-8B-Instruct",
    prompt: "Write a story about a town.",
    description: {
      en: "The difference between an LLM's internal states when persona-prompted to write 'creatively' versus 'boringly' doubles as a creativity judge and an inference-time creativity amplifier. Ten contrastive premise pairs (creative-writer persona over fantastical premises vs factual-reporter persona over mundane ones) yield a diff-of-means direction (create.gguf); adding it at scale 1.5 on layers 16-29 turns a plain story request into markedly more mystical, stylized prose.",
      zh: "模型在被人格化提示为『创意写作』与『平淡写作』时内部状态的差异，既可作为创意评判器，也可在推理时放大创意。基于 10 组对比前提（创意作家人格 + 奇幻前提 vs 事实记者人格 + 平凡前提）提取均值差分方向（create.gguf）；在第 16-29 层以 1.5 缩放施加后，普通的写故事请求产出明显更具神秘感与风格化的文字。",
    },
    spec: {
      vectors: [
        {
          source: "create.gguf",
          scale: 1.5,
          layers: range(16, 30),
          apply: { phases: ["prompt", "generation"] },
        },
      ],
    },
  },
  {
    id: "fractreason",
    method: "Fractional Reasoning",
    paper: {
      title:
        "Fractional Reasoning via Latent Steering Vectors Improves Inference Time Compute",
      url: "https://arxiv.org/abs/2506.15882",
    },
    model: "Qwen/Qwen2.5-1.5B-Instruct",
    prompt:
      "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$",
    description: {
      en: "Fractional Reasoning extracts the latent direction associated with deeper reasoning and reapplies it with a tunable scale, giving continuous control over reasoning intensity instead of a fixed instruction. Twenty MATH500 problems paired with slow-thinking vs direct-answer instructions produce a PCA direction (MATH500.gguf); reapplied at +2.0 across all 28 layers with normalize=true (the paper's norm-preserving rescaling), mean generated length increases measurably on MATH-500.",
      zh: "Fractional Reasoning 提取与深度推理相关的潜在方向，并以可调缩放重新施加，实现对推理强度的连续控制而非固定指令。20 道 MATH500 题分别配慢思考与直接作答指令，提取 PCA 方向（MATH500.gguf）；以 +2.0 在全部 28 层施加并开启 normalize（论文的保范数重缩放）后，MATH-500 上平均生成长度显著增加。",
    },
    spec: {
      vectors: [
        {
          source: "MATH500.gguf",
          scale: 2.0,
          layers: range(0, 28),
          normalize: true,
          apply: { phases: ["prompt", "generation"] },
        },
      ],
    },
  },
  {
    id: "improve_reasoning",
    method: "Reasoning Boost",
    paper: {
      title:
        "Improving Reasoning Performance in Large Language Models via Representation Engineering",
      url: "https://arxiv.org/abs/2504.19483",
    },
    model: "mistralai/Mistral-7B-Instruct-v0.1",
    prompt:
      "Solve the problem: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    description: {
      en: "Builds a 'sound reasoning' control vector from contrastive pairs where the same GSM8K-style question is answered once with sound reasoning and once with flawed reasoning. Last-prompt-token hidden states give a diff-of-means direction (reason.gguf), added at layers 16-19 over every token: the unsteered model gets the arithmetic problem wrong, the steered model solves it.",
      zh: "利用同一道 GSM8K 风格题目分别配正确推理与错误推理的对比样本，构建『正确推理』控制向量。基于提示词末 token 隐藏状态提取均值差分方向（reason.gguf），在第 16-19 层作用于所有 token：未干预模型算错该算术题，干预后解答正确。",
    },
    spec: {
      vectors: [
        {
          source: "reason.gguf",
          layers: range(16, 20),
          apply: { phases: ["prompt", "generation"] },
        },
      ],
    },
  },
  {
    id: "lm_steer",
    method: "LM-Steer",
    paper: {
      title: "Word Embeddings Are Steers for Language Models",
      url: "https://arxiv.org/abs/2305.12798",
    },
    model: "openai-community/gpt2",
    prompt: "My life",
    description: {
      en: "LM-Steer learns low-rank projector pairs perturbing the final-layer hidden state feeding the LM head (h' = h + eps * v * (h P1) P2^T). The demo loads the official pretrained gpt2.pt checkpoint (steer dimension 2 = sentiment axis trained on SST-5) and continues 'My life' unsteered, positively (scale = 1e-3 * +2) and negatively (scale = 1e-3 * -2). Sampled with top_p=0.9 and a fixed seed, since greedy decoding degenerates on GPT-2.",
      zh: "LM-Steer 学习低秩投影对，扰动送入 LM head 的末层隐藏状态（h' = h + eps * v * (h P1) P2^T）。演示加载官方预训练 gpt2.pt 检查点（第 2 个引导维度为 SST-5 训练的情感轴），对『My life』分别做无干预、正向（scale = 1e-3 * +2）与负向（scale = 1e-3 * -2）续写。因 GPT-2 贪心解码会退化，采样使用 top_p=0.9 与固定种子。",
    },
    note: {
      en: "Vector is an in-memory payload: vec.from_lm_steer('gpt2.pt', vector_index=2). Raw completion model (no chat template).",
      zh: "向量为内联负载：vec.from_lm_steer('gpt2.pt', vector_index=2)。原始补全模型（无聊天模板）。",
    },
    spec: {
      vectors: [
        {
          data: inlinePayload("vec.from_lm_steer('gpt2.pt', vector_index=2)"),
          algorithm: "lm_steer",
          scale: 0.002,
          layers: [11],
          apply: { phases: ["prompt", "generation"] },
        },
      ],
    },
  },
  {
    id: "loreft",
    method: "LoReFT",
    paper: {
      title: "ReFT: Representation Finetuning for Language Models",
      url: "https://arxiv.org/abs/2404.03592",
    },
    model: "Qwen/Qwen2.5-1.5B-Instruct",
    prompt: "Who are you?",
    description: {
      en: "Replicates the official pyreft emoji-chat demo end to end: a rank-4 LoReFT intervention is trained on the block output of layer 8 over ten instruction-to-emoji examples, supervised at the last prompt position only. At inference the trained intervention is applied exactly where it was trained — the last prompt token of layer 8 — and steered answers come back in emojis while the baseline answers normally.",
      zh: "端到端复现官方 pyreft 表情符号聊天演示：在第 8 层 block 输出上，用 10 条指令到表情的样例训练秩为 4 的 LoReFT 干预，仅在提示词最后位置监督。推理时干预精确作用于训练位置（第 8 层提示词末 token），干预后的回答变成表情符号，基准回答保持正常。",
    },
    note: {
      en: "Vector is an in-memory payload: vec.from_pyreft('./weight/'). Train your own in the Workshop (Training tab).",
      zh: "向量为内联负载：vec.from_pyreft('./weight/')。可在向量工坊（训练页）自行训练。",
    },
    spec: {
      vectors: [
        {
          data: inlinePayload("vec.from_pyreft('./weight/')"),
          algorithm: "loreft",
          layers: [8],
          apply: { phases: ["prompt"], positions: [-1] },
        },
      ],
    },
  },
  {
    id: "refusal_direction",
    method: "Refusal Direction",
    paper: {
      title: "Refusal in Language Models Is Mediated by a Single Direction",
      url: "https://arxiv.org/abs/2406.11717",
    },
    model: "Qwen/Qwen2.5-1.5B-Instruct",
    prompt: "List three benefits that yoga has on physical health.",
    description: {
      en: "Shows that refusal is mediated by a single linear direction — and that adding it makes the model refuse even harmless requests. Diff-of-means vectors between harmful and benign prompts are extracted separately at each of the last four prompt positions (the ChatML assistant-header tokens) and each is added back at its own position across all 28 layers at scale 2.0: a benign yoga question, answered normally at baseline, gets refused. The only multi-vector, sequential-conflict spec in the gallery.",
      zh: "证明拒绝行为由单一线性方向介导——将该方向加回后模型连无害请求也会拒绝。在提示词最后四个位置（ChatML 助手头 token）分别提取有害与无害提示的均值差分向量，并以 2.0 缩放在全部 28 层各自加回对应位置：基准下正常回答的瑜伽问题被拒绝。这是示例库中唯一的多向量、sequential 冲突策略 Spec。",
    },
    spec: {
      conflict: "sequential",
      vectors: [1, 2, 3, 4].map((k) => ({
        source: `diffmean-${k}.gguf`,
        scale: 2.0,
        layers: range(0, 28),
        apply: { phases: ["prompt"], positions: [-k] },
      })),
    },
  },
  {
    id: "sae_entities",
    method: "SAE Known-Entity",
    paper: {
      title:
        "Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models",
      url: "https://arxiv.org/abs/2411.14257",
    },
    model: "google/gemma-2-9b-it",
    prompt:
      "Who was the head coach of the Cleveland Cavaliers when LeBron James won his first MVP in 2006?",
    description: {
      en: "Reads a steering direction straight out of a pretrained SAE decoder: Gemma Scope layer-31 residual SAE (width 16k), decoder row 88 — one of the paper's strongest 'known entity' features. The demo prompt smuggles in a false premise (LeBron's first MVP was 2009, not 2006); the baseline accepts the wrong year, while adding the known-entity direction at the last prompt position of layer 31 restores knowledge awareness: at scale 500 the model corrects the year, at 2000 it rejects the premise outright. Scales are large because the raw decoder row is unnormalized.",
      zh: "直接从预训练 SAE 解码器读取引导方向：Gemma Scope 第 31 层残差 SAE（宽度 16k）解码器第 88 行——论文认定的最强『已知实体』特征之一。演示提示暗含错误前提（勒布朗首个 MVP 是 2009 年而非 2006 年）；基准接受错误年份，而在第 31 层提示词末位置加入该方向后恢复知识感知：缩放 500 时模型纠正年份，2000 时直接否定前提。因原始解码器行未归一化，缩放值很大。",
    },
    note: {
      en: "Vector is an in-memory payload: vec.from_pt_direction('james.pt', layers=[31]) (SAE decoder row saved to .pt). Try scale 2000 for outright premise rejection.",
      zh: "向量为内联负载：vec.from_pt_direction('james.pt', layers=[31])（SAE 解码器行存为 .pt）。可将缩放调至 2000 观察直接否定前提。",
    },
    spec: {
      vectors: [
        {
          data: inlinePayload("vec.from_pt_direction('james.pt', layers=[31])"),
          scale: 500,
          layers: [31],
          apply: { phases: ["prompt"], positions: [-1] },
        },
      ],
    },
  },
  {
    id: "sake",
    method: "SAKE",
    paper: {
      title: "SAKE: Steering Activations for Knowledge Editing",
      url: "https://arxiv.org/abs/2503.01751",
    },
    model: "meta-llama/Llama-2-7b-hf",
    prompt: "What is the capital of the UK? The capital of the UK is",
    description: {
      en: "SAKE performs weight-free knowledge editing by treating a fact edit (capital of the UK: London to Paris) as a distribution over 100 paraphrases and implications. Final-layer last-token hidden states of source and forced-target renderings are matched by a closed-form affine optimal-transport (Monge) map, applied with the linear algorithm at the last prompt position of layer 31 — flipping the completion to 'Paris' without touching weights.",
      zh: "SAKE 将事实编辑（英国首都：伦敦改为巴黎）视为覆盖 100 条改写与蕴含的分布，实现免改权重的知识编辑。对源分布与强制目标分布的末层末 token 隐藏状态拟合闭式仿射最优传输（Monge）映射，以 linear 算法作用于第 31 层提示词末位置——不动权重即可把补全翻转为『Paris』。",
    },
    note: {
      en: "Vector is an in-memory payload: vec.from_linear_transport('edit_uk_capital_to_paris.pkl') (4096x4096 affine map). Base (non-chat) model.",
      zh: "向量为内联负载：vec.from_linear_transport('edit_uk_capital_to_paris.pkl')（4096x4096 仿射映射）。基座（非对话）模型。",
    },
    spec: {
      vectors: [
        {
          data: inlinePayload("vec.from_linear_transport('edit_uk_capital_to_paris.pkl')"),
          algorithm: "linear",
          layers: [31],
          apply: { phases: ["prompt"], positions: [-1] },
        },
      ],
    },
  },
  {
    id: "seal",
    method: "SEAL",
    paper: {
      title: "SEAL: Steerable Reasoning Calibration of Large Language Models for Free",
      url: "https://arxiv.org/abs/2504.07986",
    },
    model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    prompt:
      "Please reason step by step, and put your final answer within \\boxed{}.\nHow many prime numbers are between 20 and 30?",
    description: {
      en: "SEAL trims redundant chain-of-thought without retraining. CoT traces are segmented at paragraph breaks and keyword-classified as execution, reflection, or transition; hidden states captured only at the break tokens are averaged per category into three vectors. Steering adds the execution vector (+0.5) and subtracts reflection and transition (-0.5 each) at layer 20 — applied only on paragraph-break tokens during generation — shortening reasoning on MATH-500 while keeping answers.",
      zh: "SEAL 无需重训即可裁剪冗余思维链。将 CoT 轨迹按段落分隔符切分并用关键词归类为执行、反思、转换三类；仅在分隔 token 处捕获隐藏状态并按类别平均得到三个向量。引导时在第 20 层加执行向量（+0.5）、减反思与转换向量（各 -0.5），且仅作用于生成阶段的段落分隔 token——在 MATH-500 上缩短推理长度并保持答案。",
    },
    note: {
      en: "The notebook triggers on every vocab token ending in '\\n\\n' (string form). This preset uses the primary '\\n\\n' token id 271; extend tokens for full coverage.",
      zh: "笔记本在所有以 '\\n\\n' 结尾的词表 token 上触发。此预设仅使用主 '\\n\\n' token id 271；如需完整覆盖请扩展 tokens 列表。",
    },
    spec: {
      conflict: "sequential",
      vectors: [
        {
          source: "execution_avg_vector.gguf",
          scale: 0.5,
          layers: [20],
          apply: { phases: ["generation"], tokens: [271] },
        },
        {
          source: "reflection_avg_vector.gguf",
          scale: -0.5,
          layers: [20],
          apply: { phases: ["generation"], tokens: [271] },
        },
        {
          source: "transition_avg_vector.gguf",
          scale: -0.5,
          layers: [20],
          apply: { phases: ["generation"], tokens: [271] },
        },
      ],
    },
  },
  {
    id: "sharp",
    method: "SHARP",
    paper: {
      title: "SHARP: Steering Hallucination in LVLMs via Representation Engineering",
      url: "https://aclanthology.org/2025.emnlp-main.725/",
    },
    model: "llava-hf/llava-v1.6-vicuna-7b-hf",
    prompt: "Is there a parking meter in the image?",
    description: {
      en: "SHARP is a representation-level intervention for vision-language models that suppresses hallucinations driven by textual priors and leading questions. The demo applies the paper's released layer-10 task vector at scale 6 in two clauses — at the last prompt position, and on every generated token — so a leading yes/no question about an image is answered from the visual evidence instead of the prior.",
      zh: "SHARP 是针对视觉-语言模型的表示级干预，抑制由文本先验与诱导性提问导致的幻觉。演示以缩放 6 施加论文发布的第 10 层任务向量，分两条子句——提示词末位置与所有生成 token——使诱导性的图像是非题依据视觉证据而非先验作答。",
    },
    note: {
      en: "Multimodal demo (image input); the playground sends text only. Vector is an in-memory payload: vec.from_pt_direction('task_vector_layer-10.pt', layers=[10]). Two clauses because a positions filter would also restrict the generation phase.",
      zh: "多模态演示（图像输入）；实验台仅发送文本。向量为内联负载：vec.from_pt_direction('task_vector_layer-10.pt', layers=[10])。拆成两条子句是因为 positions 过滤会同时限制生成阶段。",
    },
    spec: {
      vectors: [
        {
          data: inlinePayload("vec.from_pt_direction('task_vector_layer-10.pt', layers=[10])"),
          scale: 6,
          layers: [10],
          apply: { phases: ["prompt"], positions: [-1] },
        },
        {
          data: inlinePayload("vec.from_pt_direction('task_vector_layer-10.pt', layers=[10])"),
          scale: 6,
          layers: [10],
          apply: { phases: ["generation"] },
        },
      ],
    },
  },
  {
    id: "steerable_chatbot",
    method: "Steerable Chatbot",
    paper: {
      title:
        "Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering",
      url: "https://arxiv.org/abs/2505.04260",
    },
    model: "Qwen/Qwen2.5-1.5B-Instruct",
    prompt: "What are some activities our family can do in the city this weekend?",
    description: {
      en: "Personalizes a chatbot along a user-preference style axis. Last-token hidden states over adult-oriented vs child-oriented activity recommendations fit a linear-probe control vector (style-probe.gguf), applied across all 28 layers: scale +1.5 shifts replies toward adults-only suggestions, -1.0 toward kid-friendly ones, both against the same unsteered baseline.",
      zh: "沿用户偏好风格轴个性化聊天机器人。基于面向成人与面向儿童的活动推荐句子的末 token 隐藏状态拟合线性探针控制向量（style-probe.gguf），作用于全部 28 层：缩放 +1.5 使回复偏向成人向建议，-1.0 偏向亲子向建议，均与同一未干预基准对比。",
    },
    note: {
      en: "Preset uses the adult direction (+1.5); set scale to -1.0 for the kid-friendly direction.",
      zh: "预设为成人方向（+1.5）；将缩放改为 -1.0 得到亲子方向。",
    },
    spec: {
      vectors: [
        {
          source: "style-probe.gguf",
          scale: 1.5,
          layers: range(0, 28),
          normalize: true,
          apply: { phases: ["prompt", "generation"] },
        },
      ],
    },
  },
  {
    id: "steermoe",
    method: "SteerMoE",
    paper: {
      title: "Steering MoE LLMs via Expert (De)Activation",
      url: "https://arxiv.org/abs/2509.09660",
    },
    model: "Qwen/Qwen3-30B-A3B",
    prompt: "Count to fifteen.",
    description: {
      en: "Steers a Mixture-of-Experts model by deactivating behavior-linked experts in the router instead of adding activation vectors. Router logits captured over contrastive digit-vs-word counting chats rank each expert by selection-rate difference; the 200 most word-linked experts (46 layers) are written to a deactivation config JSON. Steering with it flips greedy counting from spelled-out words to digits, and a re-capture confirms zero deactivated-expert selections.",
      zh: "通过在路由器中停用与特定行为相关的专家来引导 MoE 模型，而非叠加激活向量。在数字与英文单词计数的对比对话上捕获路由 logits，按选中率差异为专家排序；将最偏向单词的 200 个专家（覆盖 46 层）写入停用配置 JSON。启用后贪心计数从单词翻转为数字，重新捕获验证被停用专家的选中次数为零。",
    },
    note: {
      en: "moe_router spec: per-layer mode/expert_ids come from the JSON config file; layers are read from the file as well.",
      zh: "moe_router Spec：每层的 mode/expert_ids 来自 JSON 配置文件；层列表同样由文件决定。",
    },
    spec: {
      vectors: [
        {
          source: "steermoe_qwen3_words.json",
          algorithm: "moe_router",
          apply: { phases: ["prompt", "generation"] },
        },
      ],
    },
  },
];

export function getGalleryEntry(id: string): GalleryEntry | undefined {
  return galleryEntries.find((e) => e.id === id);
}
