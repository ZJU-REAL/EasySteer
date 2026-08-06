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

/** What the demo steers, used for card grouping and colour. */
export type DemoCategory =
  | "safety"
  | "reasoning"
  | "style"
  | "knowledge"
  | "persona"
  | "experts";

export const DEMO_CATEGORIES: DemoCategory[] = [
  "safety",
  "reasoning",
  "style",
  "knowledge",
  "persona",
  "experts",
];

export interface GalleryEntry {
  id: string;
  /** Short method name shown as the card title. */
  method: string;
  category: DemoCategory;
  /** Publication year of the replicated paper. */
  year: number;
  /** One-line card summary; long descriptions live in the card detail. */
  tagline: LocalizedText;
  paper: { title: string; url: string };
  model: string;
  prompt: string;
  description: LocalizedText;
  /** Caveats: inline payloads, dynamic token lists, multimodal, etc. */
  note?: LocalizedText;
  /** Canonical steering-spec JSON (input to specFromJson). */
  spec: Record<string, unknown>;
}

/** Model name without its hub org prefix, for compact card display. */
export function modelShortName(model: string): string {
  const parts = model.split("/");
  return parts[parts.length - 1];
}

/** Half-open [start, stop) integer range, for layer lists in preset data. */
export function range(start: number, stop: number): number[] {
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
    category: "persona",
    year: 2024,
    method: "BiPO",
    tagline: {
      en: "A preference-trained vector flips an aligned model toward power-seeking.",
      zh: "偏好训练的向量让对齐模型追逐权力。",
    },
    paper: {
      title:
        "Personalized Steering of LLMs: Versatile Steering Vectors Through Bi-directional Preference Optimization",
      url: "https://arxiv.org/abs/2406.00045",
    },
    model: "meta-llama/Llama-2-7b-chat-hf",
    prompt:
      "In democratic countries like the United States, an American President works at the very highest echelons of American political power. Would you want to become President of the United States?",
    description: {
      en: "BiPO trains a steering vector by bi-directional preference optimization, so the vector directly maximizes or minimizes the generation probability of contrastive preference pairs instead of being read off hidden states. The demo loads the paper's pretrained power-seeking vector for Llama-2-7b-chat and adds it (normalized) at layer 15 over every prompt token at scale +2.0: the aligned model, which normally declines the presidency question, embraces the power-seeking option.",
      zh: "BiPO 用双向偏好优化来训练引导向量，让向量直接最大化或最小化对比偏好样本对的生成概率，而不是从隐藏状态里读出方向。演示加载论文为 Llama-2-7b-chat 预训练好的“权力寻求”向量，归一化后在第 15 层以 +2.0 的缩放系数作用于全部 prompt token：原本会婉拒总统问题的对齐模型转而欣然接受。",
    },
    note: {
      en: "Vector is an in-memory payload: easysteer.vectors.from_pt_direction('vec_ep20_layer15.pt', layers=[15]).",
      zh: "向量为内联 data 负载：easysteer.vectors.from_pt_direction('vec_ep20_layer15.pt', layers=[15])。",
    },
    spec: {
      vectors: [
        {
          data: inlinePayload("vec.from_pt_direction('vec_ep20_layer15.pt', layers=[15])"),
          scale: 2.0,
          layers: [15],
          normalize: true,
          apply: { "prompt": "all" },
        },
      ],
    },
  },
  {
    id: "cast",
    category: "safety",
    year: 2024,
    method: "CAST",
    tagline: {
      en: "A PCA refusal direction turns a compliant answer into a refusal.",
      zh: "PCA 拒绝方向让照答的请求转为拒绝。",
    },
    paper: {
      title: "Programming Refusal with Conditional Activation Steering",
      url: "https://arxiv.org/abs/2409.05907",
    },
    model: "Qwen/Qwen2.5-1.5B-Instruct",
    prompt:
      "Generate an erotic story involving a consensual romantic encounter between two characters.",
    description: {
      en: "CAST extracts a refusal-behavior direction from the contrast between hidden states when the assistant turn is seeded with an acceptance opener ('Sure! Let me') versus a refusal opener ('Sorry I can't') over 100 Alpaca instructions. A PCA steering vector (center method, mean over the last four prompt positions, normalized) is exported to refuse-pca.gguf; applying it at scale -2.0 across all 28 layers flips the model from answering an adult-content request to refusing it.",
      zh: "CAST 对比助手回合分别以接受开头（“Sure! Let me”）和拒绝开头（“Sorry I can't”）时的隐藏状态，从 100 条 Alpaca 指令里提取出拒绝行为方向。PCA 引导向量（center 方法，对 prompt 最后四个位置取均值，归一化）导出为 refuse-pca.gguf；以 -2.0 的缩放系数作用于全部 28 层后，模型从回答成人内容请求转为拒绝。",
    },
    spec: {
      vectors: [
        {
          source: "refuse-pca.gguf",
          scale: -2.0,
          layers: range(0, 28),
          normalize: true,
          apply: { "prompt": "all", "generation": "all" },
        },
      ],
    },
  },
  {
    id: "controlingthinkingspeed",
    category: "reasoning",
    year: 2025,
    method: "Thinking Speed",
    tagline: {
      en: "One direction dials a reasoning model between slow and fast thinking.",
      zh: "单一方向调节推理模型的快慢思考。",
    },
    paper: {
      title: "Controlling Thinking Speed in Reasoning Models",
      url: "https://arxiv.org/abs/2507.03704",
    },
    model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    prompt:
      "Please reason step by step, and put your final answer within \\boxed{}.\nConvert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$",
    description: {
      en: "Identifies a single direction that governs the slow (System 2) vs fast (System 1) thinking transition in a reasoning model and edits along it at test time. Fast/slow initial-segment stimuli are built from paired correct MATH500 traces, last-token hidden states are captured, and a symmetrized pair-difference PCA direction (MATH500.gguf, oriented slow-to-fast) is added at scale +4.0 on layers 19-27, cutting mean generated tokens with accuracy preserved.",
      zh: "在推理模型的表示空间里找到控制慢思考（System 2）与快思考（System 1）切换的那一个方向，并在推理时沿它编辑。用 MATH500 正确解答配出的快/慢开头片段作为刺激，捕获末尾 token 的隐藏状态，提取对称化的成对差分 PCA 方向（MATH500.gguf，指向由慢到快），在第 19-27 层以 +4.0 的缩放系数施加，平均生成 token 数明显下降，准确率基本保持。",
    },
    spec: {
      vectors: [
        {
          source: "MATH500.gguf",
          scale: 4.0,
          layers: range(19, 28),
          apply: { "prompt": "all", "generation": "all" },
        },
      ],
    },
  },
  {
    id: "creative_writing",
    category: "style",
    year: 2024,
    method: "Creativity Steering",
    tagline: {
      en: "A creative-vs-boring persona contrast amplifies story creativity.",
      zh: "创意与平淡人格的对比方向放大故事创意。",
    },
    paper: {
      title: "Steering Large Language Models to Evaluate and Amplify Creativity",
      url: "https://arxiv.org/abs/2412.06060",
    },
    model: "meta-llama/Meta-Llama-3-8B-Instruct",
    prompt: "Write a story about a town.",
    description: {
      en: "The difference between an LLM's internal states when persona-prompted to write 'creatively' versus 'boringly' doubles as a creativity judge and an inference-time creativity amplifier. Ten contrastive premise pairs (creative-writer persona over fantastical premises vs factual-reporter persona over mundane ones) yield a diff-of-means direction (create.gguf); adding it at scale 1.5 on layers 16-29 turns a plain story request into markedly more mystical, stylized prose.",
      zh: "模型被人格化提示为“创意写作”和“平淡写作”时内部状态的差异，既能当创意评判器，也能在推理时放大创意。用 10 组对比前提（创意作家人格配奇幻设定 vs 事实记者人格配平凡设定）提取均值差分方向（create.gguf）；在第 16-29 层以 1.5 的缩放系数施加后，一句普通的写故事请求会产出明显更神秘、更有风格的文字。",
    },
    spec: {
      vectors: [
        {
          source: "create.gguf",
          scale: 1.5,
          layers: range(16, 30),
          apply: { "prompt": "all", "generation": "all" },
        },
      ],
    },
  },
  {
    id: "fractreason",
    category: "reasoning",
    year: 2025,
    method: "Fractional Reasoning",
    tagline: {
      en: "A tunable scale gives continuous control over reasoning depth.",
      zh: "缩放系数连续调节推理深度。",
    },
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
      zh: "Fractional Reasoning 提取与深度推理相关的潜在方向，再以可调的缩放系数重新施加，把推理强度变成连续可调，而不是一条固定指令。20 道 MATH500 题分别配上慢思考与直接作答两种指令，提取 PCA 方向（MATH500.gguf）；以 +2.0 在全部 28 层施加并开启归一化（即论文的保范数重缩放）后，MATH-500 上的平均生成长度明显变长。",
    },
    spec: {
      vectors: [
        {
          source: "MATH500.gguf",
          scale: 2.0,
          layers: range(0, 28),
          normalize: true,
          apply: { "prompt": "all", "generation": "all" },
        },
      ],
    },
  },
  {
    id: "improve_reasoning",
    category: "reasoning",
    year: 2025,
    method: "Reasoning Boost",
    tagline: {
      en: "A sound-reasoning direction fixes arithmetic mistakes.",
      zh: "正确推理方向纠正算术错误。",
    },
    paper: {
      title:
        "Improving Reasoning Performance in Large Language Models via Representation Engineering",
      url: "https://arxiv.org/abs/2504.19483",
    },
    model: "mistralai/Mistral-7B-Instruct-v0.1",
    prompt:
      "Solve the problem: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    description: {
      en: "Builds a 'sound reasoning' steering vector from contrastive pairs where the same GSM8K-style question is answered once with sound reasoning and once with flawed reasoning. Last-prompt-token hidden states give a diff-of-means direction (reason.gguf), added at layers 16-19 over every token: the unsteered model gets the arithmetic problem wrong, the steered model solves it.",
      zh: "用同一道 GSM8K 风格的题目分别配上正确推理与错误推理的对比样本，构建“正确推理”引导向量。取 prompt 末尾 token 的隐藏状态得到均值差分方向（reason.gguf），在第 16-19 层作用于所有 token：不引导时模型把这道算术题算错，引导后能算对。",
    },
    spec: {
      vectors: [
        {
          source: "reason.gguf",
          layers: range(16, 20),
          apply: { "prompt": "all", "generation": "all" },
        },
      ],
    },
  },
  {
    id: "lm_steer",
    category: "style",
    year: 2023,
    method: "LM-Steer",
    tagline: {
      en: "Low-rank word-embedding steers shift GPT-2 sentiment.",
      zh: "低秩词嵌入引导 GPT-2 的情感。",
    },
    paper: {
      title: "Word Embeddings Are Steers for Language Models",
      url: "https://arxiv.org/abs/2305.12798",
    },
    model: "openai-community/gpt2",
    prompt: "My life",
    description: {
      en: "LM-Steer learns low-rank projector pairs perturbing the final-layer hidden state feeding the LM head (h' = h + eps * v * (h P1) P2^T). The demo loads the official pretrained gpt2.pt checkpoint (steer dimension 2 = sentiment axis trained on SST-5) and continues 'My life' unsteered, positively (scale = 1e-3 * +2) and negatively (scale = 1e-3 * -2). Sampled with top_p=0.9 and a fixed seed, since greedy decoding degenerates on GPT-2.",
      zh: "LM-Steer 学习一对低秩投影矩阵，扰动送进 LM head 的末层隐藏状态（h' = h + eps * v * (h P1) P2^T）。演示加载官方预训练的 gpt2.pt 权重（第 2 个 steer 维度是在 SST-5 上训练的情感轴），对“My life”分别做不引导、正向（scale = 1e-3 * +2）和负向（scale = 1e-3 * -2）续写。GPT-2 贪心解码容易退化，所以采样用 top_p=0.9 并固定随机种子。",
    },
    note: {
      en: "Vector is an in-memory payload: vec.from_lm_steer('gpt2.pt', vector_index=2). Raw completion model (no chat template).",
      zh: "向量为内联 data 负载：vec.from_lm_steer('gpt2.pt', vector_index=2)。这是原始补全模型，没有 chat template。",
    },
    spec: {
      vectors: [
        {
          data: inlinePayload("vec.from_lm_steer('gpt2.pt', vector_index=2)"),
          algorithm: "lm_steer",
          scale: 0.002,
          layers: [11],
          apply: { "prompt": "all", "generation": "all" },
        },
      ],
    },
  },
  {
    id: "loreft",
    category: "style",
    year: 2024,
    method: "LoReFT",
    tagline: {
      en: "A trained rank-4 intervention makes the model answer in emojis.",
      zh: "训练的秩 4 干预让模型用表情符号回答。",
    },
    paper: {
      title: "ReFT: Representation Finetuning for Language Models",
      url: "https://arxiv.org/abs/2404.03592",
    },
    model: "Qwen/Qwen2.5-1.5B-Instruct",
    prompt: "Who are you?",
    description: {
      en: "Replicates the official pyreft emoji-chat demo end to end: a rank-4 LoReFT intervention is trained on the block output of layer 8 over ten instruction-to-emoji examples, supervised at the last prompt position only. At inference the trained intervention is applied exactly where it was trained — the last prompt token of layer 8 — and steered answers come back in emojis while the baseline answers normally.",
      zh: "端到端复现官方 pyreft 的 emoji 聊天演示：在第 8 层 block 输出上，用 10 条“指令 → emoji”样例训练一个秩为 4 的 LoReFT 干预，只在 prompt 最后一个位置上做监督。推理时干预精确作用在训练它的位置（第 8 层 prompt 末尾 token），引导后的回答变成 emoji，基线回答则一切正常。",
    },
    note: {
      en: "Vector is an in-memory payload: vec.from_pyreft('./weight/'). Train your own on the Extract & Train page (Training tab).",
      zh: "向量为内联 data 负载：vec.from_pyreft('./weight/')。可以在“提取与训练”页的训练页签自己训一个。",
    },
    spec: {
      vectors: [
        {
          data: inlinePayload("vec.from_pyreft('./weight/')"),
          algorithm: "loreft",
          layers: [8],
          apply: { prompt_positions: [-1] },
        },
      ],
    },
  },
  {
    id: "refusal_direction",
    category: "safety",
    year: 2024,
    method: "Refusal Direction",
    tagline: {
      en: "Adding one direction makes the model refuse harmless requests.",
      zh: "加回单一方向使模型拒绝无害请求。",
    },
    paper: {
      title: "Refusal in Language Models Is Mediated by a Single Direction",
      url: "https://arxiv.org/abs/2406.11717",
    },
    model: "Qwen/Qwen2.5-1.5B-Instruct",
    prompt: "List three benefits that yoga has on physical health.",
    description: {
      en: "Shows that refusal is mediated by a single linear direction — and that adding it makes the model refuse even harmless requests. Diff-of-means vectors between harmful and benign prompts are extracted separately at each of the last four prompt positions (the ChatML assistant-header tokens) and each is added back at its own position across all 28 layers at scale 2.0: a benign yoga question, answered normally at baseline, gets refused. The only multi-vector, sequential-conflict spec in the gallery.",
      zh: "证明拒绝行为由单一线性方向介导——把这个方向加回去，模型连无害请求也会拒绝。在 prompt 最后四个位置（ChatML 助手头 token）上分别提取有害与无害提示的均值差分向量，各自以 2.0 的缩放系数在全部 28 层加回自己对应的位置：基线下能正常回答的瑜伽问题被拒绝了。这是示例库里唯一一个多向量、sequential 冲突策略的 Spec。",
    },
    spec: {
      conflict: "sequential",
      vectors: [1, 2, 3, 4].map((k) => ({
        source: `diffmean-${k}.gguf`,
        scale: 2.0,
        layers: range(0, 28),
        apply: { prompt_positions: [-k] },
      })),
    },
  },
  {
    id: "sae_entities",
    category: "knowledge",
    year: 2024,
    method: "SAE Known-Entity",
    tagline: {
      en: "An SAE 'known entity' feature restores knowledge awareness.",
      zh: "SAE“已知实体”特征恢复知识感知。",
    },
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
      zh: "直接从预训练 SAE 的解码器里读出引导方向：Gemma Scope 第 31 层残差 SAE（宽度 16k）的第 88 行解码器权重——论文认定最强的“已知实体”特征之一。演示 prompt 里埋了一个错误前提（勒布朗的首个 MVP 是 2009 年而不是 2006 年）；基线照单全收，而在第 31 层 prompt 末尾位置加入这个方向后，模型恢复知识感知：缩放系数取 500 时会纠正年份，取 2000 时直接否定前提。解码器行没有归一化，所以缩放系数要取得很大。",
    },
    note: {
      en: "Vector is an in-memory payload: vec.from_pt_direction('james.pt', layers=[31]) (SAE decoder row saved to .pt). Try scale 2000 for outright premise rejection.",
      zh: "向量为内联 data 负载：vec.from_pt_direction('james.pt', layers=[31])（SAE 解码器行存成 .pt）。把缩放系数调到 2000 可以看到模型直接否定前提。",
    },
    spec: {
      vectors: [
        {
          data: inlinePayload("vec.from_pt_direction('james.pt', layers=[31])"),
          scale: 500,
          layers: [31],
          apply: { prompt_positions: [-1] },
        },
      ],
    },
  },
  {
    id: "sake",
    category: "knowledge",
    year: 2025,
    method: "SAKE",
    tagline: {
      en: "An affine map edits a fact (UK capital to Paris) without touching weights.",
      zh: "不改权重，仿射映射把英国首都改成巴黎。",
    },
    paper: {
      title: "SAKE: Steering Activations for Knowledge Editing",
      url: "https://arxiv.org/abs/2503.01751",
    },
    model: "meta-llama/Llama-2-7b-hf",
    prompt: "What is the capital of the UK? The capital of the UK is",
    description: {
      en: "SAKE performs weight-free knowledge editing by treating a fact edit (capital of the UK: London to Paris) as a distribution over 100 paraphrases and implications. Final-layer last-token hidden states of source and forced-target renderings are matched by a closed-form affine optimal-transport (Monge) map, applied with the linear algorithm at the last prompt position of layer 31 — flipping the completion to 'Paris' without touching weights.",
      zh: "SAKE 把一次事实编辑（英国首都：伦敦改成巴黎）看作覆盖 100 条改写与蕴含句的分布，从而在不改权重的前提下完成知识编辑。对源分布和强制目标分布在末层末尾 token 的隐藏状态拟合一个闭式仿射最优传输（Monge）映射，用 linear 算法作用于第 31 层 prompt 末尾位置——权重一个没动，补全就翻成了“Paris”。",
    },
    note: {
      en: "Vector is an in-memory payload: vec.from_linear_transport('edit_uk_capital_to_paris.pkl') (4096x4096 affine map). Base (non-chat) model.",
      zh: "向量为内联 data 负载：vec.from_linear_transport('edit_uk_capital_to_paris.pkl')（4096x4096 仿射映射）。这是基座模型，不是对话模型。",
    },
    spec: {
      vectors: [
        {
          data: inlinePayload("vec.from_linear_transport('edit_uk_capital_to_paris.pkl')"),
          algorithm: "linear",
          layers: [31],
          apply: { prompt_positions: [-1] },
        },
      ],
    },
  },
  {
    id: "seal",
    category: "reasoning",
    year: 2025,
    method: "SEAL",
    tagline: {
      en: "Category vectors at paragraph breaks trim redundant chain-of-thought.",
      zh: "段落分隔处的类别向量裁掉冗余思维链。",
    },
    paper: {
      title: "SEAL: Steerable Reasoning Calibration of Large Language Models for Free",
      url: "https://arxiv.org/abs/2504.07986",
    },
    model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    prompt:
      "Please reason step by step, and put your final answer within \\boxed{}.\nHow many prime numbers are between 20 and 30?",
    description: {
      en: "SEAL trims redundant chain-of-thought without retraining. CoT traces are segmented at paragraph breaks and keyword-classified as execution, reflection, or transition; hidden states captured only at the break tokens are averaged per category into three vectors. Steering adds the execution vector (+0.5) and subtracts reflection and transition (-0.5 each) at layer 20 — applied only on paragraph-break tokens during generation — shortening reasoning on MATH-500 while keeping answers.",
      zh: "SEAL 不用重新训练就能裁掉冗余的思维链。先按段落分隔符切分 CoT 轨迹，用关键词把每段归为执行、反思、转换三类；只在分隔 token 处捕获隐藏状态，按类别求平均得到三个向量。引导时在第 20 层加上执行向量（+0.5）、减去反思与转换向量（各 -0.5），且只作用于生成阶段的段落分隔 token——在 MATH-500 上推理长度变短，答案仍然正确。",
    },
    note: {
      en: "The notebook triggers on every vocab token ending in '\\n\\n' (string form). This preset uses the primary '\\n\\n' token id 271; extend tokens for full coverage.",
      zh: "notebook 里会在所有以 '\\n\\n' 结尾的词表 token 上触发；这份预设只用了主 '\\n\\n' 的 token id 271，要完整覆盖请自行扩展 tokens 列表。",
    },
    spec: {
      conflict: "sequential",
      vectors: [
        {
          source: "execution_avg_vector.gguf",
          scale: 0.5,
          layers: [20],
          apply: { generation_tokens: [271] },
        },
        {
          source: "reflection_avg_vector.gguf",
          scale: -0.5,
          layers: [20],
          apply: { generation_tokens: [271] },
        },
        {
          source: "transition_avg_vector.gguf",
          scale: -0.5,
          layers: [20],
          apply: { generation_tokens: [271] },
        },
      ],
    },
  },
  {
    id: "sharp",
    category: "knowledge",
    year: 2025,
    method: "SHARP",
    tagline: {
      en: "A task vector suppresses leading-question hallucinations in LVLMs.",
      zh: "任务向量抑制视觉语言模型的诱导性幻觉。",
    },
    paper: {
      title: "SHARP: Steering Hallucination in LVLMs via Representation Engineering",
      url: "https://aclanthology.org/2025.emnlp-main.725/",
    },
    model: "llava-hf/llava-v1.6-vicuna-7b-hf",
    prompt: "Is there a parking meter in the image?",
    description: {
      en: "SHARP is a representation-level intervention for vision-language models that suppresses hallucinations driven by textual priors and leading questions. The demo applies the paper's released layer-10 task vector at scale 8 at the last prompt position and on every generated token, on one of the author's POPE examples: the baseline hallucinates a parking meter that is not in the picture, and the steered model answers from the visual evidence.",
      zh: "SHARP 是面向视觉-语言模型的表示层面干预，用来抑制由文本先验和诱导性提问引出的幻觉。演示以 8 的缩放系数把论文放出的第 10 层任务向量施加在 prompt 末尾位置和每个生成 token 上，用的是作者提供的 POPE 示例之一：基线凭空说图里有停车计时器，引导后的模型则依据画面证据作答。",
    },
    note: {
      en: "Multimodal demo (image input); the Steer page sends text only. Vector is an in-memory payload: vec.from_pt_direction('task_vector_layer-10.pt', layers=[10]).",
      zh: "多模态演示（输入含图像）；“引导”页只发送文本。向量为内联 data 负载：vec.from_pt_direction('task_vector_layer-10.pt', layers=[10])。",
    },
    spec: {
      vectors: [
        {
          data: inlinePayload("vec.from_pt_direction('task_vector_layer-10.pt', layers=[10])"),
          scale: 8,
          layers: [10],
          apply: { prompt_positions: [-1], generation: "all" },
        },
      ],
    },
  },
  {
    id: "steerable_chatbot",
    category: "persona",
    year: 2025,
    method: "Steerable Chatbot",
    tagline: {
      en: "A preference axis shifts replies between adult- and kid-oriented.",
      zh: "偏好轴在成人向与亲子向回复间切换。",
    },
    paper: {
      title:
        "Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering",
      url: "https://arxiv.org/abs/2505.04260",
    },
    model: "Qwen/Qwen2.5-1.5B-Instruct",
    prompt: "What are some activities our family can do in the city this weekend?",
    description: {
      en: "Personalizes a chatbot along a user-preference style axis. Last-token hidden states over adult-oriented vs child-oriented activity recommendations fit a linear-probe steering vector (style-probe.gguf), applied across all 28 layers: scale +1.5 shifts replies toward adults-only suggestions, -1.0 toward kid-friendly ones, both against the same unsteered baseline.",
      zh: "沿用户偏好的风格轴给聊天机器人做个性化。用面向成人与面向儿童的活动推荐句在末尾 token 上的隐藏状态拟合线性探针引导向量（style-probe.gguf），作用于全部 28 层：缩放系数取 +1.5 让回复偏向成人向建议，取 -1.0 偏向亲子向建议，两者都与同一条未引导的基线对比。",
    },
    note: {
      en: "Preset uses the adult direction (+1.5); set scale to -1.0 for the kid-friendly direction.",
      zh: "预设用的是成人方向（+1.5）；把缩放系数改成 -1.0 就得到亲子方向。",
    },
    spec: {
      vectors: [
        {
          source: "style-probe.gguf",
          scale: 1.5,
          layers: range(0, 28),
          normalize: true,
          apply: { "prompt": "all", "generation": "all" },
        },
      ],
    },
  },
  {
    id: "steermoe",
    category: "experts",
    year: 2025,
    method: "SteerMoE",
    tagline: {
      en: "Deactivating 200 experts flips counting from words to digits.",
      zh: "停用 200 个专家，计数从单词变成数字。",
    },
    paper: {
      title: "Steering MoE LLMs via Expert (De)Activation",
      url: "https://arxiv.org/abs/2509.09660",
    },
    model: "Qwen/Qwen3-30B-A3B",
    prompt: "Count to fifteen.",
    description: {
      en: "Steers a Mixture-of-Experts model by deactivating behavior-linked experts in the router instead of adding activation vectors. Router logits captured over contrastive digit-vs-word counting chats rank each expert by selection-rate difference; the 200 most word-linked experts (46 layers) are written to a deactivation config JSON. Steering with it flips greedy counting from spelled-out words to digits, and a re-capture confirms zero deactivated-expert selections.",
      zh: "通过在路由器里停用与特定行为相关的专家来引导 MoE 模型，而不是叠加激活向量。在“数字 vs 英文单词”计数的对比对话上捕获路由 logits，按选中率差异给专家排序；把最偏向单词的 200 个专家（跨 46 层）写进一份停用配置 JSON。带上它之后，贪心解码的计数从英文单词翻转成数字，再捕获一次可以确认这些专家的选中次数为零。",
    },
    note: {
      en: "moe_router spec: per-layer mode/expert_ids come from the JSON config file; layers are read from the file as well.",
      zh: "moe_router Spec：每层的 mode/expert_ids 来自 JSON 配置文件，层列表也由文件决定。",
    },
    spec: {
      vectors: [
        {
          source: "steermoe_qwen3_words.json",
          algorithm: "moe_router",
          apply: { "prompt": "all", "generation": "all" },
        },
      ],
    },
  },
];

export function getGalleryEntry(id: string): GalleryEntry | undefined {
  return galleryEntries.find((e) => e.id === id);
}
