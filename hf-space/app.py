"""
EasySteer Hugging Face Space Demo
A simplified demo showcasing LLM steering with steering vectors

Supports two modes controlled by the **DEMO_MODE** env-var:
  "api"  – lightweight, CPU-only, forwards requests to a remote vLLM server
  "gpu"  – loads the model locally with vLLM (requires NVIDIA GPU)
  "auto" – (default) if VLLM_API_URL is set → api; otherwise → gpu

Required env-vars for API mode (set them in HF Space → Settings → Secrets):
  VLLM_API_URL          – base URL of the remote vLLM server (e.g. http://host:port/v1)
  VLLM_API_KEY          – API key for authentication
  VLLM_MODEL_NAME       – model name served by the remote server
  VLLM_VECTOR_BASE_PATH – (optional) absolute prefix prepended to relative vector
                           paths so they resolve correctly on the remote server
"""

import gradio as gr
import os
import json
import time
from typing import Tuple, Dict, Any, List, Optional

# ===== Mode Detection =====
# Priority:
#   1. DEMO_MODE env-var: "api" | "gpu"  (explicit override)
#   2. "auto" (default): VLLM_API_URL set → api; otherwise → gpu
_demo_mode = os.environ.get("DEMO_MODE", "auto").strip().lower()
if _demo_mode == "api":
    USE_API = True
elif _demo_mode == "gpu":
    USE_API = False
else:  # auto
    USE_API = os.environ.get("VLLM_API_URL") is not None

if USE_API:
    from openai import OpenAI
    _api_client = OpenAI(
        base_url=os.environ["VLLM_API_URL"],
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
    )
    API_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "default")
    VECTOR_BASE_PATH = os.environ.get("VLLM_VECTOR_BASE_PATH", "")
    print(f"🌐 API mode enabled (DEMO_MODE={_demo_mode})")
else:
    from vllm import LLM, SamplingParams
    from vllm.steer_vectors import SteeringSpec
    print(f"🖥️  GPU mode (DEMO_MODE={_demo_mode})")

# ===== Configuration =====
MODEL_NAME = "/app/models/Qwen2.5-1.5B-Instruct"
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

# Global model instance (loaded once)
llm_instance = None

# Global ID counter (same pattern as frontend/core/id_generator.py)
_global_id_counter = 1

# ===== Config Loading =====
def load_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load all config files from configs/ directory."""
    single_configs = {}
    multi_configs = {}
    
    # Load single vector configs
    sv_dir = os.path.join(CONFIGS_DIR, "inference")
    if os.path.exists(sv_dir):
        for f in sorted(os.listdir(sv_dir)):
            if f.endswith(".json"):
                with open(os.path.join(sv_dir, f)) as fh:
                    single_configs[f[:-5]] = json.load(fh)
    
    # Load multi vector configs
    mv_dir = os.path.join(CONFIGS_DIR, "multi_vector")
    if os.path.exists(mv_dir):
        for f in sorted(os.listdir(mv_dir)):
            if f.endswith(".json"):
                with open(os.path.join(mv_dir, f)) as fh:
                    multi_configs[f[:-5]] = json.load(fh)
    
    return single_configs, multi_configs


def parse_int_list(s: str) -> List[int]:
    """Parse comma-separated string to list of ints."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def display_val(val, default="None"):
    """Return 'None' if the value is empty or missing, otherwise return as-is."""
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default
    return val


# Load configs at module level
SINGLE_CONFIGS, MULTI_CONFIGS = load_configs()

# ===== Config Descriptions (shown to user on selection) =====
SINGLE_CONFIG_DESCRIPTIONS: Dict[str, str] = {
    "emotion_direct": "Steers the model to respond in a happier, more positive tone — even in contexts where sadness would be expected.",
    "emoji_loreft": (
        "Steers the model to include emojis in output. "
        "Note: this is for testing only — the LoReFT vector was trained on very few "
        "examples (~dozens), so it works reliably only on certain prompts."
    ),
    "adult_style": "Steers output to be more aligned with adult interests and preferences.",
    "refuse_control": "Makes the model tend to refuse answering, even for normal and harmless requests.",
}
MULTI_CONFIG_DESCRIPTIONS: Dict[str, str] = {
    "refusal_direction": (
        "Makes the model tend to refuse answering even normal requests. "
        "Achieved by applying a different steering vector at each of the "
        "last 4 tokens of the prompt."
    ),
}

# Configs where the scale slider should NOT be user-adjustable
_SCALE_LOCKED_SINGLE = {"emoji_loreft"}


def _get_sv_description(config_name: str) -> str:
    return SINGLE_CONFIG_DESCRIPTIONS.get(config_name, "")


def _get_mv_description(config_name: str) -> str:
    return MULTI_CONFIG_DESCRIPTIONS.get(config_name, "")


# ===== Model Loading =====
def load_model():
    """Load the LLM model with steering support."""
    global llm_instance
    if llm_instance is None:
        print("🔄 Loading model...")
        llm_instance = LLM(
            model=MODEL_NAME,
            enable_steer_vector=True,
            enforce_eager=True,
            enable_chunked_prefill=False,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            tensor_parallel_size=1
        )
        print("✅ Model loaded successfully!")
    return llm_instance


def format_prompt(instruction: str) -> str:
    """Format instruction with Qwen2.5 chat template."""
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

# ===== Steering Spec Builders (v2 wire format, both modes) =====
def _resolve_path(relative_path: str) -> str:
    """In API mode, convert a relative vector path to an absolute server path."""
    if USE_API and VECTOR_BASE_PATH:
        return os.path.join(VECTOR_BASE_PATH, relative_path)
    return relative_path


# Data-only algorithms: the engine no longer loads their files by path;
# the easysteer payload adapters interpret them client-side.
_DATA_ONLY_ALGORITHMS = ("linear", "lm_steer", "loreft")


def _apply_clauses(prefill_tokens, prefill_positions, generate_tokens):
    """Translate the legacy config trigger fields into v2 apply clauses.

    Same mapping as frontend/core/steer_request_builder.py: -1 in a
    token list means "every token of that phase"; no fields at all
    means both phases unfiltered.
    """
    if prefill_tokens is None and prefill_positions is None and generate_tokens is None:
        return [{"phases": ["prompt", "generation"]}]
    want_prompt = prefill_tokens is not None or prefill_positions is not None
    if prefill_tokens is not None and -1 in prefill_tokens:
        prefill_tokens = None
        prefill_positions = None
    want_generation = generate_tokens is not None
    if generate_tokens is not None and -1 in generate_tokens:
        generate_tokens = None
    if (want_prompt and want_generation and prefill_positions is None
            and prefill_tokens == generate_tokens):
        clause = {"phases": ["prompt", "generation"]}
        if prefill_tokens:
            clause["tokens"] = prefill_tokens
        return [clause]
    clauses = []
    if want_prompt:
        clause = {"phases": ["prompt"]}
        if prefill_tokens:
            clause["tokens"] = prefill_tokens
        if prefill_positions:
            clause["positions"] = prefill_positions
        clauses.append(clause)
    if want_generation:
        clause = {"phases": ["generation"]}
        if generate_tokens:
            clause["tokens"] = generate_tokens
        clauses.append(clause)
    return clauses


def _b64_payload_wire(payload) -> dict:
    """Payload wire with tensor bytes base64-encoded for JSON transport
    (the engine's wire decoder is base64-tolerant)."""
    import base64

    wire = payload.to_wire()
    wire["tensors"] = {
        name: {"shape": t["shape"],
               "data": base64.b64encode(t["data"]).decode("ascii")}
        for name, t in wire["tensors"].items()
    }
    return wire


def _vector_source(algorithm: str, path: str) -> dict:
    """source= for engine formats; data= payload for data-only algorithms."""
    if algorithm not in _DATA_ONLY_ALGORITHMS:
        return {"source": _resolve_path(path)}
    try:
        import easysteer.vectors as vec
    except ImportError as e:
        raise RuntimeError(
            f"algorithm {algorithm!r} needs the easysteer payload adapters "
            "(server-side loading of these files by path was removed); "
            "run this demo config in GPU mode or install easysteer"
        ) from e
    adapter = {"linear": vec.from_linear_transport,
               "lm_steer": vec.from_lm_steer,
               "loreft": vec.from_pyreft}[algorithm]
    payload = adapter(path)
    return {"data": _b64_payload_wire(payload) if USE_API else payload}


def _vector_wires(vc, scale_override=None):
    """One legacy config entry -> v2 vector wire dicts (one per clause)."""
    algorithm = vc.get("algorithm", "direct")
    scale = scale_override if scale_override is not None else float(vc.get("scale", 1.0))
    base = {
        **_vector_source(algorithm, vc["path"]),
        "scale": scale,
        "algorithm": algorithm,
        "normalize": bool(vc.get("normalize", False)),
    }
    if vc.get("target_layers"):
        base["layers"] = parse_int_list(str(vc["target_layers"]))

    def _field(name):
        return parse_int_list(str(vc[name])) if vc.get(name) else None

    return [
        {**base, "apply": clause}
        for clause in _apply_clauses(
            _field("prefill_trigger_tokens"),
            _field("prefill_trigger_positions"),
            _field("generate_trigger_tokens"),
        )
    ]


def build_single_spec_wire(config, scale_override=None) -> dict:
    """v2 SteeringSpec wire for a single-vector config."""
    return {"vectors": _vector_wires(config["steer_vector"],
                                     scale_override=scale_override)}


def build_multi_spec_wire(config) -> dict:
    """v2 SteeringSpec wire for a multi-vector config."""
    vectors = []
    for vc in config["vector_configs"]:
        vectors.extend(_vector_wires(vc))
    return {
        "vectors": vectors,
        "conflict": config["steer_vector"].get("conflict_resolution", "sequential"),
    }


def _local_spec(wire):
    """Materialize a wire dict into a SteeringSpec for llm.generate."""
    return SteeringSpec.model_validate(wire)


def _api_generate(prompt: str, config, spec_wire) -> str:
    """Call the remote vLLM server; spec_wire=None means no steering."""
    sampling = config["sampling"]
    extra_body = {"repetition_penalty": float(sampling.get("repetition_penalty", 1.1))}
    if spec_wire is not None:
        extra_body["steering"] = spec_wire
    response = _api_client.chat.completions.create(
        model=API_MODEL_NAME,
        messages=[{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
        max_tokens=int(sampling.get("max_tokens", 128)),
        temperature=float(sampling.get("temperature", 0.0)),
        extra_body=extra_body,
    )
    return response.choices[0].message.content


# ===== Generation Functions =====
def generate_single(config_name: str, prompt: str, scale: float, progress=gr.Progress()) -> Tuple[str, str]:
    """Generate text using single steering vector."""
    try:
        config = SINGLE_CONFIGS[config_name]
        # For scale-locked configs, ignore user slider and use config value
        if config_name in _SCALE_LOCKED_SINGLE:
            scale = float(config["steer_vector"].get("scale", 1.0))

        if USE_API:
            # ---- API mode ----
            progress(0.2, desc="Calling API (baseline)...")
            baseline_text = _api_generate(prompt, config, None)

            progress(0.6, desc="Calling API (steered)...")
            steered_text = _api_generate(prompt, config, build_single_spec_wire(config, scale_override=scale))
        else:
            # ---- Local mode ----
            progress(0, desc="Loading model...")
            llm = load_model()

            formatted_prompt = format_prompt(prompt)
            sampling_params = SamplingParams(
                temperature=float(config["sampling"].get("temperature", 0.0)),
                max_tokens=int(config["sampling"].get("max_tokens", 128)),
                repetition_penalty=float(config["sampling"].get("repetition_penalty", 1.1))
            )

            progress(0.3, desc="Generating baseline...")
            baseline_out = llm.generate(formatted_prompt, sampling_params=sampling_params)
            baseline_text = baseline_out[0].outputs[0].text

            progress(0.6, desc="Generating steered output...")
            steered_out = llm.generate(formatted_prompt, steering=_local_spec(build_single_spec_wire(config, scale_override=scale)), sampling_params=sampling_params)
            steered_text = steered_out[0].outputs[0].text

        progress(1.0, desc="Complete!")
        return baseline_text, steered_text
    except Exception as e:
        import traceback
        err = f"❌ Error: {e}\n\n{traceback.format_exc()}"
        return err, err


def generate_multi(config_name: str, prompt: str, progress=gr.Progress()) -> Tuple[str, str]:
    """Generate text using multiple steering vectors."""
    try:
        config = MULTI_CONFIGS[config_name]

        if USE_API:
            # ---- API mode ----
            progress(0.2, desc="Calling API (baseline)...")
            baseline_text = _api_generate(prompt, config, None)

            progress(0.6, desc="Calling API (multi-vector steered)...")
            steered_text = _api_generate(prompt, config, build_multi_spec_wire(config))
        else:
            # ---- Local mode ----
            progress(0, desc="Loading model...")
            llm = load_model()

            formatted_prompt = format_prompt(prompt)
            sampling_params = SamplingParams(
                temperature=float(config["sampling"].get("temperature", 0.0)),
                max_tokens=int(config["sampling"].get("max_tokens", 128)),
                repetition_penalty=float(config["sampling"].get("repetition_penalty", 1.1))
            )

            progress(0.3, desc="Generating baseline...")
            baseline_out = llm.generate(formatted_prompt, sampling_params=sampling_params)
            baseline_text = baseline_out[0].outputs[0].text

            progress(0.6, desc="Generating multi-vector steered output...")
            steered_out = llm.generate(formatted_prompt, steering=_local_spec(build_multi_spec_wire(config)), sampling_params=sampling_params)
            steered_text = steered_out[0].outputs[0].text

        progress(1.0, desc="Complete!")
        return baseline_text, steered_text
    except Exception as e:
        import traceback
        err = f"❌ Error: {e}\n\n{traceback.format_exc()}"
        return err, err


# ===== UI Helper Functions =====
def update_sv_ui(config_name):
    """Update all single-vector UI fields when config changes."""
    config = SINGLE_CONFIGS[config_name]
    sv = config["steer_vector"]
    sampling = config["sampling"]
    scale_val = float(sv.get("scale", 1.0))
    is_locked = config_name in _SCALE_LOCKED_SINGLE
    return (
        _get_sv_description(config_name),
        display_val(sampling.get("temperature"), "0.0"),
        display_val(sampling.get("max_tokens"), "128"),
        display_val(sampling.get("repetition_penalty"), "1.1"),
        display_val(sv.get("path")),
        display_val(sv.get("algorithm"), "direct"),
        display_val(sv.get("target_layers")),
        display_val(sv.get("prefill_trigger_tokens")),
        display_val(sv.get("prefill_trigger_positions")),
        display_val(sv.get("generate_trigger_tokens")),
        display_val(str(sv.get("normalize", False))),
        gr.update(value=scale_val, interactive=not is_locked),
        display_val(config["model"].get("instruction")),
    )


# Max number of vector tabs to pre-create (based on all multi-vector configs)
MAX_VECTORS = max((len(c["vector_configs"]) for c in MULTI_CONFIGS.values()), default=4)
FIELDS_PER_VECTOR = 8  # path, algorithm, target_layers, prefill_tokens, prefill_positions, generate_tokens, normalize, scale


def update_mv_ui(config_name):
    """Update all multi-vector UI fields when config changes.
    Returns: (description, temperature, max_tokens, rep_penalty, sv_name, conflict_resolution,
              [path, algo, layers, pf_tokens, pf_positions, gen_tokens, normalize, scale] * MAX_VECTORS,
              instruction)
    """
    config = MULTI_CONFIGS[config_name]
    sv = config["steer_vector"]
    sampling = config["sampling"]
    vecs = config["vector_configs"]

    results = [
        _get_mv_description(config_name),
        display_val(sampling.get("temperature"), "0.0"),
        display_val(sampling.get("max_tokens"), "128"),
        display_val(sampling.get("repetition_penalty"), "1.1"),
        display_val(sv.get("name")),
        display_val(sv.get("conflict_resolution"), "sequential"),
    ]

    for i in range(MAX_VECTORS):
        if i < len(vecs):
            v = vecs[i]
            results.extend([
                display_val(v.get("path")),
                display_val(v.get("algorithm"), "direct"),
                display_val(v.get("target_layers")),
                display_val(v.get("prefill_trigger_tokens")),
                display_val(v.get("prefill_trigger_positions")),
                display_val(v.get("generate_trigger_tokens")),
                display_val(str(v.get("normalize", False))),
                float(v.get("scale", 1.0)),
            ])
        else:
            results.extend(["None", "None", "None", "None", "None", "None", "None", 0.0])

    results.append(display_val(config["model"]["instruction"]))
    return tuple(results)


# ===== Build Gradio Interface =====
CUSTOM_CSS = """
/* Stronger borders on actual input elements only (tag selectors, not class) */
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container textarea,
.gradio-container select {
    border: 1.5px solid #c0c5ce !important;
}
/* Tighten badge spacing */
.badge-row {
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.badge-row a {
    display: inline-flex;
    margin: 0 !important;
    padding: 0 !important;
}
.badge-row img {
    display: block;
    margin: 0 !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), title="EasySteer Demo", css=CUSTOM_CSS) as demo:
    gr.HTML("""
    <div style="text-align: center;">
        <h2 style="white-space: nowrap; margin-bottom: 8px;">🚗 EasySteer: A Unified Framework for High-Performance LLM Steering</h2>
        <div class="badge-row">
            <a href="https://github.com/ZJU-REAL/EasySteer"><img src="https://img.shields.io/github/stars/ZJU-REAL/EasySteer?style=social" alt="GitHub"></a>
            <a href="https://arxiv.org/abs/2509.25175"><img src="https://img.shields.io/badge/arXiv-2509.25175-b31b1b.svg" alt="Paper"></a>
            <a href="https://github.com/ZJU-REAL/EasySteer/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ZJU-REAL/EasySteer" alt="License"></a>
            <a href="https://hub.docker.com/r/xuhaolei/easysteer/tags"><img src="https://img.shields.io/badge/docker-v0.13.0-orange" alt="Docker"></a>
        </div>
        <p style="color: #666; font-size: 0.9em; margin-top: 8px; max-width: 720px; margin-left: auto; margin-right: auto;">
            This online demo is for quickly testing the framework and verifying steering vector effectiveness.
            Inference results are provided by a server deployed by the ZJU REAL Lab.
            For full features (vector extraction, training, SAE, chat, etc.), please refer to the
            <a href="https://github.com/ZJU-REAL/EasySteer?tab=readme-ov-file#frontend" target="_blank">frontend deployment guide</a>
            in the GitHub repo.
        </p>
    </div>
    """)
    
    first_sv_key = "emotion_direct" if "emotion_direct" in SINGLE_CONFIGS else list(SINGLE_CONFIGS.keys())[0]
    first_sv = SINGLE_CONFIGS[first_sv_key]
    first_mv_key = list(MULTI_CONFIGS.keys())[0]
    
    with gr.Tabs():
        # ===== Single Vector Tab =====
        with gr.Tab("🎯 Single Vector"):
            # --- Import Config (placed first so description appears early) ---
            sv_config_dropdown = gr.Dropdown(
                choices=list(SINGLE_CONFIGS.keys()),
                value=first_sv_key,
                label="Import Configuration",
                info="Select a predefined steering configuration"
            )
            sv_description = gr.Markdown(value=_get_sv_description(first_sv_key))

            # --- Card 1: Sampling Configuration ---
            gr.Markdown("### 🤖 Sampling Configuration")
            with gr.Row():
                sv_temperature = gr.Textbox(label="Temperature", info="0 = deterministic, higher = more random", placeholder="e.g. 0.0", value=display_val(first_sv["sampling"].get("temperature"), "0.0"), interactive=False)
                sv_max_tokens = gr.Textbox(label="Max Tokens", info="Maximum number of tokens to generate", placeholder="e.g. 128", value=display_val(first_sv["sampling"].get("max_tokens"), "128"), interactive=False)
                sv_rep_penalty = gr.Textbox(label="Repetition Penalty", info="Penalize repeated tokens", placeholder="e.g. 1.1", value=display_val(first_sv["sampling"].get("repetition_penalty"), "1.1"), interactive=False)
            
            # --- Card 2: Steer Vector Configuration ---
            gr.Markdown("### ⚙️ Steer Vector Configuration")
            with gr.Row():
                sv_path = gr.Textbox(label="Vector Path", info="Path to the steering vector file", value=display_val(first_sv["steer_vector"].get("path")), interactive=False)
                sv_algorithm = gr.Textbox(label="Algorithm", info="Steering algorithm used for this vector", placeholder="e.g. direct", value=display_val(first_sv["steer_vector"].get("algorithm"), "direct"), interactive=False)
                sv_target_layers = gr.Textbox(label="Target Layers", info="Layer indices, comma-separated", placeholder="e.g. 10,11,12,...,23", value=display_val(first_sv["steer_vector"].get("target_layers")), interactive=False)
            with gr.Row():
                sv_prefill_tokens = gr.Textbox(label="Prefill Trigger Token IDs", info="-1 = apply to all tokens", placeholder="e.g. -1", value=display_val(first_sv["steer_vector"].get("prefill_trigger_tokens")), interactive=False)
                sv_prefill_positions = gr.Textbox(label="Prefill Trigger Positions", info="Supports negative indexing", placeholder="e.g. -1", value=display_val(first_sv["steer_vector"].get("prefill_trigger_positions")), interactive=False)
                sv_generate_tokens = gr.Textbox(label="Generate Trigger Token IDs", info="-1 = apply to all tokens", placeholder="e.g. -1", value=display_val(first_sv["steer_vector"].get("generate_trigger_tokens")), interactive=False)
            with gr.Row():
                sv_normalize = gr.Textbox(label="Normalize", info="Whether to normalize the vector", value=display_val(str(first_sv["steer_vector"].get("normalize", False))), interactive=False)
                sv_scale = gr.Slider(
                    label="Scale Factor", info="Steering strength multiplier (drag to adjust)",
                    minimum=-3, maximum=3, step=0.1,
                    value=float(first_sv["steer_vector"].get("scale", 1.0)),
                    interactive=(first_sv_key not in _SCALE_LOCKED_SINGLE),
                )
            
            # --- Instruction + Generate ---
            sv_prompt_input = gr.Textbox(
                label="Input Instruction",
                lines=3,
                value=first_sv["model"]["instruction"]
            )
            sv_generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
            
            # --- Results ---
            gr.Markdown("### 📊 Results Comparison")
            with gr.Row():
                sv_baseline_output = gr.Textbox(label="🔹 Baseline (No Steering)", lines=8, interactive=False)
                sv_steered_output = gr.Textbox(label="🔸 Steered Output", lines=8, interactive=False)
            
            # Wire up events
            sv_config_dropdown.change(
                fn=update_sv_ui,
                inputs=[sv_config_dropdown],
                outputs=[
                    sv_description,
                    sv_temperature, sv_max_tokens, sv_rep_penalty,
                    sv_path, sv_algorithm, sv_target_layers,
                    sv_prefill_tokens, sv_prefill_positions, sv_generate_tokens,
                    sv_normalize, sv_scale,
                    sv_prompt_input
                ]
            )
            sv_generate_btn.click(
                fn=generate_single,
                inputs=[sv_config_dropdown, sv_prompt_input, sv_scale],
                outputs=[sv_baseline_output, sv_steered_output]
            )
        
        # ===== Multi-Vector Tab =====
        with gr.Tab("🎨 Multi-Vector"):
            first_mv = MULTI_CONFIGS[first_mv_key]
            first_mv_sv = first_mv["steer_vector"]
            first_mv_vecs = first_mv["vector_configs"]

            # --- Import Config (placed first so description appears early) ---
            mv_config_dropdown = gr.Dropdown(
                choices=list(MULTI_CONFIGS.keys()),
                value=first_mv_key,
                label="Import Configuration",
                info="Select a predefined multi-vector configuration"
            )
            mv_description = gr.Markdown(value=_get_mv_description(first_mv_key))

            # --- Sampling Configuration ---
            gr.Markdown("### 🤖 Sampling Configuration")
            with gr.Row():
                mv_temperature = gr.Textbox(label="Temperature", info="0 = deterministic, higher = more random", value=display_val(first_mv["sampling"].get("temperature"), "0.0"), interactive=False)
                mv_max_tokens = gr.Textbox(label="Max Tokens", info="Maximum number of tokens to generate", value=display_val(first_mv["sampling"].get("max_tokens"), "128"), interactive=False)
                mv_rep_penalty = gr.Textbox(label="Repetition Penalty", info="Penalize repeated tokens", value=display_val(first_mv["sampling"].get("repetition_penalty"), "1.1"), interactive=False)

            # --- Steer Vector Configuration (top-level) ---
            gr.Markdown("### ⚙️ Steer Vector Configuration")
            with gr.Row():
                mv_sv_name = gr.Textbox(label="Steer Vector Name", info="Identifier name for this steering vector group", value=display_val(first_mv_sv.get("name")), interactive=False)
                mv_conflict_resolution = gr.Textbox(label="Conflict Resolution", info="How to combine multiple vectors", value=display_val(first_mv_sv.get("conflict_resolution"), "sequential"), interactive=False)

            # --- Per-Vector Configurations (sub-tabs) ---
            gr.Markdown("### 🎯 Vector Configurations")
            mv_vec_fields = []  # flat list per vector: [path, algo, layers, pf_tokens, pf_positions, gen_tokens, normalize, scale]
            with gr.Tabs():
                for vi in range(MAX_VECTORS):
                    v_data = first_mv_vecs[vi] if vi < len(first_mv_vecs) else {}
                    with gr.Tab(f"Vector {vi + 1}"):
                        with gr.Row():
                            f_path = gr.Textbox(label="Vector Path", info="Path to the steering vector file", value=display_val(v_data.get("path")), interactive=False)
                            f_algo = gr.Textbox(label="Algorithm", info="Steering algorithm used for this vector", value=display_val(v_data.get("algorithm"), "direct"), interactive=False)
                            f_layers = gr.Textbox(label="Target Layers", info="Layer indices, comma-separated", value=display_val(v_data.get("target_layers")), interactive=False)
                        with gr.Row():
                            f_pf_tokens = gr.Textbox(label="Prefill Trigger Token IDs", info="-1 = apply to all tokens", value=display_val(v_data.get("prefill_trigger_tokens")), interactive=False)
                            f_pf_positions = gr.Textbox(label="Prefill Trigger Positions", info="Supports negative indexing", value=display_val(v_data.get("prefill_trigger_positions")), interactive=False)
                            f_gen_tokens = gr.Textbox(label="Generate Trigger Token IDs", info="-1 = apply to all tokens", value=display_val(v_data.get("generate_trigger_tokens")), interactive=False)
                        with gr.Row():
                            f_normalize = gr.Textbox(label="Normalize", info="Whether to normalize the vector", value=display_val(str(v_data.get("normalize", False))), interactive=False)
                            f_scale = gr.Slider(label="Scale Factor", info="Steering strength multiplier", minimum=-3, maximum=3, step=0.1, value=float(v_data.get("scale", 1.0)), interactive=False)
                        mv_vec_fields.extend([f_path, f_algo, f_layers, f_pf_tokens, f_pf_positions, f_gen_tokens, f_normalize, f_scale])

            # --- Instruction + Generate ---
            mv_prompt_input = gr.Textbox(
                label="Input Instruction",
                lines=3,
                value=first_mv["model"]["instruction"]
            )
            mv_generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

            # --- Results ---
            gr.Markdown("### 📊 Results Comparison")
            with gr.Row():
                mv_baseline_output = gr.Textbox(label="🔹 Baseline (No Steering)", lines=8, interactive=False)
                mv_steered_output = gr.Textbox(label="🎨 Steered Output (Multi-Vector)", lines=8, interactive=False)

            # --- Wire up events ---
            mv_config_dropdown.change(
                fn=update_mv_ui,
                inputs=[mv_config_dropdown],
                outputs=[
                    mv_description,
                    mv_temperature, mv_max_tokens, mv_rep_penalty,
                    mv_sv_name, mv_conflict_resolution,
                    *mv_vec_fields,
                    mv_prompt_input
                ]
            )
            mv_generate_btn.click(fn=generate_multi, inputs=[mv_config_dropdown, mv_prompt_input], outputs=[mv_baseline_output, mv_steered_output])
    
    gr.Markdown("---\n*Powered by [EasySteer](https://github.com/ZJU-REAL/EasySteer)*")

# ===== Launch =====
if __name__ == "__main__":
    print("🚀 Starting EasySteer Demo...")
    print(f"📁 Configs: {len(SINGLE_CONFIGS)} single, {len(MULTI_CONFIGS)} multi")
    for name in SINGLE_CONFIGS:
        print(f"   Single: {name}")
    for name in MULTI_CONFIGS:
        print(f"   Multi: {name}")

    if USE_API:
        print(f"\n🌐 Running in API mode (DEMO_MODE={_demo_mode})")
        print(f"   Model: {API_MODEL_NAME}")
    else:
        print(f"\n🖥️  Running in GPU mode (DEMO_MODE={_demo_mode})")
        print("📦 Pre-loading model...")
        try:
            load_model()
            print(f"✅ Model loaded: {MODEL_NAME}")
        except Exception as e:
            print(f"⚠️  Model pre-loading failed: {e}")

    print("\n🌐 Launching Gradio interface...")
    demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=7860, share=False)
