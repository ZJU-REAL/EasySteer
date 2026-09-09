"""Gradio steering demo with remote API and local GPU modes.

DEMO_MODE defaults to api, which requires VLLM_API_URL and VLLM_MODEL_NAME.
Optional API settings are VLLM_API_KEY and VLLM_VECTOR_BASE_PATH. Set
DEMO_MODE=gpu to load the model locally with vLLM.
"""

import json
import logging
import os
from typing import Any, Dict, Tuple

from runtime import ALGORITHM_CAPABILITIES, demo_mode, load_payload
from steering_config import build_multi_spec_wire, build_single_spec_wire

logger = logging.getLogger(__name__)

# Validate configuration before importing UI or local GPU dependencies.
_demo_mode = demo_mode()
USE_API = _demo_mode == "api"

import gradio as gr  # noqa: E402

if USE_API:
    from openai import OpenAI

    _api_client = OpenAI(
        base_url=os.environ["VLLM_API_URL"],
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
    )
    API_MODEL_NAME = os.environ["VLLM_MODEL_NAME"]
    VECTOR_BASE_PATH = os.environ.get("VLLM_VECTOR_BASE_PATH", "")
    print(f"🌐 API mode enabled (DEMO_MODE={_demo_mode})")
else:
    from vllm import LLM, SamplingParams
    from vllm.steer_vectors import SteeringSpec

    print(f"🖥️  GPU mode (DEMO_MODE={_demo_mode})")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.environ.get("EASYSTEER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
CONFIGS_DIR = os.path.join(APP_DIR, "configs")

llm_instance = None


def _load_config_dir(subdir: str) -> Dict[str, Any]:
    """Load JSON presets from one directory in filename order."""
    directory = os.path.join(CONFIGS_DIR, subdir)
    configs = {}
    if os.path.exists(directory):
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".json"):
                with open(os.path.join(directory, filename)) as file:
                    configs[filename[:-5]] = json.load(file)
    return configs


def load_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load single-vector and multi-vector presets."""
    return _load_config_dir("inference"), _load_config_dir("multi_vector")


def display_val(val, default="None"):
    """Substitute the display default for None or an empty string."""
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default
    return val


SINGLE_CONFIGS, MULTI_CONFIGS = load_configs()

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


def load_model():
    """Load the LLM model with steering support."""
    global llm_instance
    if llm_instance is None:
        print("🔄 Loading model...")
        llm_instance = LLM(
            model=MODEL_NAME,
            enable_steer_vector=True,
            steer_algorithms="all",  # the demo serves user-picked algorithms
            steer_multi_vector=True,
            enforce_eager=True,
            enable_chunked_prefill=False,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            tensor_parallel_size=1,
        )
        print("✅ Model loaded successfully!")
    return llm_instance


def format_prompt(instruction: str) -> str:
    """Format instruction with Qwen2.5 chat template."""
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"


def _resolve_path(relative_path: str) -> str:
    """In API mode, convert a relative vector path to an absolute server path."""
    if USE_API and VECTOR_BASE_PATH:
        return os.path.join(VECTOR_BASE_PATH, relative_path)
    return relative_path if USE_API else os.path.join(APP_DIR, relative_path)


def _vector_source(algorithm: str, path: str, payload_path=None) -> dict:
    """Engine-owned formats use server paths; other formats use payloads."""
    if ALGORITHM_CAPABILITIES[algorithm]["source"] != "none":
        return {"source": _resolve_path(path)}
    if payload_path:
        return {"data": load_payload(os.path.join(APP_DIR, payload_path), algorithm)}
    if USE_API:
        raise ValueError(
            f"API preset {algorithm!r} needs payload_path: export its "
            "checkpoint with export_payload.py in an EasySteer environment"
        )

    import easysteer.vectors as vec

    adapter = {
        "linear": vec.from_linear_transport,
        "lm_steer": vec.from_lm_steer,
        "loreft": vec.from_pyreft,
    }[algorithm]
    payload = adapter(os.path.join(APP_DIR, path))
    return {"data": payload}


def _local_spec(wire):
    """Materialize a wire dict into a SteeringSpec for llm.generate."""
    return SteeringSpec.model_validate(wire)


def _api_generate(prompt: str, config, spec_wire) -> str:
    """Call the remote vLLM server with an explicit spec or False for baseline."""
    sampling = config["sampling"]
    extra_body = {"repetition_penalty": float(sampling.get("repetition_penalty", 1.1))}
    if spec_wire is not None:
        extra_body["steering"] = spec_wire
    response = _api_client.chat.completions.create(
        model=API_MODEL_NAME,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        max_tokens=int(sampling.get("max_tokens", 128)),
        temperature=float(sampling.get("temperature", 0.0)),
        extra_body=extra_body,
    )
    return response.choices[0].message.content


def _generate_comparison(
    config_name: str, prompt: str, progress, *, scale=None, multi_vector=False
) -> Tuple[str, str]:
    """Compare baseline and steered output with one preset and sampling config."""
    try:
        config = (MULTI_CONFIGS if multi_vector else SINGLE_CONFIGS)[config_name]
        if not multi_vector and config_name in _SCALE_LOCKED_SINGLE:
            scale = float(config["steer_vector"].get("scale", 1.0))

        if USE_API:
            progress(0.2, desc="Calling API (baseline)...")
            baseline_text = _api_generate(prompt, config, False)
        else:
            progress(0, desc="Loading model...")
            llm = load_model()
            formatted_prompt = format_prompt(prompt)
            sampling_params = SamplingParams(
                temperature=float(config["sampling"].get("temperature", 0.0)),
                max_tokens=int(config["sampling"].get("max_tokens", 128)),
                repetition_penalty=float(
                    config["sampling"].get("repetition_penalty", 1.1)
                ),
            )
            progress(0.3, desc="Generating baseline...")
            baseline_out = llm.generate(
                formatted_prompt, steering=False, sampling_params=sampling_params
            )
            baseline_text = baseline_out[0].outputs[0].text

        spec_wire = (
            build_multi_spec_wire(config, _vector_source)
            if multi_vector
            else build_single_spec_wire(config, _vector_source, scale_override=scale)
        )
        if USE_API:
            progress(
                0.6,
                desc="Calling API (multi-vector steered)..."
                if multi_vector
                else "Calling API (steered)...",
            )
            steered_text = _api_generate(prompt, config, spec_wire)
        else:
            progress(
                0.6,
                desc="Generating multi-vector steered output..."
                if multi_vector
                else "Generating steered output...",
            )
            steered_out = llm.generate(
                formatted_prompt,
                steering=_local_spec(spec_wire),
                sampling_params=sampling_params,
            )
            steered_text = steered_out[0].outputs[0].text

        progress(1.0, desc="Complete!")
        return baseline_text, steered_text
    except Exception:
        logger.exception("Text generation failed")
        err = "❌ Generation failed. Please try again later."
        return err, err


def generate_single(
    config_name: str, prompt: str, scale: float, progress=gr.Progress()
) -> Tuple[str, str]:
    """Generate text using a single steering vector."""
    return _generate_comparison(config_name, prompt, progress, scale=scale)


def generate_multi(
    config_name: str, prompt: str, progress=gr.Progress()
) -> Tuple[str, str]:
    """Generate text using multiple steering vectors."""
    return _generate_comparison(config_name, prompt, progress, multi_vector=True)


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


def update_mv_ui(config_name):
    """Build multi-vector field values in Gradio output order.

    Returns:
        Description, sampling settings, group name and conflict policy,
        eight fields per vector tab, and the instruction. Unused tabs receive
        empty display values.
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
            results.extend(
                [
                    display_val(v.get("path")),
                    display_val(v.get("algorithm"), "direct"),
                    display_val(v.get("target_layers")),
                    display_val(v.get("prefill_trigger_tokens")),
                    display_val(v.get("prefill_trigger_positions")),
                    display_val(v.get("generate_trigger_tokens")),
                    display_val(str(v.get("normalize", False))),
                    float(v.get("scale", 1.0)),
                ]
            )
        else:
            results.extend(
                ["None", "None", "None", "None", "None", "None", "None", 0.0]
            )

    results.append(display_val(config["model"]["instruction"]))
    return tuple(results)


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
            <a href="https://hub.docker.com/r/xuhaolei/easysteer/tags"><img src="https://img.shields.io/badge/docker-images-orange" alt="Docker"></a>
        </div>
        <p style="color: #666; font-size: 0.9em; margin-top: 8px; max-width: 720px; margin-left: auto; margin-right: auto;">
            This online demo is for quickly testing the framework and verifying steering vector effectiveness.
            Inference uses the configured vllm-steer server or local model.
            For full features (vector extraction, training, SAE, chat, etc.), please refer to the
            <a href="https://github.com/ZJU-REAL/EasySteer?tab=readme-ov-file#frontend" target="_blank">frontend deployment guide</a>
            in the GitHub repo.
        </p>
    </div>
    """)

    first_sv_key = (
        "emotion_direct"
        if "emotion_direct" in SINGLE_CONFIGS
        else list(SINGLE_CONFIGS.keys())[0]
    )
    first_sv = SINGLE_CONFIGS[first_sv_key]
    first_mv_key = list(MULTI_CONFIGS.keys())[0]

    with gr.Tabs():
        with gr.Tab("🎯 Single Vector"):
            sv_config_dropdown = gr.Dropdown(
                choices=list(SINGLE_CONFIGS.keys()),
                value=first_sv_key,
                label="Import Configuration",
                info="Select a predefined steering configuration",
            )
            sv_description = gr.Markdown(value=_get_sv_description(first_sv_key))

            gr.Markdown("### 🤖 Sampling Configuration")
            with gr.Row():
                sv_temperature = gr.Textbox(
                    label="Temperature",
                    info="0 = greedy decoding, higher = more random",
                    placeholder="e.g. 0.0",
                    value=display_val(first_sv["sampling"].get("temperature"), "0.0"),
                    interactive=False,
                )
                sv_max_tokens = gr.Textbox(
                    label="Max Tokens",
                    info="Maximum number of tokens to generate",
                    placeholder="e.g. 128",
                    value=display_val(first_sv["sampling"].get("max_tokens"), "128"),
                    interactive=False,
                )
                sv_rep_penalty = gr.Textbox(
                    label="Repetition Penalty",
                    info="Penalize repeated tokens",
                    placeholder="e.g. 1.1",
                    value=display_val(
                        first_sv["sampling"].get("repetition_penalty"), "1.1"
                    ),
                    interactive=False,
                )

            gr.Markdown("### ⚙️ Steer Vector Configuration")
            with gr.Row():
                sv_path = gr.Textbox(
                    label="Vector Path",
                    info="Path to the steering vector file",
                    value=display_val(first_sv["steer_vector"].get("path")),
                    interactive=False,
                )
                sv_algorithm = gr.Textbox(
                    label="Algorithm",
                    info="Steering algorithm used for this vector",
                    placeholder="e.g. direct",
                    value=display_val(
                        first_sv["steer_vector"].get("algorithm"), "direct"
                    ),
                    interactive=False,
                )
                sv_target_layers = gr.Textbox(
                    label="Target Layers",
                    info="Layer indices, comma-separated",
                    placeholder="e.g. 10,11,12,...,23",
                    value=display_val(first_sv["steer_vector"].get("target_layers")),
                    interactive=False,
                )
            with gr.Row():
                sv_prefill_tokens = gr.Textbox(
                    label="Prefill Trigger Token IDs",
                    info="-1 = apply to all tokens",
                    placeholder="e.g. -1",
                    value=display_val(
                        first_sv["steer_vector"].get("prefill_trigger_tokens")
                    ),
                    interactive=False,
                )
                sv_prefill_positions = gr.Textbox(
                    label="Prefill Trigger Positions",
                    info="Supports negative indexing",
                    placeholder="e.g. -1",
                    value=display_val(
                        first_sv["steer_vector"].get("prefill_trigger_positions")
                    ),
                    interactive=False,
                )
                sv_generate_tokens = gr.Textbox(
                    label="Generate Trigger Token IDs",
                    info="-1 = apply to all tokens",
                    placeholder="e.g. -1",
                    value=display_val(
                        first_sv["steer_vector"].get("generate_trigger_tokens")
                    ),
                    interactive=False,
                )
            with gr.Row():
                sv_normalize = gr.Textbox(
                    label="Normalize",
                    info="Whether to normalize the vector",
                    value=display_val(
                        str(first_sv["steer_vector"].get("normalize", False))
                    ),
                    interactive=False,
                )
                sv_scale = gr.Slider(
                    label="Scale Factor",
                    info="Steering strength multiplier (drag to adjust)",
                    minimum=-3,
                    maximum=3,
                    step=0.1,
                    value=float(first_sv["steer_vector"].get("scale", 1.0)),
                    interactive=(first_sv_key not in _SCALE_LOCKED_SINGLE),
                )

            sv_prompt_input = gr.Textbox(
                label="Input Instruction",
                lines=3,
                value=first_sv["model"]["instruction"],
            )
            sv_generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

            gr.Markdown("### 📊 Results Comparison")
            with gr.Row():
                sv_baseline_output = gr.Textbox(
                    label="🔹 Baseline (No Steering)", lines=8, interactive=False
                )
                sv_steered_output = gr.Textbox(
                    label="🔸 Steered Output", lines=8, interactive=False
                )

            sv_config_dropdown.change(
                fn=update_sv_ui,
                inputs=[sv_config_dropdown],
                outputs=[
                    sv_description,
                    sv_temperature,
                    sv_max_tokens,
                    sv_rep_penalty,
                    sv_path,
                    sv_algorithm,
                    sv_target_layers,
                    sv_prefill_tokens,
                    sv_prefill_positions,
                    sv_generate_tokens,
                    sv_normalize,
                    sv_scale,
                    sv_prompt_input,
                ],
            )
            sv_generate_btn.click(
                fn=generate_single,
                inputs=[sv_config_dropdown, sv_prompt_input, sv_scale],
                outputs=[sv_baseline_output, sv_steered_output],
            )

        with gr.Tab("🎨 Multi-Vector"):
            first_mv = MULTI_CONFIGS[first_mv_key]
            first_mv_sv = first_mv["steer_vector"]
            first_mv_vecs = first_mv["vector_configs"]

            mv_config_dropdown = gr.Dropdown(
                choices=list(MULTI_CONFIGS.keys()),
                value=first_mv_key,
                label="Import Configuration",
                info="Select a predefined multi-vector configuration",
            )
            mv_description = gr.Markdown(value=_get_mv_description(first_mv_key))

            gr.Markdown("### 🤖 Sampling Configuration")
            with gr.Row():
                mv_temperature = gr.Textbox(
                    label="Temperature",
                    info="0 = greedy decoding, higher = more random",
                    value=display_val(first_mv["sampling"].get("temperature"), "0.0"),
                    interactive=False,
                )
                mv_max_tokens = gr.Textbox(
                    label="Max Tokens",
                    info="Maximum number of tokens to generate",
                    value=display_val(first_mv["sampling"].get("max_tokens"), "128"),
                    interactive=False,
                )
                mv_rep_penalty = gr.Textbox(
                    label="Repetition Penalty",
                    info="Penalize repeated tokens",
                    value=display_val(
                        first_mv["sampling"].get("repetition_penalty"), "1.1"
                    ),
                    interactive=False,
                )

            gr.Markdown("### ⚙️ Steer Vector Configuration")
            with gr.Row():
                mv_sv_name = gr.Textbox(
                    label="Steer Vector Name",
                    info="Identifier name for this steering vector group",
                    value=display_val(first_mv_sv.get("name")),
                    interactive=False,
                )
                mv_conflict_resolution = gr.Textbox(
                    label="Conflict Resolution",
                    info="How to combine multiple vectors",
                    value=display_val(
                        first_mv_sv.get("conflict_resolution"), "sequential"
                    ),
                    interactive=False,
                )

            gr.Markdown("### 🎯 Vector Configurations")
            mv_vec_fields = []  # flat list per vector: [path, algo, layers, pf_tokens, pf_positions, gen_tokens, normalize, scale]
            with gr.Tabs():
                for vi in range(MAX_VECTORS):
                    v_data = first_mv_vecs[vi] if vi < len(first_mv_vecs) else {}
                    with gr.Tab(f"Vector {vi + 1}"):
                        with gr.Row():
                            f_path = gr.Textbox(
                                label="Vector Path",
                                info="Path to the steering vector file",
                                value=display_val(v_data.get("path")),
                                interactive=False,
                            )
                            f_algo = gr.Textbox(
                                label="Algorithm",
                                info="Steering algorithm used for this vector",
                                value=display_val(v_data.get("algorithm"), "direct"),
                                interactive=False,
                            )
                            f_layers = gr.Textbox(
                                label="Target Layers",
                                info="Layer indices, comma-separated",
                                value=display_val(v_data.get("target_layers")),
                                interactive=False,
                            )
                        with gr.Row():
                            f_pf_tokens = gr.Textbox(
                                label="Prefill Trigger Token IDs",
                                info="-1 = apply to all tokens",
                                value=display_val(v_data.get("prefill_trigger_tokens")),
                                interactive=False,
                            )
                            f_pf_positions = gr.Textbox(
                                label="Prefill Trigger Positions",
                                info="Supports negative indexing",
                                value=display_val(
                                    v_data.get("prefill_trigger_positions")
                                ),
                                interactive=False,
                            )
                            f_gen_tokens = gr.Textbox(
                                label="Generate Trigger Token IDs",
                                info="-1 = apply to all tokens",
                                value=display_val(
                                    v_data.get("generate_trigger_tokens")
                                ),
                                interactive=False,
                            )
                        with gr.Row():
                            f_normalize = gr.Textbox(
                                label="Normalize",
                                info="Whether to normalize the vector",
                                value=display_val(str(v_data.get("normalize", False))),
                                interactive=False,
                            )
                            f_scale = gr.Slider(
                                label="Scale Factor",
                                info="Steering strength multiplier",
                                minimum=-3,
                                maximum=3,
                                step=0.1,
                                value=float(v_data.get("scale", 1.0)),
                                interactive=False,
                            )
                        mv_vec_fields.extend(
                            [
                                f_path,
                                f_algo,
                                f_layers,
                                f_pf_tokens,
                                f_pf_positions,
                                f_gen_tokens,
                                f_normalize,
                                f_scale,
                            ]
                        )

            mv_prompt_input = gr.Textbox(
                label="Input Instruction",
                lines=3,
                value=first_mv["model"]["instruction"],
            )
            mv_generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

            gr.Markdown("### 📊 Results Comparison")
            with gr.Row():
                mv_baseline_output = gr.Textbox(
                    label="🔹 Baseline (No Steering)", lines=8, interactive=False
                )
                mv_steered_output = gr.Textbox(
                    label="🎨 Steered Output (Multi-Vector)", lines=8, interactive=False
                )

            mv_config_dropdown.change(
                fn=update_mv_ui,
                inputs=[mv_config_dropdown],
                outputs=[
                    mv_description,
                    mv_temperature,
                    mv_max_tokens,
                    mv_rep_penalty,
                    mv_sv_name,
                    mv_conflict_resolution,
                    *mv_vec_fields,
                    mv_prompt_input,
                ],
            )
            mv_generate_btn.click(
                fn=generate_multi,
                inputs=[mv_config_dropdown, mv_prompt_input],
                outputs=[mv_baseline_output, mv_steered_output],
            )

    gr.Markdown("---\n*Powered by [EasySteer](https://github.com/ZJU-REAL/EasySteer)*")

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
    demo.queue(max_size=20).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
    )
