"""Gradio steering demo with remote API and local GPU modes.

DEMO_MODE defaults to api, which requires VLLM_API_URL and VLLM_MODEL_NAME.
Optional API settings are VLLM_API_KEY and VLLM_VECTOR_BASE_PATH. Set
DEMO_MODE=gpu to load the model locally with vLLM.
"""

import json
import logging
import os
from typing import Any

from runtime import demo_mode, load_steering_spec

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


def _load_config_dir(subdir: str) -> dict[str, Any]:
    """Load JSON presets from one directory in filename order."""
    directory = os.path.join(CONFIGS_DIR, subdir)
    configs = {}
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename)) as file:
                configs[filename[:-5]] = json.load(file)
    return configs


SINGLE_CONFIGS = _load_config_dir("inference")
MULTI_CONFIGS = _load_config_dir("multi_vector")

SINGLE_CONFIG_DESCRIPTIONS: dict[str, str] = {
    "emotion_direct": "Steers the model to respond in a happier, more positive tone — even in contexts where sadness would be expected.",
    "emoji_loreft": (
        "Steers the model to include emojis in output. "
        "Note: this is for testing only — the LoReFT vector was trained on very few "
        "examples (~dozens), so it works reliably only on certain prompts."
    ),
    "adult_style": "Steers output to be more aligned with adult interests and preferences.",
    "refuse_control": "Makes the model tend to refuse answering, even for normal and harmless requests.",
}
MULTI_CONFIG_DESCRIPTIONS: dict[str, str] = {
    "refusal_direction": (
        "Makes the model tend to refuse answering even normal requests. "
        "Achieved by applying a different steering vector at each of the "
        "last 4 tokens of the prompt."
    ),
}

# The bundled LoReFT example uses its trained scale.
_SCALE_LOCKED_SINGLE = {"emoji_loreft"}


def load_model():
    """Load the LLM model with steering support."""
    global llm_instance
    if llm_instance is None:
        print("🔄 Loading model...")
        llm_instance = LLM(
            model=MODEL_NAME,
            enable_steer_vector=True,
            steer_algorithms=",".join(
                sorted(
                    {
                        vector["algorithm"]
                        for config in (
                            *SINGLE_CONFIGS.values(),
                            *MULTI_CONFIGS.values(),
                        )
                        for vector in config["steering"]["vectors"]
                    }
                )
            ),
            steer_multi_vector=True,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            tensor_parallel_size=1,
        )
        print("✅ Model loaded successfully!")
    return llm_instance


def _resolve_path(relative_path: str) -> str:
    """In API mode, convert a relative vector path to an absolute server path."""
    if USE_API and VECTOR_BASE_PATH:
        return os.path.join(VECTOR_BASE_PATH, relative_path)
    return relative_path if USE_API else os.path.join(APP_DIR, relative_path)


def _api_generate(messages, config, spec_wire) -> str:
    """Call the remote vLLM server with an explicit spec or False for baseline."""
    sampling = config["sampling"]
    response = _api_client.chat.completions.create(
        model=API_MODEL_NAME,
        messages=messages,
        max_tokens=sampling["max_tokens"],
        temperature=sampling["temperature"],
        extra_body={
            "repetition_penalty": sampling["repetition_penalty"],
            "steering": spec_wire,
        },
    )
    return response.choices[0].message.content


def _generate_comparison(
    config_name: str, prompt: str, progress, *, scale=None, multi_vector=False
) -> tuple[str, str]:
    """Compare baseline and steered output with one preset and sampling config."""
    try:
        config = (MULTI_CONFIGS if multi_vector else SINGLE_CONFIGS)[config_name]
        if not multi_vector and config_name in _SCALE_LOCKED_SINGLE:
            scale = config["steering"]["vectors"][0]["scale"]
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]

        if USE_API:
            progress(0.2, desc="Calling API (baseline)...")
            baseline_text = _api_generate(messages, config, False)
        else:
            progress(0, desc="Loading model...")
            llm = load_model()
            tokenized_prompt = {
                "prompt_token_ids": llm.get_tokenizer().apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                )
            }
            sampling_params = SamplingParams(**config["sampling"])
            progress(0.3, desc="Generating baseline...")
            baseline_out = llm.generate(
                tokenized_prompt, steering=False, sampling_params=sampling_params
            )
            baseline_text = baseline_out[0].outputs[0].text

        spec_wire = load_steering_spec(
            config, _resolve_path, app_dir=APP_DIR, scale_override=scale
        )
        if USE_API:
            progress(
                0.6,
                desc="Calling API (multi-vector steered)..."
                if multi_vector
                else "Calling API (steered)...",
            )
            steered_text = _api_generate(messages, config, spec_wire)
        else:
            progress(
                0.6,
                desc="Generating multi-vector steered output..."
                if multi_vector
                else "Generating steered output...",
            )
            steered_out = llm.generate(
                tokenized_prompt,
                steering=SteeringSpec.model_validate(spec_wire),
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
) -> tuple[str, str]:
    """Generate text using a single steering vector."""
    return _generate_comparison(config_name, prompt, progress, scale=scale)


def generate_multi(
    config_name: str, prompt: str, progress=gr.Progress()
) -> tuple[str, str]:
    """Generate text using multiple steering vectors."""
    return _generate_comparison(config_name, prompt, progress, multi_vector=True)


SAMPLING_FIELDS = (
    ("temperature", "Temperature", None),
    ("max_tokens", "Max Tokens", 0),
    ("repetition_penalty", "Repetition Penalty", None),
)
MAX_VECTORS = max(len(c["steering"]["vectors"]) for c in MULTI_CONFIGS.values())


def _sampling_values(config):
    return [config["sampling"][key] for key, _, _ in SAMPLING_FIELDS]


def _sampling_fields(config):
    with gr.Row():
        return [
            gr.Number(
                value=config["sampling"][key],
                label=label,
                precision=precision,
                interactive=False,
            )
            for key, label, precision in SAMPLING_FIELDS
        ]


def _vector_values(vector):
    return [
        vector.get("source", vector.get("payload_path", "")),
        vector.get("algorithm", ""),
        json.dumps(vector.get("layers", [])),
        vector.get("apply", {}),
        vector.get("normalize", False),
        vector.get("scale", 0.0),
    ]


def _vector_fields(vector, *, adjustable_scale=False):
    source, algorithm, layers, apply, normalize, scale = _vector_values(vector)
    with gr.Row():
        source_field = gr.Textbox(
            label="Source / Payload File",
            value=source,
            interactive=False,
        )
        algorithm_field = gr.Textbox(
            label="Algorithm",
            value=algorithm,
            interactive=False,
        )
    with gr.Row():
        layers_field = gr.Textbox(
            label="Layers", value=layers, lines=2, interactive=False
        )
        apply_field = gr.JSON(label="Apply", value=apply, open=True)
    with gr.Row():
        normalize_field = gr.Checkbox(
            label="Normalize",
            value=normalize,
            interactive=False,
            info="Rescale the steered hidden state to its original norm.",
        )
        scale_field = gr.Slider(
            label="Scale",
            info="Steering strength multiplier",
            minimum=-3,
            maximum=3,
            step=0.1,
            value=scale,
            interactive=adjustable_scale,
        )
    return [
        source_field,
        algorithm_field,
        layers_field,
        apply_field,
        normalize_field,
        scale_field,
    ]


def _comparison_fields(instruction, *, multi_vector=False):
    prompt = gr.Textbox(label="Input Instruction", lines=3, value=instruction)
    generate = gr.Button("🚀 Generate", variant="primary", size="lg")
    gr.Markdown("### 📊 Results Comparison")
    with gr.Row():
        baseline = gr.Textbox(
            label="🔹 Baseline (No Steering)",
            lines=8,
            interactive=False,
        )
        steered = gr.Textbox(
            label="🎨 Steered Output (Multi-Vector)"
            if multi_vector
            else "🔸 Steered Output",
            lines=8,
            interactive=False,
        )
    return prompt, generate, [baseline, steered]


def update_sv_ui(config_name):
    config = SINGLE_CONFIGS[config_name]
    fields = _vector_values(config["steering"]["vectors"][0])
    fields[-1] = gr.update(
        value=fields[-1],
        interactive=config_name not in _SCALE_LOCKED_SINGLE,
    )
    return (
        SINGLE_CONFIG_DESCRIPTIONS.get(config_name, ""),
        *_sampling_values(config),
        *fields,
        config["instruction"],
    )


def update_mv_ui(config_name):
    config = MULTI_CONFIGS[config_name]
    vectors = config["steering"]["vectors"]
    fields = [
        field
        for i in range(MAX_VECTORS)
        for field in _vector_values(vectors[i] if i < len(vectors) else {})
    ]
    return (
        MULTI_CONFIG_DESCRIPTIONS.get(config_name, ""),
        *_sampling_values(config),
        config["steering"].get("conflict", "priority"),
        *fields,
        config["instruction"],
    )


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

    first_sv_key = "emotion_direct"
    first_sv = SINGLE_CONFIGS[first_sv_key]
    first_mv_key = next(iter(MULTI_CONFIGS))
    first_mv = MULTI_CONFIGS[first_mv_key]

    with gr.Tabs():
        with gr.Tab("🎯 Single Vector"):
            sv_config_dropdown = gr.Dropdown(
                choices=list(SINGLE_CONFIGS),
                value=first_sv_key,
                label="Import Configuration",
                info="Select a predefined steering configuration",
            )
            sv_description = gr.Markdown(SINGLE_CONFIG_DESCRIPTIONS[first_sv_key])
            gr.Markdown("### 🤖 Sampling Configuration")
            sv_sampling = _sampling_fields(first_sv)
            gr.Markdown("### ⚙️ Steering Configuration")
            sv_fields = _vector_fields(
                first_sv["steering"]["vectors"][0],
                adjustable_scale=first_sv_key not in _SCALE_LOCKED_SINGLE,
            )
            sv_prompt, sv_generate, sv_outputs = _comparison_fields(
                first_sv["instruction"]
            )
            sv_config_dropdown.change(
                fn=update_sv_ui,
                inputs=[sv_config_dropdown],
                outputs=[sv_description, *sv_sampling, *sv_fields, sv_prompt],
            )
            sv_generate.click(
                fn=generate_single,
                inputs=[sv_config_dropdown, sv_prompt, sv_fields[-1]],
                outputs=sv_outputs,
            )

        with gr.Tab("🎨 Multi-Vector"):
            mv_config_dropdown = gr.Dropdown(
                choices=list(MULTI_CONFIGS),
                value=first_mv_key,
                label="Import Configuration",
                info="Select a predefined multi-vector configuration",
            )
            mv_description = gr.Markdown(MULTI_CONFIG_DESCRIPTIONS[first_mv_key])
            gr.Markdown("### 🤖 Sampling Configuration")
            mv_sampling = _sampling_fields(first_mv)
            gr.Markdown("### ⚙️ Steering Configuration")
            mv_conflict = gr.Textbox(
                label="Conflict",
                info="How to combine vectors at the same position",
                value=first_mv["steering"].get("conflict", "priority"),
                interactive=False,
            )
            mv_fields = []
            with gr.Tabs():
                for i in range(MAX_VECTORS):
                    vectors = first_mv["steering"]["vectors"]
                    vector = vectors[i] if i < len(vectors) else {}
                    with gr.Tab(f"Vector {i + 1}"):
                        mv_fields.extend(_vector_fields(vector))
            mv_prompt, mv_generate, mv_outputs = _comparison_fields(
                first_mv["instruction"],
                multi_vector=True,
            )
            mv_config_dropdown.change(
                fn=update_mv_ui,
                inputs=[mv_config_dropdown],
                outputs=[
                    mv_description,
                    *mv_sampling,
                    mv_conflict,
                    *mv_fields,
                    mv_prompt,
                ],
            )
            mv_generate.click(
                fn=generate_multi,
                inputs=[mv_config_dropdown, mv_prompt],
                outputs=mv_outputs,
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
        load_model()
        print(f"✅ Model loaded: {MODEL_NAME}")

    print("\n🌐 Launching Gradio interface...")
    demo.queue(max_size=20).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
    )
