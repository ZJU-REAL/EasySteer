# Troubleshooting

Start from the first error in the engine log. A later worker-exit or connection
error often reports that an earlier loading, validation, or compilation step
failed.

## Installation and model loading

### `vllm.steer_vectors` cannot be imported

The official vLLM package does not include EasySteer's additions. Check which
interpreter and package your command uses:

```bash
python -c 'import sys, vllm; print(sys.executable); print(vllm.__file__)'
python -c 'from vllm.steer_vectors import SteeringSpec; print(SteeringSpec)'
git submodule status vllm-steer
```

Use the same environment for installation and execution. If you installed the
fork with a file overlay, reinstalling `vllm` replaces that overlay; reapply it
from the submodule commit recorded by EasySteer. See
[installation](installation.md).

### A kernel module or shared library is missing

Keep the official wheel, fork source, PyTorch, and CUDA package versions aligned.
An editable/precompiled installation includes files supplied by the wheel that
may not exist in the source checkout. Synchronizing the checkout with a deletion
option can remove those files from an otherwise working installation.

Use the documented precompiled install or overlay without `--delete`. Check the
imported package path before rebuilding. A missing Python interface module and
a CUDA compilation error have different causes.

### The first run compiles kernels

The wheel supplies prebuilt components, but the selected backend and graph
configuration can still perform JIT compilation at startup or first use. Keep
the compilation cache and separate this initialization time from steady-state
generation measurements. See [performance](../user-guide/performance.md).

For `ld: cannot find -lcudart`, check the toolkit's library directory and
`libcudart.so` link as described under
[CUDA library discovery](installation.md#cuda-library-discovery).

### CUDA runtime and driver do not match

Check `nvidia-smi` on the machine actually running the model. A container or
Python environment supplies its runtime libraries but still uses the host
driver. Choose the compatible build described in
[Docker installation](installation.md#docker), or update the host driver.
Do not copy a working environment to another machine and assume its driver is
identical.

## Steering requests

| Symptom | What to check |
|---|---|
| Workload not declared / undeclared algorithm | Pass `steer_algorithms=[...]` at startup and include every requested algorithm, or use `"all"` for an exploratory `split` engine. |
| Multi-vector request rejected | Start with `steer_multi_vector=True`; multi-vector composition needs `split`. |
| In-graph rank or routing-mode error | Check the [capability table](../api-reference/algorithms.md). Raise the startup rank bound for a known checkpoint, or choose `split`. |
| `normalize=True` rejected | Normalization is algorithm-specific; attention, LoReFT, LM-Steer, linear, and router steering require `False`. |
| Payload/source format rejected | Use an EasySteer native source or an explicit checkpoint adapter. A `.pt` suffix alone is not a schema. |
| Source file not found over HTTP | `source` is resolved on the vLLM server, relative to its working directory. Use a server path or send an encoded payload. |
| Layer or width mismatch | Use weights from the correct model and component. `layers` restricts recorded layer IDs; it does not relocate weights. |
| Preload required | Preload the file with the same algorithm and payload-changing router parameters before submitting the request. |

### The output appears unsteered

Compare the exact prompt, chat template, vector file, model, layers, scale, and
selection with the working example. Positive and negative direct-steering
coefficients move in opposite directions. The happy demo uses a positive scale
of `2.0` with its bundled Qwen vector.

Use `steering=False` for the baseline and an explicit spec for the comparison.
An omitted `steering` inherits the engine default. Selectors also matter: a
generation-only intervention does not change the prompt forward pass that
predicts the first answer token. Add `prompt_positions=[-1]` when that prediction
should be steered.

## Capture

- **Unsupported runner or worker count:** capture uses the V2 GPU runner and the
  public client helper requires one worker. Keep `tensor_parallel_size=1`.
- **Empty rows:** confirm the token selection matches a forward pass. With
  `max_tokens=1`, there are prompt rows but no decode-forward rows for a
  generation-only selector.
- **Unexpected layer IDs after extraction:** pass `CaptureResult` directly to
  extractors. Converting a subset to a plain nested list loses its true layer
  mapping.
- **No attention or MoE component:** check the model's component support.
  Attention head capture excludes MLA and encoder/cross-attention; fused MoE
  implementations may bypass a gate module.
- **Incomplete prefix capture:** use `hs.capture()` to establish the selection
  before request admission. Starting a raw capture session after selected
  prompt rows were skipped by a cache hit cannot recover those activations.

Capture normally manages cache reads automatically. There is no need to add
random salts or disable prefix caching for the whole engine. See the
[capture guide](../user-guide/hidden-state-capture.md).

## Serving and demos

Check the backend before debugging a UI:

```bash
curl http://localhost:8017/health
curl http://localhost:8017/v1/models
```

If the server requires an API key, include its bearer authorization header.
Use the returned model ID in requests. Configure clients with a base URL ending
in `/v1`, and confirm that URL is reachable from the client host. The Vue
frontend's generation requests originate in the browser; the Hugging Face API
demo's requests originate in the Space.

An HTTP 400 usually contains an actionable request-validation message. A
connection refusal means the host/port is not accepting the connection; a
timeout can also involve routing or a backend still initializing. Consult the
server log and its health endpoint instead of changing steering fields to fix
a transport problem.

For a bug report, include package versions, GPU/driver, the engine arguments,
the spec, a minimal prompt, and the first error. Remove API keys and other
credentials from shared logs.
