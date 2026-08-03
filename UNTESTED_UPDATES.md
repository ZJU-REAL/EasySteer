# Updated but not yet tested

Code brought up to date with API changes without runtime verification —
the APIs may still move, so these carry a note instead of a test run.
Remove entries as they get exercised.

| Change | Files | How to verify later |
| --- | --- | --- |
| Data-only algorithms (`linear`, `lm_steer`, `loreft`) route through the `easysteer.vectors` payload adapters (`data=`) instead of `source=` paths, which the engine now rejects at admission | `frontend/core/steer_request_builder.py` (`_vector_source_kwargs`) | Frontend smoke test: run the `emoji_loreft` inference config end-to-end |
| `model_type` inference block and kwarg removed from the extraction endpoint (parameter no longer exists on extractors) | `frontend/extraction_api.py` | Frontend smoke test: run an extraction config end-to-end |
| `model_type=` stripped from the extractor call | `experiment/hallucination/data.ipynb` | Runs when the hallucination directory is un-deferred |
| Extractor example no longer passes `model_type`; LoReFT loading shown via `data=easysteer.vectors.from_pyreft(...)` | `docs/user-guide/extracting-vectors.md`, `docs/user-guide/reft-training.md` | Docs build (strict) covers rendering; examples mirror tested notebook usage |
| HF Space demo ported to the v2 steering API: local mode builds `SteeringSpec` objects (`steering=` on `llm.generate`, baseline = no steering), API mode sends the v2 wire under `extra_body={"steering": ...}` (the old `steer_vector_request` extra-body key no longer exists); data-only algorithms go through the payload adapters, with tensor bytes base64-encoded for JSON transport; dead unique-id helpers removed | `hf-space/app.py`, `hf-space/requirements.txt` | Deploy the Space in both modes after the API stabilizes; exercise every bundled config (incl. `emoji_loreft`, which needs easysteer importable) |
| Frontend parameter audit: algorithm dropdowns (inference panel, multi-vector, SAE explore) list the full engine registry (direct/erase/replace/concept_replace/linear/lm_steer/loreft/moe_router) with new i18n keys; the builder routes raw `.pt` directions for direct/erase/replace through `from_pt_direction` (the cat/chinese chat presets steer `.pt` files the engine no longer loads by path) | `frontend/static/templates/inference.html`, `frontend/static/js/multi-vector.js`, `frontend/static/js/sae-explore.js`, `frontend/i18n.js`, `frontend/core/steer_request_builder.py` | Frontend smoke test: each algorithm option end-to-end; chat cat/chinese presets |
| Docker build pins moved off the 0.17.1 era: `SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0+easysteer` and `VLLM_MERGE_BASE_COMMIT=568afb3a…` (the upstream parent of the v0.26.0 port commit, for the precompiled-wheel download) | `docker/build.sh` | Full `bash docker/build.sh` + `docker_test.py` run after the API stabilizes (`docker/docker_test.py` itself is already v2 and needed no changes) |

Checked and found already current (no changes needed): the frontend's
v1→v2 trigger-field translation (`build_apply_specs`), `chat_api` /
`inference_api` / `demo_training` (legacy wire-field names are
intentional UI contract), `easysteer/hidden_states` compatibility
wrappers, `docs/user-guide/steering.md`, and `docs/api-reference/`.
