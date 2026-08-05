# README snapshot from before the documentation site

Status: **superseded.** This file preserved the repository `README.md` as it
stood before the documentation-site rework (mkdocs site under `docs/`,
`mkdocs.yml` at the repo root), so that no content was lost while it was
being migrated. The migration is complete and everything the snapshot
contained now lives in maintained locations:

| Old README section | Now lives at |
|---|---|
| About / feature bullets, News, badges | `README.md` |
| Installation (wheel, source build, Docker) | `docs/getting-started/installation.md` |
| Quick example | `docs/getting-started/quickstart.md` and `README.md` |
| OpenAI-compatible API usage | `docs/user-guide/openai-server.md` |
| Steering spec examples (`SteeringSpec`/`VectorSpec`/`ApplySpec`) | `docs/user-guide/steering.md`, `vllm-steer/docs/features/steer_vectors.md` |
| Adding a new algorithm | `docs/developer-guide/contributing.md` |
| Hidden-states extraction | `docs/user-guide/hidden-state-capture.md` |
| Vector extraction (`easysteer.steer`) | `docs/user-guide/extracting-vectors.md` |
| ReFT training (`easysteer.reft`) | `docs/user-guide/reft-training.md` |
| Frontend / web demo | `docs/user-guide/web-demo.md` |
| Paper replication table | `docs/replications/index.md` |
| License, usage statement, citation, acknowledgements | `README.md` |

The snapshot itself has been removed from this file because it predated the
vLLM 0.26.0 migration and contained stale facts (v0.17.1 wheel pin and Docker
tag, capture requiring eager execution, pre-declaration engine flags). For
the pre-rework text, see the git history of `README.md` (or of this file).
