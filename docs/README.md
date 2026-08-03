# EasySteer documentation

MkDocs (Material theme) documentation site, modeled on vLLM's docs setup
(mkdocs-material + mkdocstrings; vLLM additionally uses api-autonav, gen-files, minify,
redirects — we can adopt those later as the site grows).

This `docs/README.md` is excluded from the built site (`exclude_docs` in `mkdocs.yml`),
as is `docs/design/` — internal engineering records, linked from the contributing page.

Canonical-home principle: the top-level `README.md` is the shopfront (badges, news,
one install path, one example, pointer lines); this site is the reference. Every piece
of content has exactly one home — README changes that add detail belong here, with a
pointer line there.

## Build locally

```bash
pip install -r requirements-docs.txt
mkdocs serve          # http://127.0.0.1:8000, live-reloads on edit
mkdocs build --strict # what CI runs
```

The api-reference pages use mkdocstrings in **static-analysis mode**
(`allow_inspection: false` in `mkdocs.yml`): griffe parses the `easysteer` source tree
without importing it, so the full site — api-reference included — builds in a plain
environment with only `requirements-docs.txt`. No torch, no easysteer install, no
`vllm-steer`. The steering-spec API page is deliberately hand-written
(`api-reference/steering-specs.md`) because rendering the fork's classes would require
importing `vllm-steer`; the [Steering guide](user-guide/steering.md) is canonical for
that surface.

## CI and versioned deployment (mike + GitHub Pages)

`.github/workflows/docs.yml`:

- **PRs** touching `docs/`, `mkdocs.yml`, or `requirements-docs.txt` run
  `mkdocs build --strict`.
- **Pushes to `main`** deploy the rolling `dev` version with the `latest` alias
  (`mike deploy --push --update-aliases dev latest`).
- **Version tags** (`v*`) deploy that version (`mike deploy --push <tag>`).

One-time manual setup before the site is served: enable GitHub Pages
(Settings → Pages → deploy from branch `gh-pages`, root), and run
`mike set-default --push latest` so the site root redirects. `mkdocs.yml` already
declares `extra.version.provider: mike`, which renders the version selector.

## Status / what remains to be written

Seed content is in place; look for `TODO` comments in the pages. Highest-value gaps:

- Per-algorithm documentation (file formats, payload shapes) under user-guide.
- `easysteer.reft` API surface (guide seed exists at `user-guide/reft-training.md`;
  api-reference page pending docstring coverage).
- Frontend tour/screenshots (seed exists at `user-guide/web-demo.md`).
- Docstring coverage in `easysteer.steer` (several extractors have minimal or Chinese
  docstrings; api-reference pages will render whatever is there).
- Link checking in CI.
- Logo/favicon assets (`docs/assets/`), currently commented out in `mkdocs.yml`.
- The Chinese README (`README_zh.md`) mirrors the English shopfront; decide whether to
  add mkdocs-static-i18n for a Chinese docs site.
