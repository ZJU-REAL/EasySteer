# EasySteer documentation

MkDocs (Material theme) documentation site, modeled on vLLM's docs setup
(mkdocs-material + mkdocstrings; vLLM additionally uses api-autonav, gen-files, minify,
redirects — we can adopt those later as the site grows).

This `docs/README.md` is excluded from the built site (`exclude_docs` in `mkdocs.yml`).

## Build locally

```bash
pip install -r requirements-docs.txt
mkdocs serve          # http://127.0.0.1:8000, live-reloads on edit
mkdocs build          # static site into site/
```

The api-reference pages use mkdocstrings, which **imports** the documented modules:

- `api-reference/steer.md` and `api-reference/hidden-states.md` need `easysteer`
  importable (i.e. torch + the easysteer install). `easysteer/__init__.py` also imports
  `easysteer.reft`, which pulls in `transformers`.
- `api-reference/steering-specs.md` needs the `vllm-steer` fork importable. If that is
  too heavy for a docs CI job, drop the page from `nav` and rely on the hand-written
  user guide.

`mkdocs build` without those packages still builds the rest of the site; the
mkdocstrings pages will error (run with `--strict` in CI to catch it, or without to
tolerate it during drafting).

## Versioned deployment (mike + GitHub Pages)

Not wired up yet. The standard flow (same as vLLM-adjacent projects):

```bash
pip install mike
mike deploy --push --update-aliases 0.1 latest   # publish this version to gh-pages
mike set-default --push latest
```

`mkdocs.yml` already declares `extra.version.provider: mike`, which renders the version
selector. A CI job (e.g. `.github/workflows/docs.yml`) should run mike on pushes to
`main` (deploy as `latest`) and on release tags (deploy as the version number).

## Status / what remains to be written

Seed content is in place; look for `TODO` comments in the pages. Highest-value gaps:

- Per-algorithm documentation (file formats, payload shapes) under user-guide.
- `easysteer.reft` training API (guide + api-reference page).
- Frontend / demo documentation.
- Docstring coverage in `easysteer.steer` (several extractors have minimal or Chinese
  docstrings; api-reference pages will render whatever is there).
- CI workflow for build + deploy (mike), link checking, and `--strict` builds.
- Logo/favicon assets (`docs/assets/`), currently commented out in `mkdocs.yml`.
- The Chinese README (`README_zh.md`) has no docs-site counterpart; decide whether to
  add mkdocs-static-i18n or keep Chinese docs as the README only.
