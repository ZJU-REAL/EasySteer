# Maintaining the documentation

The MkDocs site is the reference for installation, public APIs, examples and
contributor workflows. The English and Chinese root READMEs introduce the project
and link to the relevant guides. Update both READMEs when their shared example or
installation instructions change.

## Build locally

```bash
python -m pip install -r requirements-docs.txt
mkdocs serve          # http://127.0.0.1:8000
mkdocs build --strict # also run in CI
```

API pages use mkdocstrings in static-analysis mode (`allow_inspection: false`), so
building the site requires only `requirements-docs.txt`. It does not import torch
or vLLM. Check out the recorded submodule with
`git submodule update --init --recursive` before building: the fork's module
directories are source search paths for its class and method references. Lazy public
exports are documented through their defining modules, with public import names
in the rendered headings. The official `griffe-pydantic` extension renders spec
fields without computing JSON schemas or importing model classes.
A successful site build checks rendering, navigation and the Python
identifiers referenced by mkdocstrings; it does not import or execute fenced
examples. Validate changed examples
against the matching package or engine API and run the relevant tests described
in [Testing](developer-guide/testing.md).

Keep reproducible usage, API contracts, architecture and general testing methods
in the public documentation. Keep machine-specific paths, process IDs, individual
run logs, migration worklists and superseded proposals in the Git-ignored
`.local/development/` directory. Do not link public pages to those local records.

## CI and deployment

The docs workflow runs a strict build for pull requests that change documentation,
READMEs or the package source used by the API reference. Pushes to `main` deploy
the rolling `dev` version. Version tags (`v*`) publish a versioned snapshot and
update the `latest` alias through mike.

For initial GitHub Pages setup, select the `gh-pages` branch and its root directory
in repository settings, then run `mike set-default --push latest`. The version
selector is configured in `mkdocs.yml`.

This README is excluded from the rendered navigation; public guides live under
the directories listed in `mkdocs.yml`.
