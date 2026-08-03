# Star History Action

Keep a self-updating star history chart in **your own** repository's README.

> **Unofficial.** Not affiliated with or endorsed by [star-history.com](https://www.star-history.com). It reuses their open-source renderer with credit (see [Credits](#credits)).

## Why

On June 30, 2026 GitHub restricted its stargazers endpoint to a repo's own admins and collaborators, so the hosted `api.star-history.com/svg` badge now renders blank for many repos.

A repo's owner or collaborator can still read their own repo's stargazers. This action does exactly that: it runs in your CI with your own access, renders the chart, and commits it into your repo (an SVG, plus a PNG for package registries) so the README embeds a static file. It is for charting repos you own or collaborate on. It does not scrape star-history.com and embeds only the repo owner's avatar, no individual stargazer's identity.

The chart is drawn by [star-history's own renderer](https://github.com/star-history/star-history), vendored under `renderer/vendor` and run in Node, so output matches star-history.com without a headless browser or third-party CLI. See `renderer/NOTICE.md` for the pinned commit and attribution.

## Endorsement

The star-history maintainer pointed to this approach in the tracking issue, for anyone who would rather not hand a fine-grained token to the star-history API:

> If you are not comfortable to share your fine-grained token with star-history API, then @narayann7's method is good (the tradeoff is it's a static image, though you can configure a cron to refresh it periodically).
>
> star-history/star-history#539: https://github.com/star-history/star-history/issues/539#issuecomment-4896077391

## Demo

This repo runs the action on itself; the chart below is the real, self-updating output. It refreshes when the repo gains a star.

<!-- star-history:start -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/star-history/star-history-dark.svg">
  <img alt="Star history" src="assets/star-history/star-history-light.svg">
</picture>
<!-- star-history:end -->

## How it works

```
  schedule / new star / manual dispatch
             │
             ▼
  ┌──────────────────────────────────┐
  │  GitHub Action (runs in your CI)  │
  │                                    │
  │  render chart ──► changed?         │
  │       ├─ yes ─► commit SVG         │
  │       │         + update README    │
  │       └─ no  ─► do nothing         │
  └────────────────┬───────────────────┘
                   ▼
    README <picture> shows the chart
```

The action renders with your own token and commits only when the star data actually changes. For the full `render.ts` pipeline and change-detection logic, see [docs/architecture.md](docs/architecture.md).

## Usage

Add `.github/workflows/star-history.yml`:

```yaml
name: Star History

on:
  schedule:
    - cron: '0 */6 * * *'   # every 6 hours; see interval table below
  watch:
    types: [started]        # optional: also refresh right after a new star
  workflow_dispatch:

permissions:
  contents: write

concurrency:
  group: star-history
  cancel-in-progress: true  # collapse a burst of stars into one run

jobs:
  star-history:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: narayann7/star-history-action@v1
        with:
          repos: ${{ github.repository }}   # or a list: owner/repo,owner/repo2
```

Then add two marker comments to your README where you want the chart, and leave them empty:

```html
<!-- star-history:start -->
<!-- star-history:end -->
```

On each run the action fills the space between them with the current chart and updates it when the chart changes.

Files use stable names (`star-history-light.svg`, `star-history-dark.svg`, `star-history.png`) and are overwritten in place. A fixed path matters for package registries: npm and pub.dev freeze a README's image URL, so a moving filename would 404 once the old file is deleted. On GitHub the chart still refreshes because a push purges GitHub's image cache. The `<picture>` block also swaps in the dark chart automatically on GitHub's dark theme.

If your README is also published to npm or pub.dev, set `readme-format: png`. Those sites strip `<picture>` and do not render SVG, so the action writes a plain-markdown PNG at an absolute `raw.githubusercontent.com` URL instead. The tradeoff is a single PNG with no dark/light swap.

To manage the embed yourself, set `update-readme: false` and point a plain `<img>` at whatever the action writes.

## Inputs

| input | default | description |
|---|---|---|
| `repos` | current repo | Comma-separated `owner/repo` list. |
| `output-dir` | `assets/star-history` | Where chart files (SVGs and PNG) are written. |
| `token` | `${{ github.token }}` | Token for the stargazers API. |
| `type` | `Date` | `Date` or `Timeline`. |
| `themes` | `light,dark` | Comma list of themes to render. |
| `width` | `800` | Image width in pixels. |
| `font-family` | `` | Optional [Google Fonts](https://fonts.google.com/) family for the PNG (e.g. `Patrick Hand`). Empty uses the bundled Comic Neue. |
| `update-readme` | `true` | Rewrite the README between the `star-history` markers to point at the newest chart. |
| `readme` | `README.md` | Path to the README to update. |
| `readme-format` | `picture` | `picture` (SVG `<picture>`, GitHub dark/light) or `png` (plain-markdown absolute-URL PNG that also renders on npm and pub.dev). |
| `commit` | `true` | Commit and push the generated files. |
| `commit-message` | `chore: update star history [skip ci]` | Commit message. |

Outputs: `files` (newline-separated generated paths), `changed` (`true`/`false`), `light` and `dark` (newest SVG paths), and `png` (PNG path).

**Fonts:** GitHub strips `@font-face` from SVGs it serves through `<img>`/`<picture>`, so a custom font renders only in the PNG output (`readme-format: png`), not the SVG. Set `font-family` to any Google Fonts name to restyle the PNG. The action downloads the font and reads its real internal name so it renders even when that differs from the family string. If the download fails, it falls back to Comic Neue and the run still succeeds. Non-Latin families (e.g. `Noto Sans SC`) work too, though single-weight fonts render the title in the regular weight since no bold face exists.

## Triggers

- **schedule** runs on a fixed cadence. Recommended: only the schedule refreshes the time axis on days with no star change, and only it can pick up unstars. This repo omits it (running on `watch` alone to keep the demo minimal), which means its chart will not refresh on quiet days or reflect an unstar.
- **watch / started** (optional) runs right after someone stars the repo, so a new star shows without waiting for the next scheduled run. It fires on new stars only, never unstars, so it supplements the schedule. With `concurrency: cancel-in-progress: true`, a burst of stars collapses into one run. The workflow file must be on your default branch for this trigger to fire.
- **workflow_dispatch** gives a manual run button for the first run and ad-hoc refreshes.

A `push` trigger is possible but **not recommended**: it re-runs on every commit, wasting CI minutes and adding churn without fresher data.

### Cron intervals

Swap the `cron` line for whichever cadence fits. All times are UTC.

| interval | cron | recommended |
|---|---|---|
| 5m | `*/5 * * * *` | no, testing only |
| 1h | `0 * * * *` | rarely |
| 3h | `0 */3 * * *` | yes |
| 6h | `0 */6 * * *` | yes (default) |
| 12h | `0 */12 * * *` | yes |
| daily | `0 0 * * *` | yes |
| weekly | `0 0 * * 0` | fine |

**Pick 3h, 6h, or daily.** Star counts move slowly. The 5-minute option is for testing only. GitHub's minimum interval is 5 minutes, scheduled runs fire approximately rather than on the dot, and a repo with no activity for 60 days has its scheduled runs paused.

## Token

The default `${{ github.token }}` is the automatic token GitHub injects into every run, scoped to the repo the workflow lives in. For your own repo that satisfies the stargazers restriction, so **most users need no personal token**.

Only if a run fails as unauthorized (e.g. charting a repo the default token cannot read) supply a personal access token via the `token` input from a secret:

```yaml
        with:
          token: ${{ secrets.GH_PAT }}
```

Use the **least privilege that works**: a classic token with only `public_repo` scope, or a fine-grained token with read-only access limited to the repos you chart. A token with no scopes does not work against the stargazers endpoint.

The `token` input reads stargazers only. The commit and push are authorized by the credentials `actions/checkout` persists (the default `GITHUB_TOKEN`), which is why `permissions: contents: write` is required. A PAT here does not authorize the push. Because the push uses the default token, it does not trigger other `on: push` workflows and can be rejected on a branch with required status checks.

## Rate limits

Each refresh reads the stargazers API, roughly 40 requests per run (one per stargazer page, plus a count and the owner avatar, per theme). The automatic `${{ github.token }}` is capped at **1000 requests/hour per repo** (shared across every workflow in that repo); a personal access token is capped at **5000 requests/hour per token**.

A single refresh is well under either. The failure mode is a *burst*: if `watch` fires a run per star for many stars landing close together, the queued runs can drain the hourly quota and a run then fails on its first request with a `403`. star-history reports every 403 as "rate limit exceeded", so an access problem (a token that cannot read the repo) looks the same in the log.

The action handles both so a transient limit does not fail your workflow:

- The `watch` workflow uses `concurrency: cancel-in-progress: true`, so a burst collapses into one refresh instead of one run per star.
- If a refresh hits a `403`/`401` while a chart is **already committed**, the run keeps that chart, warns, and exits cleanly, then refreshes on the next run once the limit resets. Only the **first** run (no chart yet) fails on a `403`, since there is nothing to keep and it usually means the token cannot read the target repo.

If you chart in volume or hit this repeatedly, prefer a spread-out `schedule` over `watch`, or pass a personal access token via `token` for the higher 5000/hour limit.

## Limitation

Charts need a token that can read the repo's stargazers. Your own repos always work with the default `${{ github.token }}`, and most other public repos work with a personal access token. GitHub's stargazers restriction can still block some repos, and when it does the chart comes back empty with no workaround.

## Credits

Chart rendering is powered by [star-history](https://github.com/star-history/star-history) (MIT). Their chart code is vendored under `renderer/vendor` and does the real work of turning stargazer data into an SVG. This action is a thin wrapper that runs it in CI and commits the result. Thanks to the star-history maintainers.

## License

MIT. See [LICENSE](./LICENSE). It bundles star-history's code under `renderer/vendor`, also MIT, kept intact at [`renderer/vendor/LICENSE`](./renderer/vendor/LICENSE) with attribution and the pinned commit in [`renderer/NOTICE.md`](./renderer/NOTICE.md).
