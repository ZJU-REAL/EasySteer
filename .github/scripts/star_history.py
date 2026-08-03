"""Render the repository's star history as a standalone SVG.

GitHub restricted the stargazers API to authenticated callers with access to
the repository, so third-party chart embeds (star-history.com, starchart.cc)
no longer work for README images. This script runs inside GitHub Actions with
the repo's own token and produces the chart we embed instead.

Stdlib only. Usage:
    python star_history.py --repo ZJU-REAL/EasySteer --out star-history.svg
"""

import argparse
import json
import math
import os
import sys
import urllib.request
from datetime import datetime, timezone

API = "https://api.github.com"
PER_PAGE = 100
MAX_PAGES = 400  # the API serves at most 40k stargazers


def fetch_star_dates(repo: str, token: str) -> list[datetime]:
    dates = []
    for page in range(1, MAX_PAGES + 1):
        req = urllib.request.Request(
            f"{API}/repos/{repo}/stargazers?per_page={PER_PAGE}&page={page}",
            headers={
                "Accept": "application/vnd.github.star+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(req) as resp:
            batch = json.load(resp)
        for entry in batch:
            dates.append(
                datetime.strptime(entry["starred_at"], "%Y-%m-%dT%H:%M:%SZ")
                .replace(tzinfo=timezone.utc)
            )
        if len(batch) < PER_PAGE:
            break
    dates.sort()
    return dates


def _nice_step(raw: float) -> float:
    power = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 5, 10):
        if raw <= mult * power:
            return mult * power
    return 10 * power


def render_svg(repo: str, dates: list[datetime]) -> str:
    if len(dates) < 2:
        raise SystemExit(f"not enough stargazer data to chart ({len(dates)} points)")

    width, height = 800, 420
    left, right, top, bottom = 64, 24, 48, 44
    plot_w, plot_h = width - left - right, height - top - bottom

    t0, t1 = dates[0].timestamp(), dates[-1].timestamp()
    span = max(t1 - t0, 1.0)
    total = len(dates)

    y_step = _nice_step(total / 4)
    y_max = y_step * math.ceil(total / y_step)

    def x_at(ts: float) -> float:
        return left + (ts - t0) / span * plot_w

    def y_at(count: float) -> float:
        return top + plot_h - count / y_max * plot_h

    points = [(x_at(d.timestamp()), y_at(i + 1)) for i, d in enumerate(dates)]
    # Thin dense series so the SVG stays small; always keep the endpoints.
    if len(points) > 400:
        stride = len(points) / 399
        points = [points[min(int(i * stride), len(points) - 1)] for i in range(400)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = f"{left:.1f},{top + plot_h} {line} {points[-1][0]:.1f},{top + plot_h}"

    grid, y_labels = [], []
    for count in range(0, int(y_max) + 1, int(y_step)):
        y = y_at(count)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="#8b949e" stroke-opacity="0.25" stroke-width="1"/>'
        )
        y_labels.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#8b949e" font-size="12">{count}</text>'
        )

    x_labels = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        ts = t0 + frac * span
        label = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%b %Y")
        anchor = "start" if frac == 0.0 else "end" if frac == 1.0 else "middle"
        x_labels.append(
            f'<text x="{x_at(ts):.1f}" y="{top + plot_h + 24}" text-anchor="{anchor}" '
            f'fill="#8b949e" font-size="12">{label}</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Star history of {repo}">
  <style>text {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}</style>
  <text x="{left}" y="26" fill="#8b949e" font-size="15" font-weight="600">{repo} &#183; GitHub stars over time</text>
  <text x="{left + plot_w}" y="26" text-anchor="end" fill="#f0a832" font-size="15" font-weight="700">&#9733; {total}</text>
  {''.join(grid)}
  {''.join(y_labels)}
  {''.join(x_labels)}
  <polygon points="{area}" fill="#f0a832" fill-opacity="0.12"/>
  <polyline points="{line}" fill="none" stroke="#f0a832" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="{points[-1][0]:.1f}" cy="{points[-1][1]:.1f}" r="4" fill="#f0a832"/>
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--out", required=True, help="output SVG path")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required (stargazer data needs auth)")

    dates = fetch_star_dates(args.repo, token)
    svg = render_svg(args.repo, dates)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"wrote {args.out}: {len(dates)} stars, "
          f"{dates[0]:%Y-%m-%d} .. {dates[-1]:%Y-%m-%d}", file=sys.stderr)


if __name__ == "__main__":
    main()
