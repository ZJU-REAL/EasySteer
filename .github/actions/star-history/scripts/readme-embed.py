#!/usr/bin/env python3
"""Rewrite the README star-history marker block.

readme-format=picture -> <picture> with relative per-theme SVG paths
                         (GitHub renders it with dark/light support).
readme-format=png      -> plain-markdown PNG at an ABSOLUTE raw URL, the only
                         form that renders on npm and pub.dev (their sanitizers
                         drop <picture>; raw SVG is served as text/plain and
                         will not render off GitHub).
"""
import os, re, sys

readme = os.environ["README"]
light = os.environ["LIGHT"]
dark = os.environ.get("DARK", "")
png = os.environ.get("PNG", "")
fmt = os.environ.get("README_FORMAT", "picture")
repo = os.environ.get("REPO", "")
branch = os.environ.get("BRANCH", "main")
start = "<!-- star-history:start -->"
end = "<!-- star-history:end -->"

if not os.path.exists(readme):
    print(f"{readme} not found; skipping README update.")
    sys.exit(0)
text = open(readme, encoding="utf-8").read()
if start not in text or end not in text:
    print("star-history markers not found in README; skipping update.")
    sys.exit(0)

link = f"https://star-history.com/#{repo}&Date" if repo else "https://star-history.com"

if fmt == "png":
    raw = f"https://raw.githubusercontent.com/{repo}/{branch}/{png}"
    body = f'[![Star History]({raw})]({link})'
elif dark:
    body = (
        '<picture>\n'
        f'  <source media="(prefers-color-scheme: dark)" srcset="{dark}">\n'
        f'  <img alt="Star history" src="{light}">\n'
        '</picture>'
    )
else:
    body = f'<img alt="Star history" src="{light}">'

block = f"{start}\n{body}\n{end}"
# Replace only the FIRST marker pair; later literal markers (docs/examples)
# must not be clobbered.
new = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: block, text, count=1, flags=re.S)
open(readme, "w", encoding="utf-8").write(new)
print(f"Updated {readme} between markers ({fmt}).")
