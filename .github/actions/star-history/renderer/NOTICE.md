# Vendored code attribution

`vendor/shared/` is derived from the star-history project:

- Source: https://github.com/star-history/star-history
- Pinned commit: `fb8e1078c9e48c612f830f2cb6c36e046a6697d5`
- License: MIT (see `vendor/LICENSE`)

`render.ts` reproduces the SVG generation flow from that project's
`backend/main.ts`, and the `fixJsdomSvgCasing` helper is copied from its
`backend/utils.ts`.

## Local changes to the vendored copy

The vendored tree is not a verbatim copy. We made two changes:

1. **Removed unused files** that are not reachable from the renderer's entry
   points (`packages/xy-chart` and `common/chart`): `common/repo-data.ts`,
   `types/gh.ts`, `packages/card-landscape1.tsx`, and `packages/radar-svg.ts`.
   These belong to the OG-card feature, which this action does not use.

2. **Blanked `packages/utils/fontData.ts`.** Upstream embeds a ~53 KB base64
   "xkcd" web font there. That font is licensed separately from star-history's
   MIT code, and GitHub strips `@font-face` from SVGs served via `<img>`, so it
   never renders in a README anyway. `render.ts` removes the `<style>` block
   before writing the SVG, so the font is unused; we blank the data to avoid
   redistributing the font.

All other files under `vendor/shared/` are copied unchanged from the pinned
commit.

To update the vendored code, re-copy `shared/` from a newer star-history commit,
re-apply the two changes above, and bump the pinned commit.
