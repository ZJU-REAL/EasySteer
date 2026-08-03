// MODIFIED FROM UPSTREAM (see ../../../NOTICE.md).
//
// Upstream star-history ships a ~53 KB base64 "xkcd" web font here. That font
// carries its own license, separate from star-history's MIT, and GitHub strips
// @font-face from SVGs served via <img>, so it never renders in a README.
// render.ts removes the <style>/@font-face block before writing the SVG, so the
// font URL is never used. We blank it here to avoid redistributing the font.
export const xkcdFontUrl = "";
