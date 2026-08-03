/**
 * Render a star history chart to an SVG file using star-history's own code.
 *
 * This mirrors the SVG generation in star-history's backend/main.ts: build a
 * JSDOM svg element, fetch stargazer data with getRepoData, draw it with the
 * XYChart renderer in node mode, then optimize with svgo. No browser, no
 * third-party CLI. The vendored star-history source lives under vendor/shared
 * (see vendor/LICENSE and NOTICE.md for attribution).
 *
 * The GitHub token is read from the GITHUB_TOKEN environment variable (kept off
 * the command line so it never appears in a process listing).
 *
 * Usage:
 *   GITHUB_TOKEN=<t> tsx render.ts --repos owner/repo[,owner/repo2] \
 *     --theme light|dark --type Date|Timeline --width <px> --output <file> \
 *     [--png <file>] [--signature <file>]
 */
import axios from "axios";
import { JSDOM } from "jsdom";
import { optimize } from "svgo";
import { Resvg } from "@resvg/resvg-js";
import XYChart from "./vendor/shared/packages/xy-chart";
import { convertDataToChartData, getRepoData } from "./vendor/shared/common/chart";
import { writeFileSync, mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { fetchGoogleFontFiles } from "./googleFont";
import { installRetry } from "./retry";

// Comic Neue (SIL OFL) is vendored under fonts/ and pinned for PNG rasterization
// so the PNG renders the same handwriting-style text on every machine, instead
// of resvg substituting whatever system font the build host happens to have.
// The chart SVG asks for font-family "xkcd" (that font is stripped and its data
// blanked, see fontData.ts); resvg maps the missing family to this default.
const PNG_FONT_FILES = [
  fileURLToPath(new URL("./fonts/ComicNeue-Regular.ttf", import.meta.url)),
  fileURLToPath(new URL("./fonts/ComicNeue-Bold.ttf", import.meta.url)),
];
const PNG_FONT_FAMILY = "Comic Neue";

// star-history fetches at most this many pages of stargazers per repo.
const MAX_REQUEST_AMOUNT = 16;

// JSDOM lowercases camelCase SVG names; restore the ones D3's filter emits.
// Copied from star-history backend/utils.ts.
function fixJsdomSvgCasing(svgContent: string): string {
  return svgContent
    .replace(/feturbulence/g, "feTurbulence")
    .replace(/fedisplacementmap/g, "feDisplacementMap")
    .replace(/filterunits/g, "filterUnits")
    .replace(/basefrequency/g, "baseFrequency")
    .replace(/xchannelselector/g, "xChannelSelector")
    .replace(/ychannelselector/g, "yChannelSelector");
}

// The chart draws the repo/owner logo as <image href="https://avatars...">.
// GitHub sanitizes committed SVGs and blocks external image refs, so those show
// as broken-image boxes. Inline each external image as a base64 data URL, the
// same thing star-history's own browser export does before saving.
const MAX_IMAGE_BYTES = 2 * 1024 * 1024; // 2 MB cap per inlined image.

async function inlineExternalImages(svg: SVGSVGElement): Promise<void> {
  const images = Array.from(svg.querySelectorAll("image"));
  await Promise.all(
    images.map(async (img) => {
      const href = img.getAttribute("href") || img.getAttribute("xlink:href");
      if (!href || !/^https?:\/\//i.test(href)) return;
      try {
        const res = await fetch(href, {
          signal: AbortSignal.timeout(10000),
          redirect: "follow",
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        // Only inline actual images, so a hostile or wrong endpoint cannot embed
        // arbitrary content into the committed SVG.
        const type = res.headers.get("content-type") || "";
        if (!/^image\//i.test(type)) {
          throw new Error(`unexpected content-type "${type || "none"}"`);
        }

        // Reject oversized responses. Check the advertised length first, then
        // enforce the cap on the actual bytes.
        const declared = Number(res.headers.get("content-length") || "0");
        if (declared > MAX_IMAGE_BYTES) {
          throw new Error(`image too large (${declared} bytes)`);
        }
        const buf = Buffer.from(await res.arrayBuffer());
        if (buf.length > MAX_IMAGE_BYTES) {
          throw new Error(`image too large (${buf.length} bytes)`);
        }

        img.setAttribute("href", `data:${type};base64,${buf.toString("base64")}`);
      } catch (e) {
        // Drop an unreachable or unacceptable logo rather than leaving a broken
        // external ref (which GitHub would show as a broken-image box).
        img.remove();
        process.stderr.write(`Inlined image skipped (${href}): ${e}\n`);
      }
    })
  );
}

function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const next = argv[i + 1];
    // Only consume the next token as a value if it is not itself a flag. This
    // avoids "--signature --output x" silently setting signature to "--output".
    if (next !== undefined && !next.startsWith("--")) {
      out[a.slice(2)] = next;
      i++;
    } else {
      out[a.slice(2)] = "";
    }
  }
  return out;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const repos = (args.repos || "").split(",").map((r) => r.trim()).filter(Boolean);
  // Token is env-only, never accepted on argv, so it cannot leak into a process
  // listing or CI logs.
  const token = process.env.GITHUB_TOKEN || "";
  const theme = args.theme === "dark" ? "dark" : "light";
  const type = args.type === "Timeline" ? "Timeline" : "Date";
  const width = Number(args.width) || 800;
  const output = args.output;

  if (repos.length === 0) throw new Error("--repos is required");
  if (!output) throw new Error("--output is required");
  if (!token) throw new Error("GITHUB_TOKEN environment variable is required");

  // The vendored star-history code calls the GitHub stargazers API through the
  // default axios instance. Add retry-with-backoff so a transient rate-limit
  // 403 (per-repo hourly limit or secondary/abuse limit on the Actions token)
  // clears on its own instead of failing the whole run.
  installRetry(axios, {
    onRetry: ({ attempt, status, waitMs, url }) => {
      process.stderr.write(
        `Retry ${attempt} after HTTP ${status ?? "network error"} in ${Math.round(waitMs)}ms: ${url ?? ""}\n`,
      );
    },
  });

  const repoData = await getRepoData(repos, token, MAX_REQUEST_AMOUNT);

  // A signature over just the star data (dates + counts, not the rendered
  // pixels). The SVG has sub-pixel float jitter between runs, so comparing SVGs
  // would always look "changed"; the data hash only moves when stars actually
  // change or the day rolls over. The action uses it to decide whether to commit.
  if (args.signature) {
    // Normalize each record's date to day granularity. star-history stamps the
    // final "now" point with a full timestamp (seconds), which would otherwise
    // change the signature on every run. Truncating to the day makes it stable
    // within a day and only change on real star movement or a day rollover.
    // Include the requested font so changing only the font (no star movement,
    // same day) still invalidates the cache and forces a PNG re-render.
    const sigFont = (args["font-family"] || "").trim();
    const sigInput = JSON.stringify({
      font: sigFont,
      repos: repoData.map((r: any) => ({
        repo: r.repo,
        records: r.starRecords.map((x: any) => ({
          d: String(x.date).split(" ")[0],
          c: x.count,
        })),
      })),
    });
    const sig = createHash("sha256").update(sigInput).digest("hex");
    mkdirSync(dirname(args.signature), { recursive: true });
    writeFileSync(args.signature, sig, "utf-8");
  }

  const dom = new JSDOM(`<!DOCTYPE html><body></body>`);
  const body = dom.window.document.querySelector("body")!;
  const svg = dom.window.document.createElement("svg") as unknown as SVGSVGElement;
  body.append(svg);
  svg.setAttribute("width", `${width}`);
  svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");

  XYChart(
    svg,
    {
      title: "Star History",
      xLabel: type === "Date" ? "Date" : "Timeline",
      yLabel: "GitHub Stars",
      data: convertDataToChartData(repoData, type, { insertZeroPoint: true }),
      showDots: false,
      transparent: false,
      theme,
    },
    {
      xTickLabelType: type === "Date" ? "Date" : "Number",
      chartWidth: width,
    }
  );

  await inlineExternalImages(svg);

  // Remove the embedded @font-face <style>. GitHub strips <style>/@font-face
  // when an SVG is served via <img>, so the bundled font never renders in a
  // README anyway; dropping it cuts ~53 KB per file and avoids shipping a font
  // whose license differs from star-history's MIT.
  svg.querySelectorAll("style").forEach((el) => el.remove());
  // star-history's own image export strips browser-only nodes; do the same.
  // These are also the only Math.random()-driven elements, so removing them
  // keeps the output deterministic.
  svg.querySelectorAll(".browser-only").forEach((el) => el.remove());

  const svgContent = fixJsdomSvgCasing(svg.outerHTML);
  const optimized = optimize(svgContent, { multipass: true }).data;

  mkdirSync(dirname(output), { recursive: true });
  writeFileSync(output, optimized, "utf-8");
  process.stderr.write(`Wrote ${output} (${optimized.length} bytes)\n`);

  // Optional PNG rasterization for consumers that cannot render SVG (npm,
  // pub.dev). raw.githubusercontent serves .svg as text/plain (will not render
  // off-GitHub) and pub.dev's sanitizer drops <picture>; a PNG sidesteps both.
  if (args.png) {
    // resvg panics on the feTurbulence/feDisplacementMap filter star-history
    // uses for its hand-sketched look (assertion failure in its filter code).
    // The filter is purely decorative, so strip it before rasterizing. The SVG
    // file was already written above with the filter intact, so only the PNG
    // loses it. Re-serialize + re-optimize the now-filterless DOM.
    svg.querySelectorAll("filter").forEach((el) => el.remove());
    svg.querySelectorAll("[filter]").forEach((el) => el.removeAttribute("filter"));
    const pngSvg = optimize(fixJsdomSvgCasing(svg.outerHTML), { multipass: true }).data;
    // Default to the vendored Comic Neue. If the user asked for a Google font,
    // download it and use that instead; on any failure keep the vendored font
    // so a network hiccup never fails the run.
    let fontFiles = PNG_FONT_FILES;
    let fontFamily = PNG_FONT_FAMILY;
    let fontDir: string | null = null;
    const requestedFont = (args["font-family"] || "").trim();
    if (requestedFont) {
      try {
        fontDir = mkdtempSync(join(tmpdir(), "sh-font-"));
        // fetched.family is the font's real internal name, read from its name
        // table, so resvg's defaultFontFamily lookup matches the downloaded face
        // instead of assuming the Google family string equals the compiled name.
        const fetched = await fetchGoogleFontFiles(requestedFont, fontDir);
        fontFiles = fetched.files;
        fontFamily = fetched.family;
        process.stderr.write(
          `Using Google font "${requestedFont}" as "${fetched.family}" (${fetched.files.length} file(s))\n`
        );
      } catch (e) {
        process.stderr.write(
          `Google font "${requestedFont}" unavailable (${e}); falling back to ${PNG_FONT_FAMILY}\n`
        );
      }
    }
    const resvg = new Resvg(pngSvg, {
      // Chart already draws an opaque background (transparent:false), but set an
      // explicit background so the PNG never ends up with an alpha fringe.
      background: theme === "dark" ? "#0d1117" : "#ffffff",
      // The embedded @font-face <style> was stripped above. Pin the vendored
      // Comic Neue font (loadSystemFonts:false) so the PNG is byte-stable across
      // build machines; the chart's requested "xkcd" family is unavailable, so
      // resvg falls back to defaultFontFamily below.
      font: {
        loadSystemFonts: false,
        fontFiles,
        defaultFontFamily: fontFamily,
      },
      // Render at 2x the logical width for a crisp result on hi-dpi displays.
      fitTo: { mode: "width", value: width * 2 },
    });
    const pngBuf = resvg.render().asPng();
    mkdirSync(dirname(args.png), { recursive: true });
    writeFileSync(args.png, pngBuf);
    process.stderr.write(`Wrote ${args.png} (${pngBuf.length} bytes)\n`);

    // Remove the downloaded-font temp dir so repeated runs on a long-lived
    // (self-hosted) runner do not accumulate orphaned dirs in the OS tmpdir.
    if (fontDir) rmSync(fontDir, { recursive: true, force: true });
  }
}

// Exit code the shell wrapper reads to tell a transient rate-limit / access 403
// apart from a real render failure. 75 is the conventional EX_TEMPFAIL; on it,
// render-charts.sh keeps the last committed chart instead of failing the run.
const EX_RATE_LIMITED = 75;

main().catch((err) => {
  const msg = err?.message || String(err);
  const status = err?.status ? ` [status ${err.status}]` : "";
  process.stderr.write(`Render failed${status}: ${msg}\n`);
  // star-history maps every HTTP 403 to "rate limit exceeded", but under the
  // stargazers restriction a 403 usually means the token cannot read this
  // repo's stargazers. Point users at the real fix.
  if (err?.status === 403 || err?.status === 401) {
    process.stderr.write(
      "Hint: this is often an access problem, not a rate limit. The token may " +
        "lack stargazers access for this repo. Charting a repo you do not own " +
        "needs a personal access token that can read it (public_repo scope).\n"
    );
    // 403 (rate limit or access) and 401 (bad/expired token) are transient from
    // the workflow's point of view: the automatic Actions token is capped at
    // 1000 requests/hour per repo and a burst of runs can exhaust it, in which
    // case a later run should keep the existing chart rather than fail red.
    process.exit(EX_RATE_LIMITED);
  }
  process.exit(1);
});
