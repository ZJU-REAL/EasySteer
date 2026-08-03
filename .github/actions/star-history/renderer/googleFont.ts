// renderer/googleFont.ts
// Fetch a Google Fonts family's TTF files at runtime for PNG rasterization.
//
// The Google Fonts CSS2 API serves different formats by request User-Agent:
// modern browsers get woff2, older clients get plain TTF. resvg only reads
// TTF/OTF, so we send a fixed legacy User-Agent (Windows XP, which does not
// advertise woff2 support) to force .ttf URLs. Pinning it means a future
// Node/undici default-UA change cannot silently switch the response to woff2
// and break the feature for everyone.
//
// Downloaded files are written into destDir and their paths returned along with
// the font's real internal family name, read from the TTF `name` table. The
// caller (render.ts) passes that name to resvg as defaultFontFamily, so the
// requested font actually matches a loaded face instead of relying on the CSS
// family string equalling the compiled font's internal name. Any failure throws
// so the caller can fall back to a bundled font.
import { writeFileSync } from "node:fs";
import { join } from "node:path";

const MAX_FONT_BYTES = 20 * 1024 * 1024; // 20 MB per file; large CJK fonts need the headroom.
const FETCH_TIMEOUT_MS = 10000;
// Windows XP UA: Google serves TTF (not woff2) to clients that do not advertise
// woff2 support. Pinned so the format contract does not depend on undici's default.
const GOOGLE_FONTS_UA = "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36";

// Recognised sfnt magic numbers (first 4 bytes) for a real font file. Used to
// reject a 200 response that is not actually a font (an error page, a CDN
// glitch) before it reaches resvg's native parser.
function looksLikeFont(buf: Buffer): boolean {
  if (buf.length < 4) return false;
  const m = buf.readUInt32BE(0);
  return (
    m === 0x00010000 || // TrueType outlines
    m === 0x4f54544f || // 'OTTO' (CFF / OpenType)
    m === 0x74727565 || // 'true'
    m === 0x74746366 // 'ttcf' (font collection)
  );
}

function decodeUtf16BE(buf: Buffer, start: number, len: number): string {
  let s = "";
  for (let i = 0; i + 1 < len; i += 2) s += String.fromCharCode(buf.readUInt16BE(start + i));
  return s;
}

// Read the family name from a TTF `name` table so it matches what resvg looks
// up. Prefers the typographic family (nameID 16), then the classic family
// (nameID 1), and prefers a Windows-platform record (UTF-16BE). Returns null if
// the table cannot be parsed, so the caller can fall back to the requested name.
function readFamilyName(buf: Buffer): string | null {
  try {
    const numTables = buf.readUInt16BE(4);
    let nameOff = -1;
    for (let i = 0; i < numTables; i++) {
      const rec = 12 + i * 16;
      if (buf.toString("latin1", rec, rec + 4) === "name") {
        nameOff = buf.readUInt32BE(rec + 8);
        break;
      }
    }
    if (nameOff < 0) return null;

    const count = buf.readUInt16BE(nameOff + 2);
    const storage = nameOff + buf.readUInt16BE(nameOff + 4);
    let family: string | null = null;
    let typographic: string | null = null;
    for (let i = 0; i < count; i++) {
      const rec = nameOff + 6 + i * 12;
      const platformID = buf.readUInt16BE(rec);
      const nameID = buf.readUInt16BE(rec + 6);
      if (nameID !== 1 && nameID !== 16) continue;
      const len = buf.readUInt16BE(rec + 8);
      const off = storage + buf.readUInt16BE(rec + 10);
      const val =
        platformID === 3 || platformID === 0
          ? decodeUtf16BE(buf, off, len)
          : buf.toString("latin1", off, off + len);
      if (!val) continue;
      // Prefer a Windows record; only overwrite an existing value with one.
      if (nameID === 16 && (typographic === null || platformID === 3)) typographic = val;
      if (nameID === 1 && (family === null || platformID === 3)) family = val;
    }
    return typographic || family;
  } catch {
    return null;
  }
}

export async function fetchGoogleFontFiles(
  family: string,
  destDir: string,
  weights: number[] = [400, 700]
): Promise<{ files: string[]; family: string }> {
  const name = family.trim();
  if (!name) throw new Error("empty font family");

  // Google Fonts wants spaces as "+"; encode the rest normally.
  const enc = encodeURIComponent(name).replace(/%20/g, "+");
  const url = `https://fonts.googleapis.com/css2?family=${enc}:wght@${weights.join(";")}`;

  const res = await fetch(url, {
    headers: { "User-Agent": GOOGLE_FONTS_UA },
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  });
  // Google returns HTTP 400 for an unknown family.
  if (!res.ok) throw new Error(`Google Fonts CSS HTTP ${res.status} for "${name}"`);
  const css = await res.text();

  // Pull every truetype url out of the @font-face blocks.
  const urls = Array.from(css.matchAll(/url\((https:\/\/[^)]+\.ttf)\)/g)).map((m) => m[1]);
  if (urls.length === 0) throw new Error(`no ttf url in Google Fonts CSS for "${name}"`);

  const files: string[] = [];
  let resolvedFamily: string | null = null;
  for (let i = 0; i < urls.length; i++) {
    const fres = await fetch(urls[i], {
      headers: { "User-Agent": GOOGLE_FONTS_UA },
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    });
    if (!fres.ok) throw new Error(`font download HTTP ${fres.status}: ${urls[i]}`);

    // Reject an oversized response by its declared length before buffering it,
    // then enforce the cap on the actual bytes.
    const declared = Number(fres.headers.get("content-length") || "0");
    if (declared > MAX_FONT_BYTES) throw new Error(`font too large (${declared} bytes)`);
    const buf = Buffer.from(await fres.arrayBuffer());
    if (buf.length > MAX_FONT_BYTES) throw new Error(`font too large (${buf.length} bytes)`);
    // Only write a real font, so a wrong/hostile 200 cannot reach resvg's parser.
    if (!looksLikeFont(buf)) throw new Error(`downloaded file is not a font: ${urls[i]}`);

    const p = join(destDir, `font-${i}.ttf`);
    writeFileSync(p, buf);
    files.push(p);
    if (resolvedFamily === null) resolvedFamily = readFamilyName(buf);
  }
  return { files, family: resolvedFamily || name };
}
