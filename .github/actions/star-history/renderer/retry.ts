/**
 * Retry interceptor for the GitHub API calls the vendored star-history code
 * makes through axios.
 *
 * Why this exists: the stargazers endpoints are fetched with the automatic
 * Actions token, whose rate limit is per-repo (1000 req/hr) and which also
 * trips GitHub's secondary/abuse limit on request bursts. Both surface as an
 * HTTP 403 that clears on its own after a short wait. Without a retry a single
 * transient 403 fails the whole scheduled run (see the 403 "rate limit
 * exceeded" failures on consumer repos). The vendored api.tsx uses the default
 * axios singleton, so installing a response interceptor on it adds retries
 * without editing vendored code.
 *
 * Retries: network errors, 429, 5xx, and rate-limit 403s (identified by a
 * Retry-After header, x-ratelimit-remaining: 0, or a rate-limit body). A 403
 * that is a genuine access/permission error has none of those and is not
 * retried. Wait time honors Retry-After / x-ratelimit-reset when present,
 * otherwise exponential backoff with jitter, capped so a run cannot hang.
 */
import type { AxiosError, AxiosInstance, AxiosResponse } from "axios";

const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);
const DEFAULT_MAX_RETRIES = 4;
// Never wait longer than this for one attempt. A primary-limit reset can be up
// to an hour out; waiting that long inside a 10s job is pointless, so cap and
// let the job fail fast rather than hang.
const MAX_WAIT_MS = 60_000;

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** A 403 that is really a rate limit (vs a permission/access 403). */
export function isRateLimit403(res: Pick<AxiosResponse, "status" | "headers" | "data"> | undefined): boolean {
  if (!res || res.status !== 403) return false;
  const h = (res.headers || {}) as Record<string, unknown>;
  if (h["retry-after"] != null) return true;
  if (String(h["x-ratelimit-remaining"]) === "0") return true;
  const body = typeof res.data === "string" ? res.data : JSON.stringify(res.data ?? "");
  return /rate limit|secondary rate|abuse/i.test(body);
}

/**
 * Milliseconds to wait before the next attempt, or null when waiting would not
 * help (a rate-limit reset further out than the cap, so the limit will not
 * clear inside this job).
 */
export function computeWaitMs(
  res: Pick<AxiosResponse, "headers"> | undefined,
  attempt: number,
  now: number = Date.now(),
): number | null {
  const h = (res?.headers || {}) as Record<string, unknown>;

  const retryAfter = h["retry-after"];
  if (retryAfter != null) {
    const secs = Number(retryAfter);
    if (Number.isFinite(secs) && secs >= 0) {
      return secs * 1000 > MAX_WAIT_MS ? null : secs * 1000;
    }
  }

  const remaining = String(h["x-ratelimit-remaining"] ?? "");
  const reset = Number(h["x-ratelimit-reset"] ?? "");
  if (remaining === "0" && Number.isFinite(reset) && reset > 0) {
    const waitMs = reset * 1000 - now;
    if (waitMs > MAX_WAIT_MS) return null; // limit resets too far out to wait
    if (waitMs > 0) return waitMs;
  }

  // Exponential backoff with jitter: 1s, 2s, 4s, 8s (+ up to 500ms).
  const base = 1000 * 2 ** (attempt - 1);
  const jitter = Math.floor(Math.random() * 500);
  return Math.min(base + jitter, MAX_WAIT_MS);
}

export interface RetryOptions {
  maxRetries?: number;
  onRetry?: (info: { attempt: number; status?: number; waitMs: number; url?: string }) => void;
  sleepFn?: (ms: number) => Promise<void>;
  now?: () => number;
}

/** Install the retry interceptor on an axios instance. */
export function installRetry(client: AxiosInstance, opts: RetryOptions = {}): void {
  const max = opts.maxRetries ?? DEFAULT_MAX_RETRIES;
  const sleepFn = opts.sleepFn ?? sleep;
  const now = opts.now ?? Date.now;

  client.interceptors.response.use(undefined, async (error: AxiosError) => {
    const config = error.config as (typeof error.config & { __retryCount?: number }) | undefined;
    if (!config) throw error;

    const res = error.response;
    const status = res?.status;
    const retryable = !res || RETRYABLE_STATUS.has(status as number) || isRateLimit403(res);
    const attempt = (config.__retryCount ?? 0) + 1;
    if (!retryable || attempt > max) throw error;

    const waitMs = computeWaitMs(res, attempt, now());
    if (waitMs == null) throw error; // waiting would not clear the limit in time

    config.__retryCount = attempt;
    opts.onRetry?.({ attempt, status, waitMs, url: config.url });
    await sleepFn(waitMs);
    return client(config);
  });
}
