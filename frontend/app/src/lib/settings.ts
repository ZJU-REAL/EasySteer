/**
 * Global app settings, persisted to localStorage.
 *
 * - `openaiBaseUrl`: base URL of the vllm-steer OpenAI-compatible server
 *   (text generation). User-configurable; defaults to localhost:8000/v1.
 * - `flaskBaseUrl`: base URL of the Flask job backend (extraction /
 *   training / SAE). Empty string = same origin, which both the dev
 *   proxy and the combined deployment serve; there is no form field for
 *   it — a different host can still be set via localStorage.
 * - `model`: model id sent on chat.completions requests.
 */

import { reactive, watch } from "vue";

const STORAGE_KEY = "easysteer.settings.v1";

export interface Settings {
  openaiBaseUrl: string;
  flaskBaseUrl: string;
  model: string;
  language: "en" | "zh";
  theme: "dark" | "light";
  temperature: number;
  maxTokens: number;
  neuronpediaApiKey: string;
}

const defaults: Settings = {
  openaiBaseUrl: "http://localhost:8000/v1",
  flaskBaseUrl: "",
  model: "",
  language: "en",
  // Light by default; a persisted user choice (stored settings) wins.
  theme: "light",
  temperature: 0,
  maxTokens: 256,
  neuronpediaApiKey: "",
};

function load(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...defaults };
    const parsed = JSON.parse(raw) as Partial<Settings>;
    return { ...defaults, ...parsed };
  } catch {
    // Corrupt localStorage entry: fall back to defaults (worst case the
    // user re-enters a URL; nothing else depends on this value).
    return { ...defaults };
  }
}

export const settings = reactive<Settings>(load());

watch(
  settings,
  (value) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
    document.documentElement.dataset.theme = value.theme;
  },
  { deep: true },
);

document.documentElement.dataset.theme = settings.theme;
