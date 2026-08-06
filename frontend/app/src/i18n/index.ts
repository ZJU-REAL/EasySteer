/**
 * Minimal i18n composable (en / zh). Messages live in per-domain modules
 * under `./messages/`; Chinese strings reuse the wording of the legacy
 * frontend where the concept carried over. All keys and code identifiers
 * are English by project rule.
 */

import { computed } from "vue";
import { settings } from "../lib/settings";
import { chatMessages } from "./messages/chat";
import { commonMessages } from "./messages/common";
import { galleryMessages } from "./messages/gallery";
import { homeMessages } from "./messages/home";
import { playgroundMessages } from "./messages/playground";
import { saeMessages } from "./messages/sae";
import { workshopMessages } from "./messages/workshop";

export const messages = {
  ...commonMessages,
  ...homeMessages,
  ...chatMessages,
  ...galleryMessages,
  ...playgroundMessages,
  ...workshopMessages,
  ...saeMessages,
};

export type MessageKey = keyof typeof messages;

function translate(
  key: MessageKey,
  params?: Record<string, string | number>,
): string {
  const entry = messages[key];
  if (!entry) return String(key);
  let text = entry[settings.language];
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      text = text.replace(`{${k}}`, String(v));
    }
  }
  return text;
}

export function useI18n() {
  const language = computed({
    get: () => settings.language,
    set: (value: "en" | "zh") => {
      settings.language = value;
    },
  });
  return { t: translate, language };
}
