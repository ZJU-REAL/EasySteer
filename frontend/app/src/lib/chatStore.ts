/**
 * Chat state: conversation history and steering mode, kept outside the
 * component tree so the conversation survives navigation.
 */

import { reactive } from "vue";

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  /** True when the assistant turn was generated with steering applied. */
  steered?: boolean;
  /** Unsteered reply, present when the turn ran in compare mode. */
  baseline?: string;
}

export type ChatSteeringMode = "none" | "playground" | "custom";

export interface ChatState {
  turns: ChatTurn[];
  steeringMode: ChatSteeringMode;
  customSpecText: string;
  /** Answer each message twice: steered and baseline, side by side. */
  compare: boolean;
}

export const chat = reactive<ChatState>({
  turns: [],
  steeringMode: "none",
  customSpecText: "",
  // On by default: seeing the steered reply next to the unsteered one is
  // the point of the page (the checkbox greys out while steering is off).
  compare: true,
});

export function clearChat(): void {
  chat.turns = [];
}
