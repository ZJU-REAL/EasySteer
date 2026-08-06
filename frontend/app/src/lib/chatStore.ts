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
}

export type ChatSteeringMode = "none" | "playground" | "custom";

export interface ChatState {
  turns: ChatTurn[];
  systemPrompt: string;
  steeringMode: ChatSteeringMode;
  customSpecText: string;
}

export const chat = reactive<ChatState>({
  turns: [],
  systemPrompt: "",
  steeringMode: "none",
  customSpecText: "",
});

export function clearChat(): void {
  chat.turns = [];
}
