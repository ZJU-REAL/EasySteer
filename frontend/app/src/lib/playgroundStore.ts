/**
 * Shared playground state: the spec under edit, the prompt, and the
 * gallery preset it came from (if any). Lives outside the component tree
 * so the gallery can pre-fill it before navigation.
 */

import { reactive } from "vue";
import type { GalleryEntry } from "../data/gallery";
import type { SteeringSpec } from "./spec";
import { defaultSteeringSpec, specFromJson } from "./spec";

export interface PlaygroundState {
  spec: SteeringSpec;
  prompt: string;
  systemPrompt: string;
  presetId: string | null;
  presetModel: string;
  /**
   * Bumped whenever `spec` is replaced wholesale (JSON edit, gallery
   * load); form editors key on it to re-seed their local text state.
   */
  revision: number;
}

export const playground = reactive<PlaygroundState>({
  spec: defaultSteeringSpec(),
  prompt: "",
  systemPrompt: "",
  presetId: null,
  presetModel: "",
  revision: 0,
});

export function replaceSpec(spec: SteeringSpec): void {
  playground.spec = spec;
  playground.revision += 1;
}

export function loadGalleryEntry(entry: GalleryEntry): void {
  replaceSpec(specFromJson(entry.spec));
  playground.prompt = entry.prompt;
  playground.systemPrompt = entry.systemPrompt ?? "";
  playground.presetId = entry.id;
  playground.presetModel = entry.model;
}

export function resetPlayground(): void {
  replaceSpec(defaultSteeringSpec());
  playground.prompt = "";
  playground.systemPrompt = "";
  playground.presetId = null;
  playground.presetModel = "";
}
