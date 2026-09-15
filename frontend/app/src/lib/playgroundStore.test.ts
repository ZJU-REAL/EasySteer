import { afterEach, expect, it } from "vitest";
import { galleryEntries } from "../data/gallery";
import { loadCustomSpec, loadGalleryEntry, playground, resetPlayground } from "./playgroundStore";
import { renderPrompt } from "./prompts";
import { defaultSteeringSpec } from "./spec";

afterEach(resetPlayground);

it("keeps the native checkpoint's model and template until replaced", () => {
  loadCustomSpec(defaultSteeringSpec(), { model: "trained-model", promptTemplate: "Input: %s\nOutput:" });
  expect(playground.presetModel).toBe("trained-model");
  expect(playground.promptTemplate).toBe("Input: %s\nOutput:");
  loadGalleryEntry(galleryEntries.find((entry) => entry.id === "lm_steer")!);
  expect(playground.promptTemplate).toBe("%s");
  loadCustomSpec(defaultSteeringSpec());
  expect(playground.presetModel).toBe("");
  expect(playground.promptTemplate).toBeNull();
});

it("matches Python template substitution without substituting instruction content", () => {
  expect(renderPrompt("100%% complete\n%s\nAnswer:", "literal %s and $&")).toBe(
    "100% complete\nliteral %s and $&\nAnswer:",
  );
  expect(() => renderPrompt("missing placeholder", "hello")).toThrow("one %s");
});
