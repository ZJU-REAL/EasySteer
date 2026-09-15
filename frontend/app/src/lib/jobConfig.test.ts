import { describe, expect, it } from "vitest";
import { trainingApply } from "./jobConfig";

describe("training selection", () => {
  it("defaults to only the last prompt token", () => {
    const apply = trainingApply();
    expect(apply.prompt_positions).toEqual([-1]);
    expect(apply.prompt).toBeNull();
    expect(apply.generation).toBeNull();
  });

  it("preserves a supplied selection without adding the last prompt token", () => {
    const input = {
      generation_window: [0, 5] as [number, number],
      exclude_generation_tokens: [42],
    };
    const form = trainingApply(input);
    const submitted = trainingApply(form);
    const exported = trainingApply(submitted);
    expect(exported.generation_window).toEqual([0, 5]);
    expect(exported.exclude_generation_tokens).toEqual([42]);
    expect(exported.prompt_positions).toBeNull();
    expect(exported.generation).toBeNull();
    form.generation_window![1] = 2;
    expect(input.generation_window).toEqual([0, 5]);
    expect(submitted.generation_window).toEqual([0, 5]);
    expect(exported.generation_window).toEqual([0, 5]);
  });
});
