import { describe, expect, it } from "vitest";
import { specFromJson, validateSteeringSpec } from "../lib/spec";
import { galleryEntries } from "./gallery";

describe("gallery entries", () => {
  it("covers all 15 replication notebooks", () => {
    expect(galleryEntries.length).toBe(15);
    const ids = galleryEntries.map((e) => e.id);
    expect(new Set(ids).size).toBe(15);
  });

  it("every entry parses into a valid SteeringSpec", () => {
    for (const entry of galleryEntries) {
      const spec = specFromJson(entry.spec);
      const issues = validateSteeringSpec(spec);
      expect(issues, `${entry.id}: ${issues.map((i) => `${i.path}: ${i.message}`).join("; ")}`).toEqual([]);
    }
  });

  it("every entry has bilingual descriptions, a prompt and a paper link", () => {
    for (const entry of galleryEntries) {
      expect(entry.tagline.en.length, entry.id).toBeGreaterThan(10);
      expect(entry.tagline.zh.length, entry.id).toBeGreaterThan(5);
      expect(entry.description.en.length, entry.id).toBeGreaterThan(50);
      expect(entry.description.zh.length, entry.id).toBeGreaterThan(20);
      expect(entry.prompt.length, entry.id).toBeGreaterThan(0);
      expect(entry.paper.url, entry.id).toMatch(/^https?:\/\//);
      expect(entry.model.length, entry.id).toBeGreaterThan(0);
    }
  });

  it("keeps the notebook-critical spec details", () => {
    const byId = Object.fromEntries(galleryEntries.map((e) => [e.id, specFromJson(e.spec)]));

    // refusal_direction: 4 vectors, sequential, position -k per vector.
    const refusal = byId["refusal_direction"];
    expect(refusal.conflict).toBe("sequential");
    expect(refusal.vectors.length).toBe(4);
    refusal.vectors.forEach((v, i) => {
      expect(v.apply.positions).toEqual([-(i + 1)]);
      expect(v.scale).toBe(2.0);
    });

    // seal: generation-phase token-triggered multi-vector.
    const seal = byId["seal"];
    expect(seal.vectors.map((v) => v.scale)).toEqual([0.5, -0.5, -0.5]);
    expect(seal.vectors.every((v) => v.apply.phases.join() === "generation")).toBe(true);

    // steermoe: single moe_router vector from a config file.
    const steermoe = byId["steermoe"];
    expect(steermoe.vectors[0].algorithm).toBe("moe_router");
    expect(steermoe.vectors[0].source).toMatch(/\.json$/);

    // lm_steer: epsilon * steer-value scale on the final layer.
    const lmSteer = byId["lm_steer"];
    expect(lmSteer.vectors[0].scale).toBeCloseTo(0.002);
    expect(lmSteer.vectors[0].layers).toEqual([11]);
  });
});
