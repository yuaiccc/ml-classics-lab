import { describe, it, expect } from "vitest";
import { gateScores, topKExperts, routedExperts, activeFraction } from "./moe-fns";

describe("gating", () => {
  it("gateScores returns one score per expert in [0,1]", () => {
    const s = gateScores(3, 8);
    expect(s).toHaveLength(8);
    expect(Math.min(...s)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...s)).toBeLessThanOrEqual(1);
  });
  it("topKExperts returns k distinct indices, highest scores first", () => {
    const top = topKExperts([0.1, 0.9, 0.5, 0.3], 2);
    expect(top).toEqual([1, 2]);
    expect(new Set(top).size).toBe(2);
  });
});

describe("routing", () => {
  it("dense routes to ALL experts", () => {
    expect(routedExperts("dense", 5, 8, 2)).toHaveLength(8);
  });
  it("switch routes to exactly 1 expert", () => {
    expect(routedExperts("switch", 5, 8, 2)).toHaveLength(1);
  });
  it("moe routes to topK experts", () => {
    expect(routedExperts("moe", 5, 8, 2)).toHaveLength(2);
  });
});

describe("active fraction (sparsity)", () => {
  it("dense = 1 (all experts active)", () => {
    expect(activeFraction("dense", 8, 2, 1)).toBe(1);
  });
  it("switch = 1/nExperts", () => {
    expect(activeFraction("switch", 8, 2, 1)).toBeCloseTo(1 / 8, 9);
  });
  it("moe = topK/nExperts", () => {
    expect(activeFraction("moe", 8, 2, 1)).toBeCloseTo(2 / 8, 9);
  });
  it("deepseek includes shared experts: > pure topK/nExperts", () => {
    expect(activeFraction("deepseek", 8, 2, 2)).toBeGreaterThan(2 / 8);
  });
});
