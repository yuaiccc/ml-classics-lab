import { describe, it, expect } from "vitest";
import { attentionMemory, recurrentMemory, scanStep, scanNorms } from "./linear-seq-fns";

describe("memory footprint vs sequence length", () => {
  it("attention KV-cache grows linearly with L", () => {
    expect(attentionMemory(10)).toBe(10);
    expect(attentionMemory(100)).toBeGreaterThan(attentionMemory(10));
  });
  it("recurrent state is constant regardless of L", () => {
    expect(recurrentMemory(16)).toBe(16);
    expect(recurrentMemory(16)).toBe(recurrentMemory(16));
  });
});

describe("gated linear recurrence", () => {
  it("scanStep: s' = decay*s + gate*x (elementwise)", () => {
    expect(scanStep([1, 2], [1, 1], 0.5, 1)).toEqual([1.5, 2]);
  });
  it("scanNorms returns one bounded norm per token", () => {
    const norms = scanNorms([1, 5, 2, 8, 3], 8);
    expect(norms).toHaveLength(5);
    expect(norms.every((n) => Number.isFinite(n) && n < 100)).toBe(true);
  });
});
