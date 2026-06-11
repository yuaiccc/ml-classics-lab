import { describe, it, expect } from "vitest";
import { attends, attendedFraction } from "./sparse-attention-fns";

const L = 12;

describe("causality", () => {
  it("no variant attends to the future (j>i)", () => {
    const variants = ["full", "sparse", "swa", "dsa", "csa"] as const;
    for (const v of variants) {
      for (let i = 0; i < L; i++) {
        for (let j = i + 1; j < L; j++) {
          expect(attends(v, i, j, L)).toBe(false);
        }
      }
    }
  });
});

describe("attended fraction (FLOPs proxy)", () => {
  it("full causal = lower triangle incl diagonal = (L+1)/(2L)", () => {
    expect(attendedFraction("full", 10)).toBeCloseTo(55 / 100, 9);
  });
  it("SWA attends fewer cells than full (window<L)", () => {
    expect(attendedFraction("swa", L, 4)).toBeLessThan(attendedFraction("full", L));
  });
  it("SWA window=1 keeps only the diagonal = 1/L", () => {
    expect(attendedFraction("swa", 10, 1)).toBeCloseTo(1 / 10, 9);
  });
});
