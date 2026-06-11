import { describe, it, expect } from "vitest";
import { gradMagnitude } from "./residual-fns";

const L = 24;

describe("gradient magnitude across depth", () => {
  it("plain network vanishes at early layers (w<1)", () => {
    expect(gradMagnitude("plain", 0, L, 0.8)).toBeLessThan(0.01);
  });
  it("residual (RC) never vanishes: >= 1 everywhere", () => {
    expect(gradMagnitude("rc", 0, L)).toBeGreaterThanOrEqual(1);
    expect(gradMagnitude("rc", L, L)).toBeGreaterThanOrEqual(1);
  });
  it("residual >> plain at the earliest layer", () => {
    expect(gradMagnitude("rc", 0, L)).toBeGreaterThan(gradMagnitude("plain", 0, L, 0.8));
  });
  it("HC with more streams is at least as stable as RC", () => {
    expect(gradMagnitude("hc", 0, L, 0.8, 4)).toBeGreaterThanOrEqual(gradMagnitude("rc", 0, L));
  });
  it("at the output layer gradient is ~1 for residual", () => {
    expect(gradMagnitude("rc", L, L)).toBeCloseTo(1, 6);
  });
});
