import { describe, it, expect } from "vitest";
import { layerNorm, rmsNorm, mean, rms, cosine } from "./normalization-fns";

describe("LayerNorm", () => {
  it("centers to ~zero mean and ~unit std", () => {
    const out = layerNorm([1, 2, 3, 4]);
    expect(mean(out)).toBeCloseTo(0, 6);
    const variance = out.reduce((a, b) => a + b * b, 0) / out.length;
    expect(Math.sqrt(variance)).toBeCloseTo(1, 4);
  });
  it("changes direction when input mean != 0", () => {
    const v = [1, 2, 3];
    expect(cosine(layerNorm(v), v)).toBeLessThan(0.999);
  });
});

describe("RMSNorm", () => {
  it("preserves direction (output is a positive scalar multiple of input)", () => {
    const v = [1, 2, 3];
    expect(cosine(rmsNorm(v), v)).toBeCloseTo(1, 6);
  });
  it("does NOT center: keeps a non-zero mean for a positive-mean input", () => {
    const v = [1, 2, 3];
    expect(mean(rmsNorm(v))).toBeGreaterThan(0.1);
  });
  it("scales to ~unit RMS", () => {
    expect(rms(rmsNorm([2, -2, 2, -2]))).toBeCloseTo(1, 4);
  });
});
