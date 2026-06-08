import { describe, it, expect } from "vitest";
import { relu, gelu, silu, swish, dRelu, dSilu } from "./activation-fns";

describe("activation functions", () => {
  it("relu clamps negatives to 0 and passes positives", () => {
    expect(relu(-2)).toBe(0);
    expect(relu(3)).toBe(3);
    expect(relu(0)).toBe(0);
  });

  it("relu derivative is step", () => {
    expect(dRelu(-1)).toBe(0);
    expect(dRelu(2)).toBe(1);
  });

  it("silu(x) = x*sigmoid(x), 0 at 0, negative dip below 0", () => {
    expect(silu(0)).toBeCloseTo(0, 6);
    expect(silu(10)).toBeCloseTo(10, 2); // 大正值趋近 x
    expect(silu(-2)).toBeLessThan(0); // 有负值下凹
  });

  it("swish(x, beta=1) equals silu(x)", () => {
    expect(swish(1.3, 1)).toBeCloseTo(silu(1.3), 9);
  });

  it("swish beta->large approaches relu shape on positives", () => {
    expect(swish(2, 50)).toBeCloseTo(2, 2);
    expect(swish(-2, 50)).toBeCloseTo(0, 2);
  });

  it("gelu near silu in shape: ~0 at 0, ~x at large positive", () => {
    expect(gelu(0)).toBeCloseTo(0, 6);
    expect(gelu(8)).toBeCloseTo(8, 2);
    expect(gelu(-3)).toBeLessThan(0.001);
  });

  it("dSilu finite-difference sanity", () => {
    const x = 0.7;
    const h = 1e-5;
    const numeric = (silu(x + h) - silu(x - h)) / (2 * h);
    expect(dSilu(x)).toBeCloseTo(numeric, 4);
  });
});
