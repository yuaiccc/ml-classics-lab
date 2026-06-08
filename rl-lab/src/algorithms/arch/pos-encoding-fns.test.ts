import { describe, it, expect } from "vitest";
import { sinusoidalPE, ropeApply, yarnApply, dot } from "./pos-encoding-fns";

const norm = (v: number[]) => Math.sqrt(dot(v, v));

describe("sinusoidal PE", () => {
  it("at pos=0 is all sin=0 / cos=1", () => {
    expect(sinusoidalPE(0, 8)).toEqual([0, 1, 0, 1, 0, 1, 0, 1]);
  });
  it("returns a vector of the requested length", () => {
    expect(sinusoidalPE(5, 16)).toHaveLength(16);
  });
});

describe("RoPE", () => {
  it("is a rotation: preserves vector norm", () => {
    const v = [0.3, -0.7, 1.1, 0.2];
    expect(norm(ropeApply(v, 7))).toBeCloseTo(norm(v), 6);
  });
  it("relative-position property: q·k depends only on the position offset", () => {
    const q = [0.5, 0.9, -0.3, 0.4];
    const k = [0.2, -0.6, 0.8, 0.1];
    const a = dot(ropeApply(q, 5), ropeApply(k, 8)); // offset -3
    const b = dot(ropeApply(q, 2), ropeApply(k, 5)); // offset -3
    const c = dot(ropeApply(q, 10), ropeApply(k, 13)); // offset -3
    expect(a).toBeCloseTo(b, 6);
    expect(a).toBeCloseTo(c, 6);
  });
});

describe("YaRN", () => {
  it("rotates slower than RoPE (extends effective range)", () => {
    // 用第 2 个二维对（i=2）：第 1 对 freq=base^0=1 与 base 无关，体现不出 YaRN 差异。
    const v = [0, 0, 1, 0];
    const pos = 6;
    expect(dot(yarnApply(v, pos), v)).toBeGreaterThan(dot(ropeApply(v, pos), v));
  });
});
