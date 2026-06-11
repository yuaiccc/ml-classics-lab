import { describe, it, expect } from "vitest";
import { kvHeads, cacheElemsPerToken } from "./kv-cache";

const cfg = { nHeads: 8, nGroups: 2, dHead: 64, seqLen: 1024, latentDim: 64 };

describe("KV head counts", () => {
  it("MHA uses one KV head per Q head", () => {
    expect(kvHeads("mha", cfg)).toBe(8);
  });
  it("MQA uses a single shared KV head", () => {
    expect(kvHeads("mqa", cfg)).toBe(1);
  });
  it("GQA uses nGroups KV heads", () => {
    expect(kvHeads("gqa", cfg)).toBe(2);
  });
});

describe("KV-cache per token (elements)", () => {
  it("MHA = 2 * nHeads * dHead", () => {
    expect(cacheElemsPerToken("mha", cfg)).toBe(2 * 8 * 64);
  });
  it("MQA = 2 * 1 * dHead", () => {
    expect(cacheElemsPerToken("mqa", cfg)).toBe(2 * 1 * 64);
  });
  it("GQA = 2 * nGroups * dHead, between MQA and MHA", () => {
    const g = cacheElemsPerToken("gqa", cfg);
    expect(g).toBe(2 * 2 * 64);
    expect(g).toBeGreaterThan(cacheElemsPerToken("mqa", cfg));
    expect(g).toBeLessThan(cacheElemsPerToken("mha", cfg));
  });
  it("MLA stores a single compressed latent (= latentDim)", () => {
    expect(cacheElemsPerToken("mla", cfg)).toBe(64);
  });
  it("ordering: MHA > GQA > MQA, and MLA is smallest here", () => {
    const mha = cacheElemsPerToken("mha", cfg);
    const gqa = cacheElemsPerToken("gqa", cfg);
    const mqa = cacheElemsPerToken("mqa", cfg);
    const mla = cacheElemsPerToken("mla", cfg);
    expect(mha).toBeGreaterThan(gqa);
    expect(gqa).toBeGreaterThan(mqa);
    expect(mla).toBeLessThanOrEqual(mqa);
  });
});
