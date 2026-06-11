# LLM 架构解剖 · Batch 3（MoE + 残差）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 在「LLM 架构解剖」板块再加两个对比实验：MoE（Dense/MoE/Switch/DeepSeekMoE）与残差连接（Plain/RC/HC/mHC/AttnResidual）。

**Architecture:** 沿用 Batch 1-2 模式：可测纯数学抽到 `algorithms/arch/*-fns.ts` 并 vitest 单测；builder 出单帧 Trajectory 携 tutorial；交互放进 Viz。基建（papers.ts 的 `moe`/`residual` 键、Family 联合里的 `moe`/`residual`、顶级树枝）已就位。

**Tech Stack:** React + Vite + TS、recharts、SVG、tailwind、vitest。目录 `/Users/xujunshan/Code/ml-classics-lab/rl-lab`，git 根 `/Users/xujunshan/Code/ml-classics-lab`，分支 `feat/llm-architecture-anatomy`。

> `npm` 在 `rl-lab/` 跑；`git` 在仓库根跑（add 路径形如 `rl-lab/...`）。spec：`docs/superpowers/specs/2026-06-09-llm-architecture-anatomy-design.md`。

---

## File Structure（Batch 3）

- Modify `rl-lab/src/visualizers/arch/types.ts` — 加 `MoEVariant`/`MoEState`/`ResidualVariant`/`ResidualState`
- Create `rl-lab/src/algorithms/arch/moe-fns.ts` + `.test.ts`
- Create `rl-lab/src/algorithms/arch/moe.ts`（builder）
- Create `rl-lab/src/visualizers/arch/MoEViz.tsx`
- Create `rl-lab/src/algorithms/arch/residual-fns.ts` + `.test.ts`
- Create `rl-lab/src/algorithms/arch/residual.ts`（builder）
- Create `rl-lab/src/visualizers/arch/ResidualViz.tsx`
- Modify `rl-lab/src/data/backgrounds.ts` — 加 `moe`、`residual`
- Modify `rl-lab/src/pages/AlgorithmLab.tsx` — import、2 条 DEMO、TREE 加 2 子节点

---

## Task 1: 扩 arch types

**Files:** Modify `rl-lab/src/visualizers/arch/types.ts`

- [ ] **Step 1:** 在文件末尾追加：
```ts

// —— MoE ——
export type MoEVariant = "dense" | "moe" | "switch" | "deepseek";
export interface MoEState {
  tokens: number[]; // 玩具 token 序列（值用于确定性路由）
  nExperts: number;
  topK: number;
  nShared: number; // DeepSeekMoE 常驻共享专家数
}

// —— 残差连接 ——
export type ResidualVariant = "plain" | "rc" | "hc" | "mhc" | "attnres";
export interface ResidualState {
  nLayers: number;
}
```
- [ ] **Step 2:** `npm run check` → PASS。
- [ ] **Step 3:** Commit:
```bash
git add rl-lab/src/visualizers/arch/types.ts
git commit -m "feat(rl-lab): arch state types (batch 3)"
```

---

## Task 2: MoE 纯函数 + 单测（TDD）

**Files:** Create `rl-lab/src/algorithms/arch/moe-fns.ts` + `.test.ts`

- [ ] **Step 1: 失败测试** — `rl-lab/src/algorithms/arch/moe-fns.test.ts`：
```ts
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
```

- [ ] **Step 2:** `npm test` → 新文件 FAIL（模块缺失），其余通过。

- [ ] **Step 3: 实现** — `rl-lab/src/algorithms/arch/moe-fns.ts`：
```ts
import type { MoEVariant } from "@/visualizers/arch/types";

// 确定性门控分数：用 token 值与专家索引算一个 [0,1] 分数。
export function gateScores(token: number, nExperts: number): number[] {
  const s: number[] = [];
  for (let e = 0; e < nExperts; e++) {
    s.push(Math.sin(token * 0.7 + e * 1.3) * 0.5 + 0.5);
  }
  return s;
}

// 分数最高的 k 个专家索引（降序）。
export function topKExperts(scores: number[], k: number): number[] {
  return scores
    .map((v, i) => [v, i] as [number, number])
    .sort((a, b) => b[0] - a[0])
    .slice(0, Math.max(0, k))
    .map((x) => x[1]);
}

// 该 token 经路由激活的专家索引（不含常驻共享专家，由 Viz 单独标）。
export function routedExperts(variant: MoEVariant, token: number, nExperts: number, topK: number): number[] {
  const scores = gateScores(token, nExperts);
  if (variant === "dense") return scores.map((_, i) => i);
  if (variant === "switch") return topKExperts(scores, 1);
  return topKExperts(scores, topK); // moe / deepseek
}

// 每个 token 激活的专家占比（稀疏度）。
export function activeFraction(variant: MoEVariant, nExperts: number, topK: number, nShared: number): number {
  if (variant === "dense") return 1;
  if (variant === "switch") return 1 / nExperts;
  if (variant === "moe") return topK / nExperts;
  return (topK + nShared) / (nExperts + nShared); // deepseek：细粒度路由 + 常驻共享
}
```

- [ ] **Step 4:** `npm test` → PASS（全绿）。
- [ ] **Step 5:** Commit:
```bash
git add rl-lab/src/algorithms/arch/moe-fns.ts rl-lab/src/algorithms/arch/moe-fns.test.ts
git commit -m "feat(rl-lab): MoE routing math + unit tests"
```

---

## Task 3: MoE builder + Viz + 接线

**Files:** Create `moe.ts`、`MoEViz.tsx`；Modify `backgrounds.ts`、`AlgorithmLab.tsx`

- [ ] **Step 1: builder** — `rl-lab/src/algorithms/arch/moe.ts`：
```ts
import { Trajectory } from "@/player/types";
import { MoEState } from "@/visualizers/arch/types";

export function runMoE(): Trajectory<MoEState> {
  // 玩具 token 序列（值各异，触发不同路由）。
  const tokens = [1, 4, 7, 2, 9, 5, 3, 8, 6, 0, 11, 13];
  return {
    meta: {
      id: "moe",
      title: "专家混合 MoE · Dense→MoE→Switch→DeepSeekMoE",
      family: "moe",
      algorithm: "Mixture of Experts",
      description: "如何把模型参数堆大，但每个 token 只算一小部分？",
      tutorial: {
        problem: "想要更大的模型容量，又不想让每个 token 都过一遍全部参数（太贵）。",
        intuition:
          "MoE 把 FFN 拆成很多「专家」，一个路由器给每个 token 只挑少数几个专家来算——参数总量大、单 token 计算量小（稀疏激活）。Switch 极端到每 token 只选 1 个专家。DeepSeekMoE 用更细粒度的专家 + 几个常驻「共享专家」，兼顾专精与通用。难点是负载均衡：别让少数专家被挤爆、其余闲置。",
        watch: [
          "切 Dense→MoE→Switch：每个 token 点亮的专家越来越少（越稀疏）",
          "稀疏度条：激活专家/总专家，Dense=100%，Switch=1/N",
          "负载均衡条：各专家被选中的次数，越均匀越好",
          "DeepSeekMoE：常驻共享专家始终点亮 + 细粒度路由",
        ],
        concepts: [
          { term: "专家 Expert", explain: "一个独立的 FFN 子网络" },
          { term: "路由器 Router", explain: "给每个 token 打分、挑选 top-k 专家" },
          { term: "稀疏激活", explain: "总参数大，但每 token 只算少数专家" },
          { term: "负载均衡", explain: "让 token 均匀分到各专家，避免少数被挤爆" },
        ],
        tryThis: "拖动 token 滑块逐个看路由；切到 Switch 看是否只点亮 1 个专家。",
      },
    },
    frames: [{ iter: 0, state: { tokens, nExperts: 8, topK: 2, nShared: 2 }, metrics: {} }],
  };
}
```

- [ ] **Step 2: Viz** — `rl-lab/src/visualizers/arch/MoEViz.tsx`：
```tsx
import { useMemo, useState } from "react";
import { MoEState, MoEVariant } from "./types";
import { routedExperts, activeFraction } from "@/algorithms/arch/moe-fns";

const VARIANTS: { key: MoEVariant; label: string }[] = [
  { key: "dense", label: "Dense" },
  { key: "moe", label: "MoE (top-k)" },
  { key: "switch", label: "Switch (top-1)" },
  { key: "deepseek", label: "DeepSeekMoE" },
];

export default function MoEViz({ state }: { state: MoEState }) {
  const [variant, setVariant] = useState<MoEVariant>("moe");
  const [nExperts, setNExperts] = useState(state.nExperts);
  const [topK, setTopK] = useState(state.topK);
  const [tokenIdx, setTokenIdx] = useState(0);
  const tokens = state.tokens;
  const nShared = state.nShared;

  const curToken = tokens[tokenIdx];
  const active = useMemo(
    () => new Set(routedExperts(variant, curToken, nExperts, topK)),
    [variant, curToken, nExperts, topK],
  );
  const sharedActive = variant === "deepseek";

  // 负载均衡：每个专家在整条序列里被选中的次数
  const load = useMemo(() => {
    const counts = new Array(nExperts).fill(0);
    tokens.forEach((t) => routedExperts(variant, t, nExperts, topK).forEach((e) => (counts[e] += 1)));
    return counts;
  }, [tokens, variant, nExperts, topK]);
  const maxLoad = Math.max(1, ...load);

  const frac = activeFraction(variant, nExperts, topK, nShared);

  const W = 480, eW = Math.min(44, (W - 40) / nExperts - 6), gap = 6, eY = 150;
  const eX = (i: number) => 20 + i * (eW + gap);

  return (
    <div className="flex flex-col items-center gap-4 w-full">
      <div className="flex flex-wrap items-center gap-2">
        {VARIANTS.map((v) => (
          <button key={v.key} onClick={() => setVariant(v.key)}
            className={`px-3 py-1 rounded text-xs font-mono border transition-all ${
              variant === v.key ? "bg-sky-500/25 border-sky-400 text-sky-200"
                : "bg-slate-800/40 border-slate-600 text-slate-400 hover:border-slate-400"}`}>
            {v.label}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap gap-5 text-xs text-slate-400">
        <label className="flex items-center gap-2">专家数 {nExperts}
          <input type="range" min={4} max={12} step={1} value={nExperts}
            onChange={(e) => setNExperts(Number(e.target.value))} className="w-28" />
        </label>
        {(variant === "moe" || variant === "deepseek") && (
          <label className="flex items-center gap-2">top-k {topK}
            <input type="range" min={1} max={Math.min(4, nExperts)} step={1} value={topK}
              onChange={(e) => setTopK(Number(e.target.value))} className="w-24" />
          </label>
        )}
        <label className="flex items-center gap-2">token #{tokenIdx} (值={curToken})
          <input type="range" min={0} max={tokens.length - 1} step={1} value={tokenIdx}
            onChange={(e) => setTokenIdx(Number(e.target.value))} className="w-28" />
        </label>
      </div>

      {/* 路由示意：当前 token → 激活专家 */}
      <svg width={W} height={200} className="max-w-full">
        <g>
          <rect x={W / 2 - 24} y={20} width={48} height={26} rx={4} fill="#0ea5e9" />
          <text x={W / 2} y={37} textAnchor="middle" fontSize={11} fill="#fff">tok {curToken}</text>
        </g>
        {Array.from({ length: nExperts }).map((_, i) => {
          const on = active.has(i);
          return (
            <g key={i}>
              {on && <line x1={W / 2} y1={46} x2={eX(i) + eW / 2} y2={eY} stroke="#38bdf8" strokeWidth={1.5} />}
              <rect x={eX(i)} y={eY} width={eW} height={28} rx={4}
                fill={on ? "#38bdf8" : "#1e293b"} stroke={on ? "#7dd3fc" : "#334155"} />
              <text x={eX(i) + eW / 2} y={eY + 18} textAnchor="middle" fontSize={10} fill={on ? "#fff" : "#64748b"}>E{i}</text>
            </g>
          );
        })}
        {sharedActive && (
          <text x={20} y={eY - 8} fontSize={10} fill="#a855f7">+ {nShared} 个常驻共享专家始终参与</text>
        )}
      </svg>

      <div className="w-full max-w-md flex flex-col gap-1">
        <div className="text-xs text-slate-400">
          每 token 稀疏激活：<span className="font-mono text-sky-300">{(frac * 100).toFixed(0)}%</span> 的专家
        </div>
        <div className="text-xs text-slate-400 mt-2">负载均衡（各专家在整条序列被选次数）</div>
        <div className="flex items-end gap-1 h-16">
          {load.map((c, i) => (
            <div key={i} className="flex-1 bg-slate-700 rounded-t" style={{ height: `${(c / maxLoad) * 100}%`, minHeight: 2 }} title={`E${i}: ${c}`} />
          ))}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: background** — 在 `BACKGROUNDS` 加：
```ts
  moe: {
    realWorld:
      "MoE 是当下顶级大模型（Mixtral、DeepSeek-V3、传闻中的 GPT-4）做大又做省的关键——参数量上万亿，但每个 token 只激活一小撮专家，推理成本可控。",
    uses: ["万亿参数大模型", "稀疏激活/降本", "专家专精", "DeepSeek/Mixtral 架构"],
  },
```

- [ ] **Step 4: 接线 AlgorithmLab.tsx**

(a) import：
```ts
import { runMoE } from "@/algorithms/arch/moe";
import MoEViz from "@/visualizers/arch/MoEViz";
```
(b) DEMOS 里、`normalization` 那条之后加：
```ts
  {
    key: "moe",
    label: "专家混合 MoE",
    group: "LLM 架构解剖",
    build: () => runMoE(),
    Viz: MoEViz,
    metricKey: "noop",
    metricLabel: "（交互演示 · 无训练曲线）",
    metricColor: "#38bdf8",
  },
```
(c) TREE 的「LLM 架构解剖（Transformer 内部）」枝 children 末尾加：
```ts
      { label: "专家混合 MoE", children: [{ key: "moe" }] },
```

- [ ] **Step 5:** `npm run check && npm run build` → BOTH PASS。
- [ ] **Step 6:** Commit:
```bash
git add rl-lab/src/algorithms/arch/moe.ts rl-lab/src/visualizers/arch/MoEViz.tsx rl-lab/src/data/backgrounds.ts rl-lab/src/pages/AlgorithmLab.tsx
git commit -m "feat(rl-lab): MoE comparison experiment"
```

---

## Task 4: 残差纯函数 + 单测（TDD）

**Files:** Create `rl-lab/src/algorithms/arch/residual-fns.ts` + `.test.ts`

- [ ] **Step 1: 失败测试** — `rl-lab/src/algorithms/arch/residual-fns.test.ts`：
```ts
import { describe, it, expect } from "vitest";
import { gradMagnitude } from "./residual-fns";

const L = 24;

describe("gradient magnitude across depth", () => {
  it("plain network vanishes at early layers (w<1)", () => {
    // 离输出 24 层，0.8^24 ≈ 0.0047
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
```

- [ ] **Step 2:** `npm test` → 新文件 FAIL，其余通过。

- [ ] **Step 3: 实现** — `rl-lab/src/algorithms/arch/residual-fns.ts`：
```ts
import type { ResidualVariant } from "@/visualizers/arch/types";

// 反向传播到第 depth 层时的梯度幅度（玩具模型）。depth: 输入端=0 … 输出端=nLayers。
// 直觉：plain 无捷径→梯度按 w 指数衰减（消失）或膨胀（爆炸）；
// 残差有恒等捷径，梯度至少有「+1」的直通路，不会消失。
export function gradMagnitude(
  variant: ResidualVariant,
  depth: number,
  nLayers: number,
  w = 0.8,
  streams = 1,
): number {
  const fromOutput = nLayers - depth; // 离输出还有多少层要回传
  switch (variant) {
    case "plain": return Math.pow(w, fromOutput);
    case "rc": return 1 + fromOutput * 0.1;
    case "hc": return 1 + fromOutput * 0.1 * streams;
    case "mhc": return 1 + fromOutput * 0.12 * streams;
    case "attnres": return 1 + fromOutput * 0.08;
  }
}
```

- [ ] **Step 4:** `npm test` → PASS。
- [ ] **Step 5:** Commit:
```bash
git add rl-lab/src/algorithms/arch/residual-fns.ts rl-lab/src/algorithms/arch/residual-fns.test.ts
git commit -m "feat(rl-lab): residual gradient math + unit tests"
```

---

## Task 5: 残差 builder + Viz + 接线

**Files:** Create `residual.ts`、`ResidualViz.tsx`；Modify `backgrounds.ts`、`AlgorithmLab.tsx`

- [ ] **Step 1: builder** — `rl-lab/src/algorithms/arch/residual.ts`：
```ts
import { Trajectory } from "@/player/types";
import { ResidualState } from "@/visualizers/arch/types";

export function runResidual(): Trajectory<ResidualState> {
  return {
    meta: {
      id: "residual",
      title: "残差连接 · Plain→ResNet→HC→mHC→AttnResidual",
      family: "residual",
      algorithm: "Residual Connections",
      description: "网络一深就训不动，残差连接如何让梯度活下来？",
      tutorial: {
        problem: "层一多，反向传播的梯度按层数指数衰减/爆炸，深层网络根本训不动。",
        intuition:
          "残差连接（ResNet 的 x+f(x)）给梯度开一条「恒等捷径」：就算 f 这条路梯度很小，那个「+1」也能把梯度直通回去，于是几百层也训得动。它是 Transformer 能堆深的前提。HC/mHC 用多条并行残差流提升表达与稳定，AttnResidual 用注意力来调制残差流。",
        watch: [
          "Plain：梯度幅度随离输出越远指数衰减→早层学不到（消失）",
          "RC：恒等捷径让梯度≥1，越深越稳",
          "HC/mHC：多条并行残差流，进一步抬高/稳住梯度",
          "右侧示意图：skip 连接（捷径）怎么绕过 f",
        ],
        concepts: [
          { term: "残差连接", explain: "y = x + f(x)，给梯度一条恒等直通路" },
          { term: "梯度消失", explain: "无捷径时梯度按层数指数衰减，深层学不到" },
          { term: "HC/mHC", explain: "Hyper-Connections：多条并行残差流" },
          { term: "AttnResidual", explain: "用注意力调制残差流的较新变体" },
        ],
        tryThis: "切到 Plain 看梯度曲线贴地（消失），再切回 RC 看它被「+1」托起来。",
      },
    },
    frames: [{ iter: 0, state: { nLayers: 24 }, metrics: {} }],
  };
}
```

- [ ] **Step 2: Viz** — `rl-lab/src/visualizers/arch/ResidualViz.tsx`：
```tsx
import { useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { ResidualState, ResidualVariant } from "./types";
import { gradMagnitude } from "@/algorithms/arch/residual-fns";

const VARIANTS: { key: ResidualVariant; label: string }[] = [
  { key: "plain", label: "Plain（无捷径）" },
  { key: "rc", label: "RC (ResNet)" },
  { key: "hc", label: "HC" },
  { key: "mhc", label: "mHC" },
  { key: "attnres", label: "AttnResidual" },
];

export default function ResidualViz({ state }: { state: ResidualState }) {
  const [variant, setVariant] = useState<ResidualVariant>("rc");
  const [nLayers, setNLayers] = useState(state.nLayers);
  const [streams, setStreams] = useState(4);

  const data = useMemo(
    () =>
      Array.from({ length: nLayers + 1 }, (_, d) => ({
        depth: d,
        cur: gradMagnitude(variant, d, nLayers, 0.8, streams),
        plain: gradMagnitude("plain", d, nLayers, 0.8),
      })),
    [variant, nLayers, streams],
  );

  const multiStream = variant === "hc" || variant === "mhc";

  return (
    <div className="flex flex-col items-center gap-4 w-full">
      <div className="flex flex-wrap items-center gap-2">
        {VARIANTS.map((v) => (
          <button key={v.key} onClick={() => setVariant(v.key)}
            className={`px-3 py-1 rounded text-xs font-mono border transition-all ${
              variant === v.key ? "bg-sky-500/25 border-sky-400 text-sky-200"
                : "bg-slate-800/40 border-slate-600 text-slate-400 hover:border-slate-400"}`}>
            {v.label}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap gap-5 text-xs text-slate-400">
        <label className="flex items-center gap-2">层数 {nLayers}
          <input type="range" min={8} max={48} step={2} value={nLayers}
            onChange={(e) => setNLayers(Number(e.target.value))} className="w-32" />
        </label>
        {multiStream && (
          <label className="flex items-center gap-2">残差流数 {streams}
            <input type="range" min={2} max={8} step={1} value={streams}
              onChange={(e) => setStreams(Number(e.target.value))} className="w-28" />
          </label>
        )}
      </div>

      <div className="w-full" style={{ height: 240 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="depth" stroke="#64748b" tick={{ fontSize: 11 }} label={{ value: "层深（0=输入）", position: "insideBottom", offset: -2, fontSize: 11, fill: "#64748b" }} />
            <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="cur" name="当前变体梯度幅度" stroke="#38bdf8" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="plain" name="Plain 参照（消失）" stroke="#ff5252" dot={false} strokeWidth={1.5} strokeDasharray="4 3" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* 残差结构示意 */}
      <svg width={320} height={120} className="max-w-full">
        <rect x={130} y={20} width={60} height={28} rx={4} fill="#1e293b" stroke="#475569" />
        <text x={160} y={38} textAnchor="middle" fontSize={11} fill="#94a3b8">f(x)</text>
        <line x1={40} y1={34} x2={130} y2={34} stroke="#64748b" strokeWidth={2} markerEnd="url(#a)" />
        <line x1={190} y1={34} x2={280} y2={34} stroke="#64748b" strokeWidth={2} />
        {variant !== "plain" && (
          <>
            <path d={`M70,34 Q70,90 160,90 Q250,90 250,42`} fill="none" stroke="#38bdf8" strokeWidth={2} strokeDasharray={multiStream ? "0" : "0"} />
            <text x={160} y={108} textAnchor="middle" fontSize={10} fill="#38bdf8">
              {multiStream ? `${streams} 条并行残差捷径 (+)` : "恒等捷径 skip (+)"}
            </text>
          </>
        )}
        <circle cx={250} cy={34} r={10} fill="#0f172a" stroke="#38bdf8" />
        <text x={250} y={38} textAnchor="middle" fontSize={12} fill="#38bdf8">+</text>
        <defs>
          <marker id="a" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#64748b" />
          </marker>
        </defs>
      </svg>
      {variant === "plain" && (
        <div className="text-[12px] text-amber-400/80 text-center">Plain 没有捷径——梯度只能走 f(x) 这条路，按层数指数衰减。</div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: background** — 在 `BACKGROUNDS` 加：
```ts
  residual: {
    realWorld:
      "残差连接（ResNet, 2015）是深度学习能「做深」的转折点——没有它就没有几百层的网络、也没有能堆几十上百层的 Transformer。它是现代大模型的地基之一。",
    uses: ["深层网络训练", "Transformer 地基", "梯度稳定", "超深模型"],
  },
```

- [ ] **Step 4: 接线 AlgorithmLab.tsx**

(a) import：
```ts
import { runResidual } from "@/algorithms/arch/residual";
import ResidualViz from "@/visualizers/arch/ResidualViz";
```
(b) DEMOS 里、`moe` 那条之后加：
```ts
  {
    key: "residual",
    label: "残差连接",
    group: "LLM 架构解剖",
    build: () => runResidual(),
    Viz: ResidualViz,
    metricKey: "noop",
    metricLabel: "（交互演示 · 无训练曲线）",
    metricColor: "#38bdf8",
  },
```
(c) TREE 的「LLM 架构解剖（Transformer 内部）」枝 children 末尾加：
```ts
      { label: "残差连接", children: [{ key: "residual" }] },
```

- [ ] **Step 5:** `npm run check && npm test && npm run build` → 三者 PASS（测试 25 + MoE 11 + 残差 5 = 约 41，按实际全绿即可）。
- [ ] **Step 6:** Commit:
```bash
git add rl-lab/src/algorithms/arch/residual.ts rl-lab/src/visualizers/arch/ResidualViz.tsx rl-lab/src/data/backgrounds.ts rl-lab/src/pages/AlgorithmLab.tsx
git commit -m "feat(rl-lab): residual connections comparison experiment"
```

---

## Task 6: 端到端人工验证

- [ ] **Step 1:** `npm run dev`（5180），在「LLM 架构解剖」枝下确认新增「专家混合 MoE」「残差连接」：
  - MoE：4 变体可切；Dense 全亮、Switch 只亮 1；专家/topK/token 滑块生效；稀疏度与负载均衡条显示；DeepSeekMoE 标常驻共享专家。参考文献列 MoE/Switch/DeepSeekMoE。
  - 残差：5 变体可切；Plain 梯度曲线贴地（消失），RC/HC 抬起；层数/残差流滑块生效；结构示意图显示捷径。参考文献列 ResNet/HC/mHC/AttnResidual（后两者标「近似演示」）。
  - 来回切换无崩溃。
- [ ] **Step 2:** 无误打勾；有问题回对应 Task 修复。

---

## Self-Review

- **覆盖**：spec §3 表 #6 MoE、#8 残差。
- **占位符**：无。
- **类型一致**：`MoEVariant`/`MoEState`/`ResidualVariant`/`ResidualState` 在 Task 1 定义，被 fns/builder/Viz 一致引用；`gateScores`/`topKExperts`/`routedExperts`/`activeFraction`、`gradMagnitude` 命名一致；DEMO key 与 papers.ts 键、TREE 叶子 key 一致（`moe`/`residual`）。
- **family**：`moe`、`residual` 已在 Batch 1 入 Family 联合。
- **运行时安全**：单帧 + 空 metrics + metricKey "noop"，MetricCurve `?? null` 容错。
```
