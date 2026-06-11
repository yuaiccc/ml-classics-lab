# LLM 架构解剖 · Batch 4（线性·SSM + 稀疏注意力）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** 板块最后两个对比实验：线性·SSM（Lightning/GatedDeltaNet/KDA/Mamba/Mamba-2/Mamba-3）与稀疏·窗口注意力（Full/Sparse/SWA/DSA/CSA），后者叠一版**本地真实注意力**切片。

**Architecture:** 沿用 Batch 1-3 模式。稀疏注意力的「真实样本」复用仓库已有的本地真训 IMDb Transformer 真实注意力（`@/data/frames/imdb-transformer.json` 末帧 `state.data` = {tokens, attention}），用现成 `AttentionHeatmap`（`@/visualizers/AttentionHeatmap`）渲染——honor「本地真实」，不现跑 HF。基建（papers.ts 的 `linear-seq`/`sparse-attention` 键、Family 的 `linear-seq`/`sparse-attn`、顶级树枝）已就位。

**Tech Stack:** React + Vite + TS、recharts、canvas、tailwind、vitest。目录 `/Users/xujunshan/Code/ml-classics-lab/rl-lab`，git 根 `/Users/xujunshan/Code/ml-classics-lab`，分支 `feat/llm-architecture-anatomy`。

> `npm` 在 `rl-lab/` 跑；`git` 在仓库根跑。spec：`docs/superpowers/specs/2026-06-09-llm-architecture-anatomy-design.md`。极新/未公开变体（Mamba-3/KDA/DSA/CSA）在 Viz 文案里标「按公开描述近似演示」。

---

## File Structure（Batch 4）

- Modify `rl-lab/src/visualizers/arch/types.ts` — 加 `LinearSeqVariant`/`LinearSeqState`/`SparseAttnVariant`/`SparseAttnState`
- Create `rl-lab/src/algorithms/arch/linear-seq-fns.ts` + `.test.ts`
- Create `rl-lab/src/algorithms/arch/linear-seq.ts`（builder）
- Create `rl-lab/src/visualizers/arch/LinearSeqViz.tsx`
- Create `rl-lab/src/algorithms/arch/sparse-attention-fns.ts` + `.test.ts`
- Create `rl-lab/src/algorithms/arch/sparse-attention.ts`（builder）
- Create `rl-lab/src/visualizers/arch/SparseAttnViz.tsx`
- Modify `rl-lab/src/data/backgrounds.ts` — 加 `linear-seq`、`sparse-attention`
- Modify `rl-lab/src/pages/AlgorithmLab.tsx` — import、2 条 DEMO、TREE 加 2 子节点

---

## Task 1: 扩 arch types

**Files:** Modify `rl-lab/src/visualizers/arch/types.ts`

- [ ] **Step 1:** 文件末尾追加：
```ts

// —— 线性·SSM ——
export type LinearSeqVariant = "lightning" | "gateddeltanet" | "kda" | "mamba" | "mamba2" | "mamba3";
export interface LinearSeqState {
  tokens: number[];
  stateSize: number;
}

// —— 稀疏·窗口注意力 ——
export type SparseAttnVariant = "full" | "sparse" | "swa" | "dsa" | "csa";
export interface SparseAttnState {
  seqLen: number;
}
```
- [ ] **Step 2:** `npm run check` → PASS。
- [ ] **Step 3:** Commit:
```bash
git add rl-lab/src/visualizers/arch/types.ts
git commit -m "feat(rl-lab): arch state types (batch 4)"
```

---

## Task 2: 线性·SSM 纯函数 + 单测（TDD）

**Files:** Create `rl-lab/src/algorithms/arch/linear-seq-fns.ts` + `.test.ts`

- [ ] **Step 1: 失败测试** — `rl-lab/src/algorithms/arch/linear-seq-fns.test.ts`：
```ts
import { describe, it, expect } from "vitest";
import { attentionMemory, recurrentMemory, scanStep, scanNorms } from "./linear-seq-fns";

describe("memory footprint vs sequence length", () => {
  it("attention KV-cache grows linearly with L", () => {
    expect(attentionMemory(10)).toBe(10);
    expect(attentionMemory(100)).toBeGreaterThan(attentionMemory(10));
  });
  it("recurrent state is constant regardless of L", () => {
    expect(recurrentMemory(16)).toBe(16);
    // 与序列长度无关：函数签名只吃 stateSize
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
```

- [ ] **Step 2:** `npm test` → 新文件 FAIL，其余通过。

- [ ] **Step 3: 实现** — `rl-lab/src/algorithms/arch/linear-seq-fns.ts`：
```ts
// 线性/SSM 序列模型的核心对比：固定大小的递归状态 vs 注意力随长度增长的 KV-cache。

// 注意力：KV-cache 随序列长度线性增长（每个历史 token 都要缓存）。
export function attentionMemory(L: number): number {
  return L;
}
// 线性/SSM：只维护一个固定大小的递归状态，与序列长度无关。
export function recurrentMemory(stateSize: number): number {
  return stateSize;
}

// 门控线性递归一步：s' = decay*s + gate*x（逐元素）。
export function scanStep(state: number[], x: number[], decay: number, gate: number): number[] {
  return state.map((s, i) => decay * s + gate * (x[i] ?? 0));
}

// 把标量 token 序列喂进一个 stateSize 维递归状态，返回每步状态的范数（给 Viz 画扫描）。
export function scanNorms(tokens: number[], stateSize: number, decay = 0.8, gate = 0.5): number[] {
  let state = new Array(stateSize).fill(0);
  const norms: number[] = [];
  for (const t of tokens) {
    const x = new Array(stateSize).fill(0).map((_, i) => Math.sin(t * 0.5 + i));
    state = scanStep(state, x, decay, gate);
    norms.push(Math.sqrt(state.reduce((a, b) => a + b * b, 0)));
  }
  return norms;
}
```

- [ ] **Step 4:** `npm test` → PASS。
- [ ] **Step 5:** Commit:
```bash
git add rl-lab/src/algorithms/arch/linear-seq-fns.ts rl-lab/src/algorithms/arch/linear-seq-fns.test.ts
git commit -m "feat(rl-lab): linear-seq (SSM) math + unit tests"
```

---

## Task 3: 线性·SSM builder + Viz + 接线

**Files:** Create `linear-seq.ts`、`LinearSeqViz.tsx`；Modify `backgrounds.ts`、`AlgorithmLab.tsx`

- [ ] **Step 1: builder** — `rl-lab/src/algorithms/arch/linear-seq.ts`：
```ts
import { Trajectory } from "@/player/types";
import { LinearSeqState } from "@/visualizers/arch/types";

export function runLinearSeq(): Trajectory<LinearSeqState> {
  const tokens = [1, 4, 7, 2, 9, 5, 3, 8, 6, 0, 2, 7, 4, 1, 9];
  return {
    meta: {
      id: "linear-seq",
      title: "线性·SSM · Lightning/DeltaNet/KDA/Mamba 1-3",
      family: "linear-seq",
      algorithm: "Linear-time Sequence Mixing",
      description: "注意力是 O(L²)、KV-cache 随长度膨胀。线性/SSM 怎么做到 O(L)、定长状态？",
      tutorial: {
        problem: "注意力处理长序列代价是 O(L²)，推理时 KV-cache 还随长度线性膨胀，长上下文吃不消。",
        intuition:
          "线性注意力与状态空间模型（SSM）把「看所有历史」改成「维护一个固定大小的递归状态」：每来一个 token 就更新状态，不用缓存全部历史——内存与序列长度无关、计算线性。Lightning 是线性注意力，GatedDeltaNet/KDA 加了门控与 delta 更新规则，Mamba 系用输入相关的选择性 SSM。代价是「压缩历史」可能丢细节，各变体都在如何更聪明地记忆上做文章。",
        watch: [
          "内存对比曲线：注意力随长度直线上升，线性/SSM 一条水平线（定长状态）",
          "扫描视图：固定大小的状态随 token 流逐步更新",
          "切变体看说明：各自改了递归/门控的什么",
        ],
        concepts: [
          { term: "KV-cache", explain: "注意力要缓存全部历史 K/V，内存 ∝ 序列长度" },
          { term: "递归状态", explain: "SSM/线性注意力只维护定长状态，内存与长度无关" },
          { term: "选择性 SSM", explain: "Mamba：让状态更新依赖输入内容，选择性记忆" },
          { term: "delta 规则", explain: "DeltaNet/KDA：用误差驱动的状态更新提升记忆精度" },
        ],
        tryThis: "拉长序列长度，看注意力内存直线飙升而 SSM 纹丝不动。",
      },
    },
    frames: [{ iter: 0, state: { tokens, stateSize: 16 }, metrics: {} }],
  };
}
```

- [ ] **Step 2: Viz** — `rl-lab/src/visualizers/arch/LinearSeqViz.tsx`：
```tsx
import { useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { LinearSeqState, LinearSeqVariant } from "./types";
import { attentionMemory, recurrentMemory, scanNorms } from "@/algorithms/arch/linear-seq-fns";

const VARIANTS: { key: LinearSeqVariant; label: string; note: string; approx?: boolean }[] = [
  { key: "lightning", label: "Lightning", note: "线性注意力：用核函数把 softmax 注意力线性化，定长状态、O(L) 计算。" },
  { key: "gateddeltanet", label: "Gated DeltaNet", note: "在线性递归上加门控 + delta 更新规则，提升长程记忆精度。" },
  { key: "kda", label: "KDA", note: "核化 delta 注意力：用核技巧改进 delta 规则的记忆能力。", approx: true },
  { key: "mamba", label: "Mamba", note: "选择性 SSM：让状态转移依赖输入内容，选择性记忆/遗忘。" },
  { key: "mamba2", label: "Mamba-2", note: "把选择性 SSM 与注意力统一（SSD），更高效、更易扩展。" },
  { key: "mamba3", label: "Mamba-3", note: "Mamba 系最新一代的进一步改进。", approx: true },
];

export default function LinearSeqViz({ state }: { state: LinearSeqState }) {
  const [variant, setVariant] = useState<LinearSeqVariant>("mamba");
  const [maxL, setMaxL] = useState(256);
  const stateSize = state.stateSize;
  const cur = VARIANTS.find((v) => v.key === variant)!;

  const memData = useMemo(
    () =>
      Array.from({ length: 17 }, (_, i) => {
        const L = Math.round((maxL * i) / 16);
        return { L, attention: attentionMemory(L), linear: recurrentMemory(stateSize) };
      }),
    [maxL, stateSize],
  );

  const norms = useMemo(() => scanNorms(state.tokens, stateSize), [state.tokens, stateSize]);
  const scanData = norms.map((n, i) => ({ step: i, norm: Number(n.toFixed(3)) }));

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

      <div className="text-[12px] text-slate-400 max-w-xl text-center">
        {cur.note}{cur.approx && <span className="text-amber-400/80"> · 按公开描述近似演示</span>}
      </div>

      <label className="flex items-center gap-3 text-xs text-slate-400">
        最大序列长度 {maxL}
        <input type="range" min={64} max={1024} step={64} value={maxL}
          onChange={(e) => setMaxL(Number(e.target.value))} className="w-56" />
      </label>

      <div className="w-full" style={{ height: 220 }}>
        <ResponsiveContainer>
          <LineChart data={memData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="L" stroke="#64748b" tick={{ fontSize: 11 }} label={{ value: "序列长度 L", position: "insideBottom", offset: -2, fontSize: 11, fill: "#64748b" }} />
            <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="attention" name="注意力 KV-cache（∝L）" stroke="#ff5252" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="linear" name="线性/SSM 定长状态" stroke="#38bdf8" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="w-full" style={{ height: 150 }}>
        <div className="text-[11px] text-slate-500 mb-1">扫描：固定大小状态的范数随 token 流更新（{variant}）</div>
        <ResponsiveContainer>
          <LineChart data={scanData} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="step" stroke="#64748b" tick={{ fontSize: 11 }} />
            <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
            <Line type="monotone" dataKey="norm" name="状态范数" stroke="#a855f7" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: background** — 在 `BACKGROUNDS` 加：
```ts
  "linear-seq": {
    realWorld:
      "线性注意力与 Mamba 系 SSM 是「后 Transformer」最前沿的方向——用定长状态、线性计算挑战注意力的 O(L²) 瓶颈，瞄准超长上下文与高效推理（基因组、长文档、端侧大模型）。",
    uses: ["超长上下文", "高效推理/端侧", "长文档/基因组", "后 Transformer 架构"],
  },
```

- [ ] **Step 4: 接线 AlgorithmLab.tsx**

(a) import：
```ts
import { runLinearSeq } from "@/algorithms/arch/linear-seq";
import LinearSeqViz from "@/visualizers/arch/LinearSeqViz";
```
(b) DEMOS 里、`residual` 那条之后加：
```ts
  {
    key: "linear-seq",
    label: "线性·SSM",
    group: "LLM 架构解剖",
    build: () => runLinearSeq(),
    Viz: LinearSeqViz,
    metricKey: "noop",
    metricLabel: "（交互演示 · 无训练曲线）",
    metricColor: "#38bdf8",
  },
```
(c) TREE 的「LLM 架构解剖（Transformer 内部）」枝 children 末尾加：
```ts
      { label: "线性·SSM", children: [{ key: "linear-seq" }] },
```

- [ ] **Step 5:** `npm run check && npm run build` → BOTH PASS。
- [ ] **Step 6:** Commit:
```bash
git add rl-lab/src/algorithms/arch/linear-seq.ts rl-lab/src/visualizers/arch/LinearSeqViz.tsx rl-lab/src/data/backgrounds.ts rl-lab/src/pages/AlgorithmLab.tsx
git commit -m "feat(rl-lab): linear-seq (SSM) comparison experiment"
```

---

## Task 4: 稀疏注意力 纯函数 + 单测（TDD）

**Files:** Create `rl-lab/src/algorithms/arch/sparse-attention-fns.ts` + `.test.ts`

- [ ] **Step 1: 失败测试** — `rl-lab/src/algorithms/arch/sparse-attention-fns.test.ts`：
```ts
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
```

- [ ] **Step 2:** `npm test` → 新文件 FAIL，其余通过。

- [ ] **Step 3: 实现** — `rl-lab/src/algorithms/arch/sparse-attention-fns.ts`：
```ts
import type { SparseAttnVariant } from "@/visualizers/arch/types";

// 第 i 个 query 是否注意第 j 个 key（因果：只看 j<=i）。
export function attends(
  variant: SparseAttnVariant,
  i: number,
  j: number,
  L: number,
  window = 4,
  stride = 3,
): boolean {
  if (j > i) return false; // 因果掩码
  switch (variant) {
    case "full": return true;
    case "swa": return i - j < window; // 滑动窗口：只看最近 window 个
    case "sparse": return i - j < window || j % stride === 0; // 局部窗口 + 固定全局列
    case "dsa": return i - j < window || (i * 31 + j * 17) % 5 === 0; // 动态稀疏（伪随机近似演示）
    case "csa": return i - j < window || j % 2 === 0; // 压缩/跨步近似演示
  }
}

// 被计算的注意力格子占全矩阵的比例（FLOPs 代理）。
export function attendedFraction(
  variant: SparseAttnVariant,
  L: number,
  window = 4,
  stride = 3,
): number {
  let count = 0;
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      if (attends(variant, i, j, L, window, stride)) count++;
    }
  }
  return count / (L * L);
}
```

- [ ] **Step 4:** `npm test` → PASS。
- [ ] **Step 5:** Commit:
```bash
git add rl-lab/src/algorithms/arch/sparse-attention-fns.ts rl-lab/src/algorithms/arch/sparse-attention-fns.test.ts
git commit -m "feat(rl-lab): sparse attention math + unit tests"
```

---

## Task 5: 稀疏注意力 builder + Viz（含本地真实切片）+ 接线

**Files:** Create `sparse-attention.ts`、`SparseAttnViz.tsx`；Modify `backgrounds.ts`、`AlgorithmLab.tsx`

- [ ] **Step 1: builder** — `rl-lab/src/algorithms/arch/sparse-attention.ts`：
```ts
import { Trajectory } from "@/player/types";
import { SparseAttnState } from "@/visualizers/arch/types";

export function runSparseAttention(): Trajectory<SparseAttnState> {
  return {
    meta: {
      id: "sparse-attention",
      title: "稀疏·窗口注意力 · Full/Sparse/SWA/DSA/CSA",
      family: "sparse-attn",
      algorithm: "Sparse & Windowed Attention",
      description: "全注意力是 O(L²)。只算一部分格子能省多少、又保留什么？",
      tutorial: {
        problem: "全连接注意力要算 L×L 个格子，长序列代价爆炸。能不能只算「重要的」格子？",
        intuition:
          "稀疏/窗口注意力按某种图案只计算一部分注意力格子：滑动窗口（SWA）只看最近若干个 token；Sparse Transformer 用局部窗口 + 固定全局列；DSA 动态地选要算哪些；CSA/HCA 压缩或跨步。代价小很多、效果接近全注意力——因为真实注意力本来就很「集中」（看右边真实样本）。",
        watch: [
          "切变体看掩码图案：哪些格子被计算（亮）",
          "FLOPs 占比：被算格子 / 全矩阵，越稀疏越省",
          "调窗口大小，看 SWA 的带宽变化",
          "「真实样本」：本地真训 Transformer 的真实注意力——天然集中/稀疏",
        ],
        concepts: [
          { term: "因果掩码", explain: "每个 token 只能注意自己左边（含自己）" },
          { term: "SWA 滑动窗口", explain: "只注意最近 window 个 token（Longformer/Mistral）" },
          { term: "Sparse Transformer", explain: "局部窗口 + 固定全局位置的稀疏图案" },
          { term: "DSA/CSA", explain: "动态稀疏 / 压缩注意力（较新，按公开描述近似演示）" },
        ],
        tryThis: "把序列调长、切到 SWA，看 FLOPs 占比怎么从 ~50% 掉到很小。",
      },
    },
    frames: [{ iter: 0, state: { seqLen: 24 }, metrics: {} }],
  };
}
```

- [ ] **Step 2: Viz** — `rl-lab/src/visualizers/arch/SparseAttnViz.tsx`：
```tsx
import { useEffect, useMemo, useRef, useState } from "react";
import { SparseAttnState, SparseAttnVariant } from "./types";
import { attends, attendedFraction } from "@/algorithms/arch/sparse-attention-fns";
import AttentionHeatmap from "@/visualizers/AttentionHeatmap";
import imdbTransformer from "@/data/frames/imdb-transformer.json";
import type { AttentionState } from "./lab2-types";

const VARIANTS: { key: SparseAttnVariant; label: string; approx?: boolean }[] = [
  { key: "full", label: "Full（因果全连接）" },
  { key: "swa", label: "SWA 滑动窗口" },
  { key: "sparse", label: "Sparse Transformer" },
  { key: "dsa", label: "DSA", approx: true },
  { key: "csa", label: "CSA/HCA", approx: true },
];

// 仓库已有的本地真训 IMDb Transformer 的真实注意力（末帧）。
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const REAL_ATTN = (imdbTransformer as any).frames[(imdbTransformer as any).frames.length - 1].state.data as AttentionState;

function MaskCanvas({ variant, L, window }: { variant: SparseAttnVariant; L: number; window: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const cell = Math.max(4, Math.floor(360 / L));
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, L * cell, L * cell);
    for (let i = 0; i < L; i++) {
      for (let j = 0; j < L; j++) {
        const on = attends(variant, i, j, L, window);
        ctx.fillStyle = on ? "#38bdf8" : "#1e293b";
        ctx.fillRect(j * cell, i * cell, cell - 1, cell - 1);
      }
    }
  }, [variant, L, window, cell]);
  return <canvas ref={ref} width={L * cell} height={L * cell} style={{ maxWidth: "100%", height: "auto" }} />;
}

export default function SparseAttnViz({ state }: { state: SparseAttnState }) {
  const [variant, setVariant] = useState<SparseAttnVariant>("swa");
  const [L, setL] = useState(state.seqLen);
  const [window, setWindow] = useState(4);
  const [showReal, setShowReal] = useState(false);
  const cur = VARIANTS.find((v) => v.key === variant)!;
  const frac = useMemo(() => attendedFraction(variant, L, window), [variant, L, window]);

  return (
    <div className="flex flex-col items-center gap-4 w-full">
      <div className="flex flex-wrap items-center gap-2">
        {VARIANTS.map((v) => (
          <button key={v.key} onClick={() => { setVariant(v.key); setShowReal(false); }}
            className={`px-3 py-1 rounded text-xs font-mono border transition-all ${
              !showReal && variant === v.key ? "bg-sky-500/25 border-sky-400 text-sky-200"
                : "bg-slate-800/40 border-slate-600 text-slate-400 hover:border-slate-400"}`}>
            {v.label}
          </button>
        ))}
        <button onClick={() => setShowReal((s) => !s)}
          className={`px-3 py-1 rounded text-xs font-mono border transition-all ${
            showReal ? "bg-purple-500/25 border-purple-400 text-purple-200"
              : "bg-slate-800/40 border-slate-600 text-slate-400 hover:border-slate-400"}`}>
          真实样本
        </button>
      </div>

      {showReal ? (
        <div className="flex flex-col items-center gap-2">
          <div className="text-[12px] text-purple-300/90 max-w-xl text-center">
            本地真训 IMDb Transformer 的真实注意力（非掩码图案）——可见注意力天然集中在少数 token 上，这正是稀疏注意力「只算重要格子也够用」的依据。
          </div>
          <AttentionHeatmap state={REAL_ATTN} />
        </div>
      ) : (
        <>
          <div className="text-[12px] text-slate-400 max-w-xl text-center">
            掩码图案（亮=被计算）。{cur.approx && <span className="text-amber-400/80">该变体按公开描述近似演示。</span>}
          </div>
          <div className="flex flex-wrap gap-5 text-xs text-slate-400">
            <label className="flex items-center gap-2">序列长 {L}
              <input type="range" min={8} max={48} step={2} value={L}
                onChange={(e) => setL(Number(e.target.value))} className="w-32" />
            </label>
            {(variant === "swa" || variant === "sparse" || variant === "dsa" || variant === "csa") && (
              <label className="flex items-center gap-2">窗口 {window}
                <input type="range" min={1} max={12} step={1} value={window}
                  onChange={(e) => setWindow(Number(e.target.value))} className="w-28" />
              </label>
            )}
          </div>
          <MaskCanvas variant={variant} L={L} window={window} />
          <div className="text-xs text-slate-400">
            被计算格子（FLOPs 占比）：<span className="font-mono text-sky-300">{(frac * 100).toFixed(1)}%</span>
            <span className="text-slate-600"> · 行=query，列=key，因果下三角</span>
          </div>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 3: background** — 在 `BACKGROUNDS` 加：
```ts
  "sparse-attention": {
    realWorld:
      "稀疏/窗口注意力是长上下文大模型的关键省钱手段——Longformer、Mistral 的滑动窗口、DeepSeek 的稀疏注意力，都是为了把 O(L²) 压下来又尽量不掉效果。",
    uses: ["长上下文", "推理降本", "长文档处理", "高效注意力"],
  },
```

- [ ] **Step 4: 接线 AlgorithmLab.tsx**

(a) import：
```ts
import { runSparseAttention } from "@/algorithms/arch/sparse-attention";
import SparseAttnViz from "@/visualizers/arch/SparseAttnViz";
```
(b) DEMOS 里、`linear-seq` 那条之后加：
```ts
  {
    key: "sparse-attention",
    label: "稀疏·窗口注意力",
    group: "LLM 架构解剖",
    build: () => runSparseAttention(),
    Viz: SparseAttnViz,
    metricKey: "noop",
    metricLabel: "（交互演示 · 无训练曲线）",
    metricColor: "#38bdf8",
  },
```
(c) TREE 的「LLM 架构解剖（Transformer 内部）」枝 children 末尾加：
```ts
      { label: "稀疏·窗口注意力", children: [{ key: "sparse-attention" }] },
```
(d) ENGINE map：`sparse-attention` 复用了本地真训模型的真实注意力，标 `pytorch` 徽章。找到 `const ENGINE: Record<...> = {` 块，在其中加一行：
```ts
  "sparse-attention": "pytorch",
```

- [ ] **Step 5:** `npm run check && npm test && npm run build` → 三者 PASS（测试 39 + 线性4 + 稀疏3 = 约 46，全绿即可）。
- [ ] **Step 6:** Commit:
```bash
git add rl-lab/src/algorithms/arch/sparse-attention.ts rl-lab/src/visualizers/arch/SparseAttnViz.tsx rl-lab/src/data/backgrounds.ts rl-lab/src/pages/AlgorithmLab.tsx
git commit -m "feat(rl-lab): sparse attention comparison experiment (+ real local attention slice)"
```

---

## Task 6: 端到端人工验证

- [ ] **Step 1:** `npm run dev`（5180）。在「LLM 架构解剖」枝下确认新增「线性·SSM」「稀疏·窗口注意力」：
  - 线性·SSM：6 变体可切（KDA/Mamba-3 标近似）；内存曲线注意力直线上升、SSM 水平；扫描状态范数曲线显示；序列长滑块生效。参考文献列 Lightning/DeltaNet/KDA/Mamba1-3。
  - 稀疏注意力：5 变体掩码图案可切；FLOPs 占比随窗口/变体变化；「真实样本」按钮切到本地真实注意力热力图（可切层/头）；带 PyTorch 徽章。参考文献列 Sparse/SWA/Gemma2/DSA/CSA（DSA/CSA 标近似）。
  - 来回切换无崩溃。
- [ ] **Step 2:** 无误打勾。

---

## Self-Review

- **覆盖**：spec §3 表 #5 线性·SSM、#4 稀疏注意力（含 §6 本地真实切片——以复用已有本地真训 Transformer 真实注意力的方式落地，已在 builder/Viz 文案注明来源）。
- **占位符**：无。
- **类型一致**：`LinearSeqVariant`/`LinearSeqState`/`SparseAttnVariant`/`SparseAttnState` 在 Task 1 定义，被 fns/builder/Viz 一致引用；`attentionMemory`/`recurrentMemory`/`scanStep`/`scanNorms`、`attends`/`attendedFraction` 命名一致；DEMO key 与 papers.ts 键、TREE 叶子一致（`linear-seq`/`sparse-attention`）；真实注意力复用 `AttentionHeatmap` + `lab2-types` 的 `AttentionState`。
- **family**：`linear-seq`、`sparse-attn` 已在 Batch 1 入 Family 联合。
- **运行时安全**：单帧 + metricKey "noop"，MetricCurve `?? null` 容错。
- **本地真实切片说明**：因环境无可用 transformers 现跑，改为复用仓库内已有的本地真训 IMDb Transformer 真实注意力（同类产物），Viz 文案如实标注来源；保留 PyTorch 徽章表征其「真实模型产物」属性。
```
