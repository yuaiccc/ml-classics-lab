# LLM 架构解剖 · Batch 1（基建 + 激活函数 + 注意力KV）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在实验台新增「LLM 架构解剖」板块的基建（vitest 测试、论文引用机制、顶级树枝）并落地前两个对比实验：激活函数（ReLU/GELU/SiLU/Swish/SwiGLU）与注意力 KV 变体（MHA/MQA/GQA/MLA）。

**Architecture:** 复用现有「按 demo 绑 Viz + 帧播放器」。可测的纯数学（激活函数值/导数、KV-cache 公式）抽到 `algorithms/arch/*-fns.ts` 并用 vitest 单测；builder 产出 1 帧 Trajectory 携带 tutorial；交互（变体切换/滑块）放在 Viz 组件内部。论文引用用 `data/papers.ts` + `components/PaperRefs.tsx`，在实验面板里展示。

**Tech Stack:** React + Vite + TypeScript、recharts、tailwind、vitest（新增）。所在目录：`/Users/xujunshan/Code/ml-classics-lab/rl-lab`，git 仓库根：`/Users/xujunshan/Code/ml-classics-lab`，分支 `feat/llm-architecture-anatomy`。

> 约定：下文相对路径若以 `rl-lab/` 开头则相对仓库根；命令默认在 `rl-lab/` 下执行。spec：`docs/superpowers/specs/2026-06-09-llm-architecture-anatomy-design.md`。

---

## File Structure（Batch 1 涉及）

- Create `rl-lab/vitest.config.ts` — vitest 配置（node 环境 + tsconfig 路径别名）
- Modify `rl-lab/package.json` — 加 `test` 脚本 + vitest 依赖
- Modify `rl-lab/src/player/types.ts` — Family 联合类型加 8 个 arch family 名
- Create `rl-lab/src/data/papers.ts` — 8 个实验的论文引用表（全量数据）
- Create `rl-lab/src/components/PaperRefs.tsx` — 参考文献区块组件
- Create `rl-lab/src/visualizers/arch/types.ts` — arch 实验 state 形状
- Create `rl-lab/src/algorithms/arch/activation-fns.ts` — 激活函数纯实现（+导数）
- Create `rl-lab/src/algorithms/arch/activation-fns.test.ts` — 单测
- Create `rl-lab/src/algorithms/arch/activations.ts` — 激活实验 builder
- Create `rl-lab/src/visualizers/arch/ActivationViz.tsx` — 激活可视化器
- Create `rl-lab/src/algorithms/arch/kv-cache.ts` — KV-cache 公式纯实现
- Create `rl-lab/src/algorithms/arch/kv-cache.test.ts` — 单测
- Create `rl-lab/src/algorithms/arch/attention-kv.ts` — 注意力KV builder
- Create `rl-lab/src/visualizers/arch/AttnKVViz.tsx` — 注意力KV可视化器
- Modify `rl-lab/src/data/backgrounds.ts` — 加 `activations`、`attention-kv` 现实意义
- Modify `rl-lab/src/pages/AlgorithmLab.tsx` — import、2 条 DEMO、TREE 新枝、PAPERS 面板

---

## Task 1: 加 vitest 测试基建

**Files:**
- Create: `rl-lab/vitest.config.ts`
- Modify: `rl-lab/package.json`

- [ ] **Step 1: 安装 vitest**

Run（在 `rl-lab/` 下）:
```bash
npm i -D vitest
```
Expected: 安装成功，`package.json` devDependencies 出现 `vitest`。

- [ ] **Step 2: 写 vitest 配置（复用 @ 别名）**

Create `rl-lab/vitest.config.ts`:
```ts
import { defineConfig } from "vitest/config";
import tsconfigPaths from "vite-tsconfig-paths";

export default defineConfig({
  plugins: [tsconfigPaths()],
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
```

- [ ] **Step 3: 加 test 脚本**

Modify `rl-lab/package.json` 的 `"scripts"`，新增一行（放在 `"check"` 后）:
```json
"test": "vitest run"
```

- [ ] **Step 4: 冒烟验证 vitest 能跑（暂无测试文件时应优雅退出）**

Run: `npm test`
Expected: vitest 启动，提示 "No test files found"（退出码 0 或 1 均可，关键是 vitest 本身能起来，不报配置错误）。后续任务会加入真实测试。

- [ ] **Step 5: Commit**

```bash
git add rl-lab/package.json rl-lab/package-lock.json rl-lab/vitest.config.ts
git commit -m "chore(rl-lab): add vitest for arch math unit tests"
```

---

## Task 2: 扩展 Family 联合类型

**Files:**
- Modify: `rl-lab/src/player/types.ts`（`export type Family` 联合）

- [ ] **Step 1: 给 Family 加 8 个 arch family 名**

在 `rl-lab/src/player/types.ts` 里，把 `export type Family =` 的联合末尾（`| "versions";` 之前那一项之后）追加：
```ts
  | "activation"
  | "attn-kv"
  | "pos-encoding"
  | "normalization"
  | "sparse-attn"
  | "linear-seq"
  | "moe"
  | "residual"
```
（family 仅用于类型标注，不参与分发；此处一次加全 8 个，后续批次不再改本文件。）

- [ ] **Step 2: 类型检查**

Run: `npm run check`
Expected: PASS（无类型错误）。

- [ ] **Step 3: Commit**

```bash
git add rl-lab/src/player/types.ts
git commit -m "feat(rl-lab): add arch families to Family union"
```

---

## Task 3: 论文引用机制（papers.ts + PaperRefs + 面板接入）

**Files:**
- Create: `rl-lab/src/data/papers.ts`
- Create: `rl-lab/src/components/PaperRefs.tsx`
- Modify: `rl-lab/src/pages/AlgorithmLab.tsx`

- [ ] **Step 1: 写 papers.ts（全量 8 实验引用数据）**

Create `rl-lab/src/data/papers.ts`:
```ts
// 每个实验涉及的参考论文（用户提供，原样挂链接）。
// note 用于标注「按公开描述近似演示」的极新/未公开论文。
export interface PaperRef {
  name: string;
  url: string;
  note?: string;
}

export const PAPERS: Record<string, PaperRef[]> = {
  activations: [
    { name: "GELU", url: "https://arxiv.org/abs/1606.08415" },
    { name: "SiLU", url: "https://arxiv.org/abs/1702.03118" },
    { name: "Swish", url: "https://arxiv.org/abs/1710.05941v1" },
  ],
  "attention-kv": [
    { name: "MHA (Attention Is All You Need)", url: "https://arxiv.org/abs/1706.03762v7" },
    { name: "MQA", url: "https://arxiv.org/abs/1911.02150" },
    { name: "GQA", url: "https://arxiv.org/abs/2305.13245" },
    { name: "MLA (DeepSeek-V2)", url: "https://arxiv.org/abs/2405.04434" },
  ],
  "pos-encoding": [
    { name: "正弦 PE", url: "https://arxiv.org/abs/1706.03762v7" },
    { name: "RoPE", url: "https://arxiv.org/abs/2104.09864" },
    { name: "NoPE", url: "https://arxiv.org/abs/2305.19466" },
    { name: "YaRN", url: "https://arxiv.org/abs/2309.00071" },
  ],
  normalization: [
    { name: "LayerNorm", url: "https://arxiv.org/abs/1607.06450" },
    { name: "RMSNorm", url: "https://arxiv.org/abs/1910.07467" },
    { name: "QK/KVNorm", url: "https://arxiv.org/abs/2010.04245" },
    { name: "Pre/Post-Norm", url: "https://arxiv.org/abs/2002.04745" },
  ],
  "sparse-attention": [
    { name: "Sparse Transformer", url: "https://arxiv.org/abs/1904.10509" },
    { name: "SWA (Longformer)", url: "https://arxiv.org/abs/2004.05150" },
    { name: "Gemma2 SWA", url: "https://arxiv.org/abs/2408.00118" },
    { name: "DSA", url: "https://arxiv.org/abs/2512.02556", note: "极新论文，按公开描述近似演示" },
    { name: "CSA/HCA (DeepSeek-V4)", url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf", note: "未公开，按公开描述近似演示" },
  ],
  "linear-seq": [
    { name: "Lightning Attention", url: "https://arxiv.org/abs/2401.04658" },
    { name: "Gated DeltaNet", url: "https://arxiv.org/abs/2412.06464" },
    { name: "KDA", url: "https://arxiv.org/abs/2510.26692" },
    { name: "Mamba", url: "https://arxiv.org/abs/2312.00752" },
    { name: "Mamba-2", url: "https://arxiv.org/abs/2405.21060" },
    { name: "Mamba-3", url: "https://arxiv.org/abs/2603.15569", note: "极新论文，按公开描述近似演示" },
  ],
  moe: [
    { name: "MoE (Sparsely-Gated)", url: "https://arxiv.org/abs/1701.06538" },
    { name: "Switch Transformer", url: "https://arxiv.org/abs/2101.03961" },
    { name: "DeepSeekMoE", url: "https://arxiv.org/abs/2401.06066" },
  ],
  residual: [
    { name: "RC (ResNet)", url: "https://arxiv.org/abs/1512.03385" },
    { name: "HC", url: "https://arxiv.org/abs/2409.19606" },
    { name: "mHC", url: "https://arxiv.org/abs/2512.24880", note: "极新论文，按公开描述近似演示" },
    { name: "AttnResidual", url: "https://arxiv.org/abs/2603.15031", note: "极新论文，按公开描述近似演示" },
  ],
};
```

- [ ] **Step 2: 写 PaperRefs 组件**

Create `rl-lab/src/components/PaperRefs.tsx`:
```tsx
import { PaperRef } from "@/data/papers";

export default function PaperRefs({ refs }: { refs: PaperRef[] }) {
  return (
    <div className="mt-4 rounded-lg border border-slate-700/40 bg-slate-800/30 p-3">
      <div className="text-xs font-semibold text-slate-300 mb-2">📄 参考文献</div>
      <ul className="flex flex-col gap-1.5">
        {refs.map((r) => (
          <li key={r.url} className="text-[12px] leading-snug">
            <a
              href={r.url}
              target="_blank"
              rel="noreferrer"
              className="text-sky-400 hover:underline font-mono"
            >
              {r.name}
            </a>
            {r.note && <span className="ml-2 text-amber-400/80">· {r.note}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}
```

- [ ] **Step 3: 面板接入 PaperRefs**

在 `rl-lab/src/pages/AlgorithmLab.tsx` 顶部 import 区（`BackgroundPanel` import 之后）加：
```ts
import PaperRefs from "@/components/PaperRefs";
import { PAPERS } from "@/data/papers";
```
然后在渲染处，把 BackgroundPanel 那一行（`{BACKGROUNDS[demoKey] && <BackgroundPanel bg={BACKGROUNDS[demoKey]} />}`）下一行新增：
```tsx
            {PAPERS[demoKey] && <PaperRefs refs={PAPERS[demoKey]} />}
```

- [ ] **Step 4: 类型检查**

Run: `npm run check`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add rl-lab/src/data/papers.ts rl-lab/src/components/PaperRefs.tsx rl-lab/src/pages/AlgorithmLab.tsx
git commit -m "feat(rl-lab): paper citation mechanism for arch section"
```

---

## Task 4: arch state 类型（Batch 1 部分）

**Files:**
- Create: `rl-lab/src/visualizers/arch/types.ts`

- [ ] **Step 1: 写 Batch 1 的 state 形状**

Create `rl-lab/src/visualizers/arch/types.ts`:
```ts
// 「LLM 架构解剖」板块各实验的 state 形状（按需逐批扩充）。

// —— 激活函数 ——
export type ActivationVariant = "relu" | "gelu" | "silu" | "swish" | "swiglu";
export interface ActivationState {
  xs: number[]; // 采样横坐标（如 [-6,6] 等距 241 点）
}

// —— 注意力 KV 变体 ——
export type KVVariant = "mha" | "mqa" | "gqa" | "mla";
export interface AttnKVConfig {
  nHeads: number; // Q 头数
  nGroups: number; // GQA 分组数（KV 头数）
  dHead: number; // 每头维度
  seqLen: number; // 序列长度（算 KV-cache）
  latentDim: number; // MLA 压缩潜维
}
export interface AttnKVState {
  config: AttnKVConfig;
}
```

- [ ] **Step 2: 类型检查**

Run: `npm run check`
Expected: PASS。

- [ ] **Step 3: Commit**

```bash
git add rl-lab/src/visualizers/arch/types.ts
git commit -m "feat(rl-lab): arch state types (batch 1)"
```

---

## Task 5: 激活函数纯实现 + 单测

**Files:**
- Create: `rl-lab/src/algorithms/arch/activation-fns.ts`
- Test: `rl-lab/src/algorithms/arch/activation-fns.test.ts`

- [ ] **Step 1: 写失败的测试**

Create `rl-lab/src/algorithms/arch/activation-fns.test.ts`:
```ts
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `npm test`
Expected: FAIL（`activation-fns.ts` 不存在 / 函数未定义）。

- [ ] **Step 3: 写最小实现**

Create `rl-lab/src/algorithms/arch/activation-fns.ts`:
```ts
// 激活函数与其导数（纯函数，供 builder 与 Viz 复用、单测覆盖）。
const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

export const relu = (x: number): number => (x > 0 ? x : 0);
export const dRelu = (x: number): number => (x > 0 ? 1 : 0);

// SiLU(x) = x * sigmoid(x)
export const silu = (x: number): number => x * sigmoid(x);
export const dSilu = (x: number): number => {
  const s = sigmoid(x);
  return s + x * s * (1 - s);
};

// Swish(x) = x * sigmoid(beta*x)（beta=1 即 SiLU）
export const swish = (x: number, beta = 1): number => x * sigmoid(beta * x);
export const dSwish = (x: number, beta = 1): number => {
  const s = sigmoid(beta * x);
  return s + beta * x * s * (1 - s);
};

// GELU（tanh 近似，与原论文一致）
export const gelu = (x: number): number =>
  0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
export const dGelu = (x: number): number => {
  const h = 1e-5;
  return (gelu(x + h) - gelu(x - h)) / (2 * h);
};
```

- [ ] **Step 4: 运行测试确认通过**

Run: `npm test`
Expected: PASS（全部用例绿）。

- [ ] **Step 5: Commit**

```bash
git add rl-lab/src/algorithms/arch/activation-fns.ts rl-lab/src/algorithms/arch/activation-fns.test.ts
git commit -m "feat(rl-lab): activation functions + unit tests"
```

---

## Task 6: 激活实验 builder + Viz + 接入

**Files:**
- Create: `rl-lab/src/algorithms/arch/activations.ts`
- Create: `rl-lab/src/visualizers/arch/ActivationViz.tsx`
- Modify: `rl-lab/src/data/backgrounds.ts`
- Modify: `rl-lab/src/pages/AlgorithmLab.tsx`

- [ ] **Step 1: 写 builder（含 tutorial）**

Create `rl-lab/src/algorithms/arch/activations.ts`:
```ts
import { Trajectory } from "@/player/types";
import { ActivationState } from "@/visualizers/arch/types";

export function runActivations(): Trajectory<ActivationState> {
  const N = 241;
  const xs: number[] = [];
  for (let i = 0; i < N; i++) xs.push(-6 + (12 * i) / (N - 1));
  return {
    meta: {
      id: "activations",
      title: "激活函数 · ReLU→GELU→SiLU→Swish→SwiGLU",
      family: "activation",
      algorithm: "Activations",
      description: "现代 LLM 的 FFN 用什么激活？对比曲线、导数与门控。",
      tutorial: {
        problem: "神经网络要非线性才能拟合复杂函数——用哪个非线性？",
        intuition:
          "ReLU 简单但负区死掉（梯度为 0）。GELU/SiLU/Swish 在 0 附近平滑、负区有轻微下凹，训练更稳、表达更强。SwiGLU 把激活做成「门控」：一路当开关乘到另一路上，是现代 LLM（LLaMA 等）FFN 的主流。",
        watch: [
          "切换变体，看曲线在负区的差别：ReLU 硬截断，其余平滑下凹",
          "看导数曲线：ReLU 导数是阶跃，平滑激活的导数连续",
          "拖动 Swish 的 β：β→0 趋近线性，β→大 趋近 ReLU",
          "SwiGLU 面板：门控如何把一路压制/放大另一路",
        ],
        concepts: [
          { term: "ReLU", explain: "max(0,x)，最简单的非线性，负区梯度为 0" },
          { term: "SiLU/Swish", explain: "x·sigmoid(βx)，平滑、负区有下凹，β=1 时 SiLU=Swish" },
          { term: "GELU", explain: "用高斯 CDF 加权输入，BERT/GPT 常用" },
          { term: "SwiGLU", explain: "门控线性单元：SiLU(xW)·(xV)，现代 LLM FFN 主流" },
        ],
        tryThis: "把 β 从 1 调到 5，观察 Swish 怎样越来越像 ReLU。",
      },
    },
    frames: [{ iter: 0, state: { xs }, metrics: {} }],
  };
}
```

- [ ] **Step 2: 写 ActivationViz（变体按钮 + β 滑块 + 曲线/导数 + SwiGLU 门控示意）**

Create `rl-lab/src/visualizers/arch/ActivationViz.tsx`:
```tsx
import { useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";
import { ActivationState, ActivationVariant } from "./types";
import { relu, dRelu, silu, dSilu, swish, dSwish, gelu, dGelu } from "@/algorithms/arch/activation-fns";

const VARIANTS: { key: ActivationVariant; label: string }[] = [
  { key: "relu", label: "ReLU" },
  { key: "gelu", label: "GELU" },
  { key: "silu", label: "SiLU" },
  { key: "swish", label: "Swish(β)" },
  { key: "swiglu", label: "SwiGLU" },
];

function val(v: ActivationVariant, x: number, beta: number): number {
  switch (v) {
    case "relu": return relu(x);
    case "gelu": return gelu(x);
    case "silu": return silu(x);
    case "swish": return swish(x, beta);
    case "swiglu": return silu(x); // 门控的「激活支」用 SiLU；门控演示见下方面板
  }
}
function deriv(v: ActivationVariant, x: number, beta: number): number {
  switch (v) {
    case "relu": return dRelu(x);
    case "gelu": return dGelu(x);
    case "silu": return dSilu(x);
    case "swish": return dSwish(x, beta);
    case "swiglu": return dSilu(x);
  }
}

export default function ActivationViz({ state }: { state: ActivationState }) {
  const [variant, setVariant] = useState<ActivationVariant>("silu");
  const [beta, setBeta] = useState(1);
  const xs = state.xs;

  const data = useMemo(
    () => xs.map((x) => ({ x: Number(x.toFixed(3)), y: val(variant, x, beta), dy: deriv(variant, x, beta) })),
    [xs, variant, beta],
  );

  return (
    <div className="flex flex-col items-center gap-4 w-full">
      <div className="flex flex-wrap items-center gap-2">
        {VARIANTS.map((v) => (
          <button
            key={v.key}
            onClick={() => setVariant(v.key)}
            className={`px-3 py-1 rounded text-xs font-mono border transition-all ${
              variant === v.key
                ? "bg-sky-500/25 border-sky-400 text-sky-200"
                : "bg-slate-800/40 border-slate-600 text-slate-400 hover:border-slate-400"
            }`}
          >
            {v.label}
          </button>
        ))}
      </div>

      {(variant === "swish") && (
        <label className="flex items-center gap-3 text-xs text-slate-400">
          β = {beta.toFixed(2)}
          <input type="range" min={0.1} max={5} step={0.1} value={beta}
            onChange={(e) => setBeta(Number(e.target.value))} className="w-48" />
        </label>
      )}

      <div className="w-full" style={{ height: 280 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="x" type="number" domain={[-6, 6]} stroke="#64748b" tick={{ fontSize: 11 }} />
            <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={[-2, 6]} />
            <ReferenceLine x={0} stroke="#475569" />
            <ReferenceLine y={0} stroke="#475569" />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="y" name="激活 f(x)" stroke="#38bdf8" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="dy" name="导数 f'(x)" stroke="#f59e0b" dot={false} strokeWidth={1.5} strokeDasharray="4 3" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {variant === "swiglu" && (
        <div className="text-[12px] text-slate-400 leading-relaxed max-w-xl text-center">
          <span className="font-mono text-sky-300">SwiGLU(x) = SiLU(x·W) ⊙ (x·V)</span>
          ：把输入投影成两路，一路过 SiLU 当「门」，逐元素乘到另一「值」路上——
          门接近 0 就压制、接近线性就放行。现代 LLM（LLaMA 系）FFN 的标配。
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: 加 background 文案**

在 `rl-lab/src/data/backgrounds.ts` 的 `BACKGROUNDS` 对象里（任意位置，建议挨着 `mlp`）新增：
```ts
  activations: {
    realWorld:
      "激活函数是神经网络非线性的来源。从 ReLU 到 GELU/SiLU/Swish 再到门控的 SwiGLU，这条演进线直接决定了现代 LLM 的 FFN 长什么样、训练稳不稳。",
    uses: ["大模型 FFN 设计", "训练稳定性", "梯度流动", "深度网络基础"],
  },
```

- [ ] **Step 4: 接入 AlgorithmLab（import + DEMO + TREE 新枝）**

在 `rl-lab/src/pages/AlgorithmLab.tsx`：

(a) import 区加：
```ts
import { runActivations } from "@/algorithms/arch/activations";
import ActivationViz from "@/visualizers/arch/ActivationViz";
```

(b) `DEMOS` 数组末尾（最后一个 `},` 与 `];` 之间）加：
```ts
  {
    key: "activations",
    label: "激活函数",
    group: "LLM 架构解剖",
    build: () => runActivations(),
    Viz: ActivationViz,
    metricKey: "noop",
    metricLabel: "（交互演示 · 无训练曲线）",
    metricColor: "#38bdf8",
  },
```

(c) `TREE` 数组末尾（最后一个顶级 `},` 与 `];` 之间）加新顶级枝：
```ts
  {
    label: "LLM 架构解剖（Transformer 内部）",
    children: [
      { label: "FFN · 激活", children: [{ key: "activations" }] },
    ],
  },
```

- [ ] **Step 5: 类型检查 + 构建**

Run: `npm run check && npm run build`
Expected: 均 PASS（无类型错误、构建成功）。

- [ ] **Step 6: Commit**

```bash
git add rl-lab/src/algorithms/arch/activations.ts rl-lab/src/visualizers/arch/ActivationViz.tsx rl-lab/src/data/backgrounds.ts rl-lab/src/pages/AlgorithmLab.tsx
git commit -m "feat(rl-lab): activations comparison experiment"
```

---

## Task 7: KV-cache 纯实现 + 单测

**Files:**
- Create: `rl-lab/src/algorithms/arch/kv-cache.ts`
- Test: `rl-lab/src/algorithms/arch/kv-cache.test.ts`

- [ ] **Step 1: 写失败的测试**

Create `rl-lab/src/algorithms/arch/kv-cache.test.ts`:
```ts
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `npm test`
Expected: FAIL（`kv-cache.ts` 不存在）。

- [ ] **Step 3: 写最小实现**

Create `rl-lab/src/algorithms/arch/kv-cache.ts`:
```ts
import { KVVariant, AttnKVConfig } from "@/visualizers/arch/types";

// 各变体的 KV 头数（MLA 用压缩潜向量，不以「头数」计，返回 1 仅作占位）。
export function kvHeads(variant: KVVariant, cfg: AttnKVConfig): number {
  switch (variant) {
    case "mha": return cfg.nHeads;
    case "mqa": return 1;
    case "gqa": return cfg.nGroups;
    case "mla": return 1;
  }
}

// 每个 token 需缓存的元素数：K、V 各占 kvHeads*dHead；MLA 只缓存一个 latentDim 向量。
export function cacheElemsPerToken(variant: KVVariant, cfg: AttnKVConfig): number {
  if (variant === "mla") return cfg.latentDim;
  return 2 * kvHeads(variant, cfg) * cfg.dHead;
}

// 整条序列的 KV-cache 总元素数。
export function cacheTotal(variant: KVVariant, cfg: AttnKVConfig): number {
  return cfg.seqLen * cacheElemsPerToken(variant, cfg);
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `npm test`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add rl-lab/src/algorithms/arch/kv-cache.ts rl-lab/src/algorithms/arch/kv-cache.test.ts
git commit -m "feat(rl-lab): KV-cache formulas + unit tests"
```

---

## Task 8: 注意力KV builder + Viz + 接入

**Files:**
- Create: `rl-lab/src/algorithms/arch/attention-kv.ts`
- Create: `rl-lab/src/visualizers/arch/AttnKVViz.tsx`
- Modify: `rl-lab/src/data/backgrounds.ts`
- Modify: `rl-lab/src/pages/AlgorithmLab.tsx`

- [ ] **Step 1: 写 builder（含 tutorial）**

Create `rl-lab/src/algorithms/arch/attention-kv.ts`:
```ts
import { Trajectory } from "@/player/types";
import { AttnKVState } from "@/visualizers/arch/types";

export function runAttentionKV(): Trajectory<AttnKVState> {
  return {
    meta: {
      id: "attention-kv",
      title: "注意力 KV 变体 · MHA→MQA→GQA→MLA",
      family: "attn-kv",
      algorithm: "Attention KV sharing",
      description: "推理时 KV-cache 是显存大头。如何在不太掉效果下把它缩小？",
      tutorial: {
        problem: "自回归推理要缓存每个 token 的 K/V（KV-cache），长上下文时显存爆炸。",
        intuition:
          "MHA 每个查询头都有自己的一份 K/V，最准但最占显存。MQA 让所有头共享 1 份 K/V，省到极致但可能掉点。GQA 折中：分几组、组内共享。MLA（DeepSeek）更进一步：把 K/V 压成一个低秩潜向量缓存、用时再上投，显存最省且尽量保效果。",
        watch: [
          "切换 MHA/MQA/GQA/MLA，看 Q 头与 KV 头的连线如何「合并共享」",
          "右侧 KV-cache 显存条：从 MHA 到 MLA 一路缩小",
          "拖动头数/分组数，观察 GQA 在 MHA 与 MQA 之间连续过渡",
        ],
        concepts: [
          { term: "KV-cache", explain: "自回归推理缓存的历史 K/V，长度随上下文线性增长" },
          { term: "MHA", explain: "Multi-Head：每个 Q 头独立 K/V，质量高、显存大" },
          { term: "MQA", explain: "Multi-Query：所有 Q 头共享 1 份 K/V，显存最省" },
          { term: "GQA", explain: "Grouped-Query：分组共享 K/V，质量/显存折中" },
          { term: "MLA", explain: "Multi-head Latent：K/V 压成低秩潜向量缓存，再上投复原" },
        ],
        tryThis: "把分组数从 1 调到 8，看 GQA 如何在 MQA 和 MHA 两端之间滑动。",
      },
    },
    frames: [
      {
        iter: 0,
        state: { config: { nHeads: 8, nGroups: 2, dHead: 64, seqLen: 1024, latentDim: 64 } },
        metrics: {},
      },
    ],
  };
}
```

- [ ] **Step 2: 写 AttnKVViz（变体按钮 + 头数/组数/序列长滑块 + 头映射 SVG + KV-cache 对比条）**

Create `rl-lab/src/visualizers/arch/AttnKVViz.tsx`:
```tsx
import { useState } from "react";
import { AttnKVState, KVVariant } from "./types";
import { kvHeads, cacheTotal } from "@/algorithms/arch/kv-cache";

const VARIANTS: { key: KVVariant; label: string }[] = [
  { key: "mha", label: "MHA" },
  { key: "mqa", label: "MQA" },
  { key: "gqa", label: "GQA" },
  { key: "mla", label: "MLA" },
];

export default function AttnKVViz({ state }: { state: AttnKVState }) {
  const [variant, setVariant] = useState<KVVariant>("gqa");
  const [nHeads, setNHeads] = useState(state.config.nHeads);
  const [nGroups, setNGroups] = useState(state.config.nGroups);
  const [seqLen, setSeqLen] = useState(state.config.seqLen);
  const cfg = { ...state.config, nHeads, nGroups, seqLen };

  // 每个 Q 头映射到哪个 KV 头（索引）
  const kvOf = (qi: number): number => {
    switch (variant) {
      case "mha": return qi;
      case "mqa": return 0;
      case "gqa": return Math.floor(qi / Math.ceil(nHeads / Math.max(1, nGroups)));
      case "mla": return 0; // 潜向量，单独画
    }
  };
  const nKV = kvHeads(variant, cfg);

  // KV-cache 对比（相对 MHA 的百分比）
  const totals = VARIANTS.map((v) => ({ key: v.key, label: v.label, total: cacheTotal(v.key, cfg) }));
  const maxTotal = Math.max(...totals.map((t) => t.total));

  const W = 460, qY = 40, kvY = 200, headW = 34, gap = 12;
  const qX = (i: number) => 30 + i * (headW + gap);
  const kvX = (i: number) => 30 + i * (headW + gap) + (variant === "mqa" || variant === "mla" ? (W - 60) / 2 - headW / 2 : 0);

  return (
    <div className="flex flex-col items-center gap-5 w-full">
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
        <label className="flex items-center gap-2">Q 头数 {nHeads}
          <input type="range" min={2} max={12} step={1} value={nHeads}
            onChange={(e) => setNHeads(Number(e.target.value))} className="w-32" />
        </label>
        {variant === "gqa" && (
          <label className="flex items-center gap-2">分组数 {nGroups}
            <input type="range" min={1} max={nHeads} step={1} value={Math.min(nGroups, nHeads)}
              onChange={(e) => setNGroups(Number(e.target.value))} className="w-32" />
          </label>
        )}
        <label className="flex items-center gap-2">序列长 {seqLen}
          <input type="range" min={256} max={8192} step={256} value={seqLen}
            onChange={(e) => setSeqLen(Number(e.target.value))} className="w-32" />
        </label>
      </div>

      {/* Q→KV 头映射示意 */}
      <svg width={W} height={260} className="max-w-full">
        {variant !== "mla" &&
          Array.from({ length: nHeads }).map((_, qi) => {
            const kx = kvX(kvOf(qi)) + headW / 2;
            return <line key={qi} x1={qX(qi) + headW / 2} y1={qY + 26} x2={kx} y2={kvY} stroke="#475569" strokeWidth={1} />;
          })}
        {variant === "mla" &&
          Array.from({ length: nHeads }).map((_, qi) => (
            <line key={qi} x1={qX(qi) + headW / 2} y1={qY + 26} x2={W / 2} y2={kvY} stroke="#a855f7" strokeWidth={1} />
          ))}
        {Array.from({ length: nHeads }).map((_, qi) => (
          <g key={qi}>
            <rect x={qX(qi)} y={qY} width={headW} height={26} rx={4} fill="#0ea5e9" opacity={0.8} />
            <text x={qX(qi) + headW / 2} y={qY + 17} textAnchor="middle" fontSize={10} fill="#fff">Q{qi}</text>
          </g>
        ))}
        {variant !== "mla" &&
          Array.from({ length: nKV }).map((_, ki) => (
            <g key={ki}>
              <rect x={kvX(ki)} y={kvY} width={headW} height={26} rx={4} fill="#f59e0b" opacity={0.85} />
              <text x={kvX(ki) + headW / 2} y={kvY + 17} textAnchor="middle" fontSize={10} fill="#fff">KV{ki}</text>
            </g>
          ))}
        {variant === "mla" && (
          <g>
            <rect x={W / 2 - headW} y={kvY} width={headW * 2} height={26} rx={4} fill="#a855f7" opacity={0.85} />
            <text x={W / 2} y={kvY + 17} textAnchor="middle" fontSize={10} fill="#fff">潜向量 latent</text>
          </g>
        )}
        <text x={30} y={qY - 8} fontSize={11} fill="#94a3b8">查询头 Query heads</text>
        <text x={30} y={kvY - 8} fontSize={11} fill="#94a3b8">KV 头（缓存）</text>
      </svg>

      {/* KV-cache 显存对比条 */}
      <div className="w-full max-w-md flex flex-col gap-2">
        <div className="text-xs text-slate-400">KV-cache 显存（相对最大值）· 序列长 {seqLen}</div>
        {totals.map((t) => (
          <div key={t.key} className="flex items-center gap-2">
            <span className={`w-12 text-[11px] font-mono ${t.key === variant ? "text-sky-300" : "text-slate-500"}`}>{t.label}</span>
            <div className="flex-1 h-4 bg-slate-800 rounded overflow-hidden">
              <div className="h-full rounded" style={{ width: `${(t.total / maxTotal) * 100}%`, background: t.key === variant ? "#38bdf8" : "#475569" }} />
            </div>
            <span className="w-16 text-right text-[11px] font-mono text-slate-400">{(t.total / 1e6).toFixed(2)}M</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 加 background 文案**

在 `rl-lab/src/data/backgrounds.ts` 的 `BACKGROUNDS` 里新增：
```ts
  "attention-kv": {
    realWorld:
      "长上下文推理时，KV-cache 是显存瓶颈。MQA/GQA/MLA 这条线就是工业界为了「把大模型推理跑得起、跑得便宜」一步步压缩 KV-cache 的真实演进——DeepSeek 的 MLA 是其中的关键一招。",
    uses: ["大模型推理优化", "显存/成本控制", "长上下文", "服务化部署"],
  },
```

- [ ] **Step 4: 接入 AlgorithmLab（import + DEMO + TREE 子节点）**

在 `rl-lab/src/pages/AlgorithmLab.tsx`：

(a) import 区加：
```ts
import { runAttentionKV } from "@/algorithms/arch/attention-kv";
import AttnKVViz from "@/visualizers/arch/AttnKVViz";
```

(b) `DEMOS` 数组里、`activations` 那条之后加：
```ts
  {
    key: "attention-kv",
    label: "注意力 KV 变体",
    group: "LLM 架构解剖",
    build: () => runAttentionKV(),
    Viz: AttnKVViz,
    metricKey: "noop",
    metricLabel: "（交互演示 · 无训练曲线）",
    metricColor: "#38bdf8",
  },
```

(c) `TREE` 里新枝「LLM 架构解剖（Transformer 内部）」的 children 加一项：
```ts
      { label: "注意力 · KV 共享", children: [{ key: "attention-kv" }] },
```
（即该枝 children 变为 `[{ label:"FFN · 激活", ... }, { label:"注意力 · KV 共享", children:[{key:"attention-kv"}] }]`。）

- [ ] **Step 5: 类型检查 + 构建 + 全量测试**

Run: `npm run check && npm test && npm run build`
Expected: 三者均 PASS。

- [ ] **Step 6: Commit**

```bash
git add rl-lab/src/algorithms/arch/attention-kv.ts rl-lab/src/visualizers/arch/AttnKVViz.tsx rl-lab/src/data/backgrounds.ts rl-lab/src/pages/AlgorithmLab.tsx
git commit -m "feat(rl-lab): attention KV variants comparison experiment"
```

---

## Task 9: 端到端人工验证

**Files:** 无（仅运行）

- [ ] **Step 1: 起 dev 并人工检查**

Run: `npm run dev`（已在 5180）。浏览器打开 http://localhost:5180/
逐项确认：
- 左侧树出现新枝「LLM 架构解剖（Transformer 内部）」，含「FFN · 激活」「注意力 · KV 共享」。
- 进「激活函数」：变体按钮可切；选 Swish 出现 β 滑块且曲线随 β 变化；导数曲线显示；SwiGLU 出现门控说明；面板底部「📄 参考文献」列出 GELU/SiLU/Swish 链接。
- 进「注意力 KV 变体」：切 MHA/MQA/GQA/MLA，Q→KV 连线随之合并；GQA 出现分组滑块；KV-cache 对比条从 MHA 到 MLA 递减；参考文献列出 MHA/MQA/GQA/MLA。
- 切换两个实验来回若干次，无白屏/崩溃（state 与 Viz 按 demoKey 匹配）。

- [ ] **Step 2: 记录结果**

确认无误后在此打勾；若有问题，回到对应 Task 修复并重跑该 Task 的验证步骤。

---

## Self-Review（已对照 spec）

- **Spec 覆盖**：本批落地 spec §3 表中 #7 激活、#3 注意力KV，并搭好 §5 论文引用机制、§3 顶级树枝、§4 文件结构骨架、§9 成功标准里的 tsc/build。其余 6 个实验 + §6 真实稀疏切片留给 Batch 2-4（各自单独计划）。
- **占位符扫描**：无 TBD/TODO；所有代码步骤含完整代码。
- **类型一致性**：`ActivationState`/`ActivationVariant`/`AttnKVState`/`AttnKVConfig`/`KVVariant` 在 `arch/types.ts` 定义，被 builder、Viz、`activation-fns`、`kv-cache` 一致引用；`kvHeads`/`cacheElemsPerToken`/`cacheTotal` 命名在测试与实现一致；`relu/gelu/silu/swish/dRelu/dSilu/dSwish/dGelu` 命名一致。
- **family**：`activation`、`attn-kv` 已在 Task 2 加入 Family 联合。

## 后续批次（不在本计划内，确认 Batch 1 模式后各自出计划）

- Batch 2：位置编码(1) + 归一化(2)
- Batch 3：MoE(6) + 残差(8)
- Batch 4：线性·SSM(5) + 稀疏注意力(4，含 `solvers/arch_sparse_attention.py` 本地 HF 真实切片)
