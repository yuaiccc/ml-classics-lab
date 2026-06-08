# LLM 架构解剖 · Batch 2（位置编码 + 归一化）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 在「LLM 架构解剖」板块再加两个对比实验：位置编码（None/正弦PE/RoPE/NoPE/YaRN）与归一化（LayerNorm/RMSNorm/QKNorm/Pre·Post-Norm）。

**Architecture:** 沿用 Batch 1 模式：可测纯数学抽到 `algorithms/arch/*-fns.ts` 并 vitest 单测；builder 产出单帧 Trajectory 携带 tutorial；交互（变体切换/滑块）放进 Viz 内部。基建（papers.ts 的 `pos-encoding`/`normalization` 键、Family 联合里的 `pos-encoding`/`normalization`、顶级树枝「LLM 架构解剖（Transformer 内部）」）已在 Batch 1 就位，本批只新增/接线。

**Tech Stack:** React + Vite + TS、recharts、canvas、SVG、tailwind、vitest。所在目录 `/Users/xujunshan/Code/ml-classics-lab/rl-lab`，git 根 `/Users/xujunshan/Code/ml-classics-lab`，分支 `feat/llm-architecture-anatomy`。

> 约定：`npm` 在 `rl-lab/` 下跑；`git` 在仓库根跑（add 路径形如 `rl-lab/...`）。spec：`docs/superpowers/specs/2026-06-09-llm-architecture-anatomy-design.md`。

---

## File Structure（Batch 2）

- Modify `rl-lab/src/visualizers/arch/types.ts` — 加 `PosEncVariant`/`PosEncState`/`NormVariant`/`NormState`
- Create `rl-lab/src/algorithms/arch/pos-encoding-fns.ts` + `.test.ts`
- Create `rl-lab/src/algorithms/arch/pos-encoding.ts`（builder）
- Create `rl-lab/src/visualizers/arch/PosEncodingViz.tsx`
- Create `rl-lab/src/algorithms/arch/normalization-fns.ts` + `.test.ts`
- Create `rl-lab/src/algorithms/arch/normalization.ts`（builder）
- Create `rl-lab/src/visualizers/arch/NormViz.tsx`
- Modify `rl-lab/src/data/backgrounds.ts` — 加 `pos-encoding`、`normalization`
- Modify `rl-lab/src/pages/AlgorithmLab.tsx` — import、2 条 DEMO、TREE 加 2 子节点

---

## Task 1: 扩 arch types

**Files:** Modify `rl-lab/src/visualizers/arch/types.ts`

- [ ] **Step 1: 追加类型**

在 `rl-lab/src/visualizers/arch/types.ts` 末尾追加：
```ts

// —— 位置编码 ——
export type PosEncVariant = "none" | "sinusoidal" | "rope" | "nope" | "yarn";
export interface PosEncState {
  dim: number; // 演示维度（如 32）
  maxPos: number; // 位置轴长度（如 64）
}

// —— 归一化 ——
export type NormVariant = "layernorm" | "rmsnorm" | "qknorm" | "prepost";
export interface NormState {
  vectors: number[][]; // 若干 d 维玩具激活向量（这里 d=3）
}
```

- [ ] **Step 2:** Run `npm run check` → PASS。
- [ ] **Step 3:** Commit:
```bash
git add rl-lab/src/visualizers/arch/types.ts
git commit -m "feat(rl-lab): arch state types (batch 2)"
```

---

## Task 2: 位置编码纯函数 + 单测（TDD）

**Files:** Create `rl-lab/src/algorithms/arch/pos-encoding-fns.ts` + `.test.ts`

- [ ] **Step 1: 写失败测试** — `rl-lab/src/algorithms/arch/pos-encoding-fns.test.ts`：
```ts
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
    // 旋转越慢 → 与原向量越对齐 → 点积越大
    expect(dot(yarnApply(v, pos), v)).toBeGreaterThan(dot(ropeApply(v, pos), v));
  });
});
```

- [ ] **Step 2:** Run `npm test` → expect the new file FAILS (module missing); existing tests still pass.

- [ ] **Step 3: 实现** — `rl-lab/src/algorithms/arch/pos-encoding-fns.ts`：
```ts
// 位置编码的纯数学（供 builder/Viz 复用、单测覆盖）。

export function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// 正弦位置编码：PE(pos,2i)=sin(pos/10000^(2i/d))，PE(pos,2i+1)=cos(...)
export function sinusoidalPE(pos: number, dim: number): number[] {
  const v: number[] = [];
  for (let i = 0; i < dim; i += 2) {
    const freq = 1 / Math.pow(10000, i / dim);
    v.push(Math.sin(pos * freq));
    if (v.length < dim) v.push(Math.cos(pos * freq));
  }
  return v.slice(0, dim);
}

// RoPE：把相邻 2D 对按角度 pos*freq_i 旋转。
export function ropeApply(vec: number[], pos: number, base = 10000): number[] {
  const d = vec.length;
  const out = vec.slice();
  for (let i = 0; i + 1 < d; i += 2) {
    const freq = 1 / Math.pow(base, i / d);
    const theta = pos * freq;
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const x = vec[i];
    const y = vec[i + 1];
    out[i] = x * c - y * s;
    out[i + 1] = x * s + y * c;
  }
  return out;
}

// 单个 2D 对在某位置的旋转角（给 Viz 画 angle-vs-pos 曲线用）。
export function ropeAngle(pairIndex: number, pos: number, dim: number, base = 10000): number {
  const freq = 1 / Math.pow(base, (2 * pairIndex) / dim);
  return pos * freq;
}

// YaRN：用更大的 base 拉长波长→旋转更慢→有效上下文更长（NTK-aware 的简化演示）。
export function yarnApply(vec: number[], pos: number, scale = 8, base = 10000): number[] {
  return ropeApply(vec, pos, base * scale);
}
export function yarnAngle(pairIndex: number, pos: number, dim: number, scale = 8, base = 10000): number {
  return ropeAngle(pairIndex, pos, dim, base * scale);
}
```

- [ ] **Step 4:** Run `npm test` → PASS（全绿；新增用例 + 既有用例）。
- [ ] **Step 5:** Commit:
```bash
git add rl-lab/src/algorithms/arch/pos-encoding-fns.ts rl-lab/src/algorithms/arch/pos-encoding-fns.test.ts
git commit -m "feat(rl-lab): positional encoding math + unit tests"
```

---

## Task 3: 位置编码 builder + Viz + 接线

**Files:** Create `pos-encoding.ts`、`PosEncodingViz.tsx`；Modify `backgrounds.ts`、`AlgorithmLab.tsx`

- [ ] **Step 1: builder** — `rl-lab/src/algorithms/arch/pos-encoding.ts`：
```ts
import { Trajectory } from "@/player/types";
import { PosEncState } from "@/visualizers/arch/types";

export function runPosEncoding(): Trajectory<PosEncState> {
  return {
    meta: {
      id: "pos-encoding",
      title: "位置编码 · PE→RoPE→NoPE→YaRN",
      family: "pos-encoding",
      algorithm: "Positional Encoding",
      description: "注意力本身对顺序无感，位置信息怎么注入？",
      tutorial: {
        problem: "自注意力是「集合」运算，打乱 token 顺序结果不变——可语言是有序的。位置信息怎么加进去？",
        intuition:
          "正弦 PE 给每个位置一串固定的 sin/cos 指纹，直接加到词向量上。RoPE 改成「旋转」：按位置把 Q/K 的二维对转一个角度，巧妙之处是注意力分数 q·k 只依赖两者的相对距离。NoPE 干脆不加，靠因果掩码隐式获得位置感。YaRN 在 RoPE 基础上把旋转频率调慢，让训练在短上下文、推理能外推到长上下文。",
        watch: [
          "正弦 PE：看位置×维度的热力图条纹，高维变化慢、低维变化快",
          "RoPE：拖位置滑块，看二维查询向量被旋转；分数只看相对距离",
          "YaRN：对比 RoPE 与 YaRN 的角度-位置曲线，YaRN 转得更慢→能外推更远",
          "NoPE：完全不加位置，靠因果掩码隐式排序",
        ],
        concepts: [
          { term: "正弦 PE", explain: "不同频率的 sin/cos 给每个位置编码，加到词向量上" },
          { term: "RoPE", explain: "按位置旋转 Q/K，使注意力分数只依赖相对位置" },
          { term: "NoPE", explain: "不显式加位置编码，靠因果掩码获得顺序信息" },
          { term: "YaRN", explain: "调慢 RoPE 频率以把短上下文模型外推到长上下文" },
        ],
        tryThis: "在 RoPE 下拖动位置滑块，观察向量匀速旋转；切到 YaRN 看曲线变缓。",
      },
    },
    frames: [{ iter: 0, state: { dim: 32, maxPos: 64 }, metrics: {} }],
  };
}
```

- [ ] **Step 2: Viz** — `rl-lab/src/visualizers/arch/PosEncodingViz.tsx`：
```tsx
import { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { PosEncState, PosEncVariant } from "./types";
import { sinusoidalPE, ropeApply, ropeAngle, yarnAngle } from "@/algorithms/arch/pos-encoding-fns";

const VARIANTS: { key: PosEncVariant; label: string }[] = [
  { key: "sinusoidal", label: "正弦 PE" },
  { key: "rope", label: "RoPE" },
  { key: "yarn", label: "YaRN" },
  { key: "nope", label: "NoPE" },
];

function Heatmap({ maxPos, dim, curPos }: { maxPos: number; dim: number; curPos: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const cell = 6;
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, dim * cell, maxPos * cell);
    for (let p = 0; p < maxPos; p++) {
      const pe = sinusoidalPE(p, dim);
      for (let i = 0; i < dim; i++) {
        const t = (pe[i] + 1) / 2; // [-1,1]→[0,1]
        const r = Math.round(10 + t * -10);
        const g = Math.round(14 + t * 241);
        const b = Math.round(23 + t * 113);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(i * cell, p * cell, cell, cell);
      }
    }
    // 当前位置高亮行
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.strokeRect(0, curPos * cell, dim * cell, cell);
  }, [maxPos, dim, curPos]);
  return <canvas ref={ref} width={dim * cell} height={maxPos * cell} style={{ maxWidth: "100%", height: "auto" }} />;
}

export default function PosEncodingViz({ state }: { state: PosEncState }) {
  const { dim, maxPos } = state;
  const [variant, setVariant] = useState<PosEncVariant>("rope");
  const [pos, setPos] = useState(8);

  // RoPE：取第一个二维对的单位向量，按位置旋转
  const ropeVec = useMemo(() => {
    const base = new Array(dim).fill(0);
    base[0] = 1; // 第一个二维对的 x
    const rotated = ropeApply(base, pos);
    return { x: rotated[0], y: rotated[1] };
  }, [dim, pos]);

  // YaRN vs RoPE 角度-位置曲线（取第一个二维对）
  const angleData = useMemo(
    () =>
      Array.from({ length: maxPos }, (_, p) => ({
        pos: p,
        rope: ropeAngle(0, p, dim),
        yarn: yarnAngle(0, p, dim),
      })),
    [dim, maxPos],
  );

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

      {(variant === "sinusoidal" || variant === "rope") && (
        <label className="flex items-center gap-3 text-xs text-slate-400">
          位置 pos = {pos}
          <input type="range" min={0} max={maxPos - 1} step={1} value={pos}
            onChange={(e) => setPos(Number(e.target.value))} className="w-56" />
        </label>
      )}

      {variant === "sinusoidal" && (
        <div className="flex flex-col items-center gap-2">
          <div className="text-[11px] text-slate-500 font-mono">位置(纵) × 维度(横) 的 sin/cos 指纹 · 橙框=当前 pos</div>
          <Heatmap maxPos={maxPos} dim={dim} curPos={pos} />
        </div>
      )}

      {variant === "rope" && (
        <div className="flex flex-col items-center gap-2">
          <svg width={220} height={220} className="max-w-full">
            <line x1={110} y1={10} x2={110} y2={210} stroke="#334155" />
            <line x1={10} y1={110} x2={210} y2={110} stroke="#334155" />
            <circle cx={110} cy={110} r={90} fill="none" stroke="#1e293b" />
            <line x1={110} y1={110} x2={110 + ropeVec.x * 90} y2={110 - ropeVec.y * 90}
              stroke="#38bdf8" strokeWidth={3} />
            <circle cx={110 + ropeVec.x * 90} cy={110 - ropeVec.y * 90} r={4} fill="#38bdf8" />
          </svg>
          <div className="text-[12px] text-slate-400 max-w-md text-center">
            第一个二维对的查询向量随位置 <span className="font-mono text-sky-300">pos={pos}</span> 旋转。
            关键：注意力分数 <span className="font-mono">q·k</span> 只依赖 <b>相对距离</b>，与绝对位置无关。
          </div>
        </div>
      )}

      {variant === "yarn" && (
        <div className="w-full" style={{ height: 260 }}>
          <ResponsiveContainer>
            <LineChart data={angleData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="pos" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="rope" name="RoPE 角度" stroke="#38bdf8" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="yarn" name="YaRN 角度（更慢）" stroke="#f59e0b" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
          <div className="text-[12px] text-slate-400 text-center mt-1">
            YaRN 把旋转频率调慢——同样的位置走过更小的角度，于是短上下文训练的模型能外推到更长上下文。
          </div>
        </div>
      )}

      {variant === "nope" && (
        <div className="text-[13px] text-slate-400 leading-relaxed max-w-xl text-center py-8">
          <div className="text-base text-slate-200 mb-2 font-mono">NoPE：不加任何位置编码</div>
          只靠<b>因果掩码</b>（每个 token 只能看到自己左边）就能隐式获得顺序信息。
          研究发现这种纯解码器在一定规模下也能学到位置概念，挑战了「必须显式编码位置」的假设。
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: background** — 在 `rl-lab/src/data/backgrounds.ts` 的 `BACKGROUNDS` 里加：
```ts
  "pos-encoding": {
    realWorld:
      "位置编码决定了大模型能不能处理长文本、能不能把 4K 训练的模型外推到 128K 上下文。RoPE 是当下几乎所有开源大模型（LLaMA/Qwen 等）的标配，YaRN/NTK 则是「长上下文」卖点背后的关键技术。",
    uses: ["长上下文外推", "大模型位置感", "RoPE 工程", "检索/长文档"],
  },
```

- [ ] **Step 4: 接线 AlgorithmLab.tsx**

(a) import：
```ts
import { runPosEncoding } from "@/algorithms/arch/pos-encoding";
import PosEncodingViz from "@/visualizers/arch/PosEncodingViz";
```

(b) 在 DEMOS 里、`attention-kv` 那条之后加：
```ts
  {
    key: "pos-encoding",
    label: "位置编码",
    group: "LLM 架构解剖",
    build: () => runPosEncoding(),
    Viz: PosEncodingViz,
    metricKey: "noop",
    metricLabel: "（交互演示 · 无训练曲线）",
    metricColor: "#38bdf8",
  },
```

(c) 在 TREE 的「LLM 架构解剖（Transformer 内部）」枝 children 末尾加：
```ts
      { label: "位置编码", children: [{ key: "pos-encoding" }] },
```

- [ ] **Step 5:** Run `npm run check && npm run build` → BOTH PASS。
- [ ] **Step 6:** Commit:
```bash
git add rl-lab/src/algorithms/arch/pos-encoding.ts rl-lab/src/visualizers/arch/PosEncodingViz.tsx rl-lab/src/data/backgrounds.ts rl-lab/src/pages/AlgorithmLab.tsx
git commit -m "feat(rl-lab): positional encoding comparison experiment"
```

---

## Task 4: 归一化纯函数 + 单测（TDD）

**Files:** Create `rl-lab/src/algorithms/arch/normalization-fns.ts` + `.test.ts`

- [ ] **Step 1: 写失败测试** — `rl-lab/src/algorithms/arch/normalization-fns.test.ts`：
```ts
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
```

- [ ] **Step 2:** Run `npm test` → new file FAILS (module missing); others pass.

- [ ] **Step 3: 实现** — `rl-lab/src/algorithms/arch/normalization-fns.ts`：
```ts
// 归一化的纯数学（供 Viz 复用、单测覆盖）。
export function mean(v: number[]): number {
  return v.reduce((a, b) => a + b, 0) / v.length;
}
export function rms(v: number[]): number {
  return Math.sqrt(v.reduce((a, b) => a + b * b, 0) / v.length);
}
export function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

// LayerNorm：去均值再除以标准差（中心化 + 缩放）。
export function layerNorm(v: number[], eps = 1e-5): number[] {
  const m = mean(v);
  const variance = v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length;
  const denom = Math.sqrt(variance + eps);
  return v.map((x) => (x - m) / denom);
}

// RMSNorm：只按均方根缩放，不去均值（保持方向）。
export function rmsNorm(v: number[], eps = 1e-5): number[] {
  const ms = v.reduce((a, b) => a + b * b, 0) / v.length;
  const denom = Math.sqrt(ms + eps);
  return v.map((x) => x / denom);
}
```

- [ ] **Step 4:** Run `npm test` → PASS（全绿）。
- [ ] **Step 5:** Commit:
```bash
git add rl-lab/src/algorithms/arch/normalization-fns.ts rl-lab/src/algorithms/arch/normalization-fns.test.ts
git commit -m "feat(rl-lab): normalization math + unit tests"
```

---

## Task 5: 归一化 builder + Viz + 接线

**Files:** Create `normalization.ts`、`NormViz.tsx`；Modify `backgrounds.ts`、`AlgorithmLab.tsx`

- [ ] **Step 1: builder** — `rl-lab/src/algorithms/arch/normalization.ts`：
```ts
import { Trajectory } from "@/player/types";
import { NormState } from "@/visualizers/arch/types";

export function runNormalization(): Trajectory<NormState> {
  // 固定的玩具激活向量（d=3，均值/尺度各异，便于看清中心化与缩放）。
  const vectors = [
    [2, 1, -1],
    [3, 3, 0],
    [-1, -2, -3],
    [0.5, 2, 4],
    [5, -1, 2],
    [1, 1, 1],
  ];
  return {
    meta: {
      id: "normalization",
      title: "归一化 · LayerNorm→RMSNorm→QKNorm→Pre/Post",
      family: "normalization",
      algorithm: "Normalization",
      description: "深层网络靠归一化才训得稳，它们到底做了什么、差别在哪？",
      tutorial: {
        problem: "层一深，激活值的尺度就乱飘，梯度爆炸/消失。归一化把每层激活拉回稳定范围。",
        intuition:
          "LayerNorm 先去均值再除以标准差（中心化+缩放）。RMSNorm 省掉去均值，只按均方根缩放——更快、且现代大模型（LLaMA 等）证明效果不输。QKNorm 把注意力里的 Q/K 先归一化，压住过大的注意力 logits。Pre-Norm（归一化放在残差之前）让深层训练更稳，是现代 Transformer 的默认；Post-Norm 表达力强但难训。",
        watch: [
          "LayerNorm：每个向量归一化后均值≈0、与原方向不同",
          "RMSNorm：只缩放、方向不变（余弦相似度≈1），且不强行去均值",
          "QKNorm：归一化 Q/K 后注意力 logits 的幅度被压住",
          "Pre vs Post：残差流幅度随层深的增长曲线，Post 更易发散",
        ],
        concepts: [
          { term: "LayerNorm", explain: "(x−均值)/标准差，中心化+缩放" },
          { term: "RMSNorm", explain: "x/RMS，只缩放不去均值，保持方向、更省算力" },
          { term: "QKNorm", explain: "对 Q/K 归一化以稳定注意力分数" },
          { term: "Pre/Post-Norm", explain: "归一化放残差前(Pre,更稳)还是后(Post,更强但难训)" },
        ],
        tryThis: "在 LayerNorm 和 RMSNorm 间切换，盯住每个向量的「方向(余弦)」是否改变。",
      },
    },
    frames: [{ iter: 0, state: { vectors }, metrics: {} }],
  };
}
```

- [ ] **Step 2: Viz** — `rl-lab/src/visualizers/arch/NormViz.tsx`：
```tsx
import { useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { NormState, NormVariant } from "./types";
import { layerNorm, rmsNorm, mean, rms, cosine } from "@/algorithms/arch/normalization-fns";

const VARIANTS: { key: NormVariant; label: string }[] = [
  { key: "layernorm", label: "LayerNorm" },
  { key: "rmsnorm", label: "RMSNorm" },
  { key: "qknorm", label: "QKNorm" },
  { key: "prepost", label: "Pre/Post-Norm" },
];

function VecRow({ v, normed }: { v: number[]; normed: number[] }) {
  return (
    <tr className="font-mono text-[11px]">
      <td className="px-2 py-1 text-slate-500">[{v.map((x) => x.toFixed(1)).join(", ")}]</td>
      <td className="px-2 py-1 text-sky-300">[{normed.map((x) => x.toFixed(2)).join(", ")}]</td>
      <td className="px-2 py-1 text-slate-400">{mean(normed).toFixed(2)}</td>
      <td className="px-2 py-1 text-slate-400">{rms(normed).toFixed(2)}</td>
      <td className="px-2 py-1 text-amber-400">{cosine(normed, v).toFixed(3)}</td>
    </tr>
  );
}

export default function NormViz({ state }: { state: NormState }) {
  const [variant, setVariant] = useState<NormVariant>("layernorm");
  const vectors = state.vectors;

  const normFn = variant === "rmsnorm" ? rmsNorm : layerNorm;

  // Pre/Post：残差流幅度随层深（合成示意）
  const depthData = useMemo(
    () =>
      Array.from({ length: 24 }, (_, d) => ({
        depth: d,
        post: Math.pow(1.18, d), // 后归一化：幅度滚雪球
        pre: 1 + Math.log1p(d) * 0.6, // 前归一化：受控缓增
      })),
    [],
  );

  // QKNorm：随机 q,k 尺度放大时的 logit 幅度，归一化前后对比
  const qkData = useMemo(
    () =>
      Array.from({ length: 12 }, (_, i) => {
        const scale = 1 + i * 0.6;
        const d = 8;
        const raw = scale * scale * d * 0.5; // 未归一化 logit 随尺度平方膨胀
        const normed = 1 * 1 * d * 0.5; // 归一化后 q,k 为单位向量，logit 受控
        return { scale: Number(scale.toFixed(1)), raw, normed };
      }),
    [],
  );

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

      {(variant === "layernorm" || variant === "rmsnorm") && (
        <div className="w-full max-w-xl overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="text-[11px] text-slate-500">
                <th className="px-2 py-1">原向量</th>
                <th className="px-2 py-1">归一化后</th>
                <th className="px-2 py-1">均值</th>
                <th className="px-2 py-1">RMS</th>
                <th className="px-2 py-1">余弦(原)</th>
              </tr>
            </thead>
            <tbody>
              {vectors.map((v, i) => (
                <VecRow key={i} v={v} normed={normFn(v)} />
              ))}
            </tbody>
          </table>
          <div className="text-[12px] text-slate-400 mt-2 px-2">
            {variant === "layernorm"
              ? "LayerNorm：归一化后每个向量均值≈0；因为先去了均值，方向(余弦)通常≠1。"
              : "RMSNorm：只按 RMS 缩放，方向不变(余弦≈1.000)，也不强行把均值压到 0——更省、效果不输。"}
          </div>
        </div>
      )}

      {variant === "qknorm" && (
        <div className="w-full" style={{ height: 260 }}>
          <ResponsiveContainer>
            <LineChart data={qkData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="scale" stroke="#64748b" tick={{ fontSize: 11 }} label={{ value: "Q/K 尺度", position: "insideBottom", offset: -2, fontSize: 11, fill: "#64748b" }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="raw" name="未归一化 logit" stroke="#ff5252" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="normed" name="QKNorm 后 logit" stroke="#38bdf8" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
          <div className="text-[12px] text-slate-400 text-center mt-1">
            不归一化时注意力 logit 随 Q/K 尺度平方膨胀→softmax 饱和、梯度消失；QKNorm 把 Q/K 拉回单位尺度，logit 受控。
          </div>
        </div>
      )}

      {variant === "prepost" && (
        <div className="w-full" style={{ height: 260 }}>
          <ResponsiveContainer>
            <LineChart data={depthData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="depth" stroke="#64748b" tick={{ fontSize: 11 }} label={{ value: "层深", position: "insideBottom", offset: -2, fontSize: 11, fill: "#64748b" }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="post" name="Post-Norm 残差流幅度" stroke="#ff5252" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="pre" name="Pre-Norm 残差流幅度" stroke="#38bdf8" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
          <div className="text-[12px] text-slate-400 text-center mt-1">
            Pre-Norm（归一化放残差之前）让残差流幅度受控、深层好训，是现代 Transformer 默认；Post-Norm 幅度易滚雪球、难训但表达力强。
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: background** — 在 `BACKGROUNDS` 里加：
```ts
  normalization: {
    realWorld:
      "归一化是深层网络能训起来的关键。RMSNorm 取代 LayerNorm、Pre-Norm 取代 Post-Norm，这些「不起眼的小改」正是现代大模型能堆到几十上百层还稳定收敛的幕后功臣。",
    uses: ["深层训练稳定", "大模型架构", "训练加速", "梯度健康"],
  },
```

- [ ] **Step 4: 接线 AlgorithmLab.tsx**

(a) import：
```ts
import { runNormalization } from "@/algorithms/arch/normalization";
import NormViz from "@/visualizers/arch/NormViz";
```

(b) DEMOS 里、`pos-encoding` 那条之后加：
```ts
  {
    key: "normalization",
    label: "归一化",
    group: "LLM 架构解剖",
    build: () => runNormalization(),
    Viz: NormViz,
    metricKey: "noop",
    metricLabel: "（交互演示 · 无训练曲线）",
    metricColor: "#38bdf8",
  },
```

(c) TREE 的「LLM 架构解剖（Transformer 内部）」枝 children 末尾加：
```ts
      { label: "归一化", children: [{ key: "normalization" }] },
```

- [ ] **Step 5:** Run `npm run check && npm test && npm run build` → 三者 PASS（测试应为既有 15 + 位置编码 5 + 归一化 6 = 26 左右，按实际为准全绿即可）。
- [ ] **Step 6:** Commit:
```bash
git add rl-lab/src/algorithms/arch/normalization.ts rl-lab/src/visualizers/arch/NormViz.tsx rl-lab/src/data/backgrounds.ts rl-lab/src/pages/AlgorithmLab.tsx
git commit -m "feat(rl-lab): normalization comparison experiment"
```

---

## Task 6: 端到端人工验证

**Files:** 无（仅运行）

- [ ] **Step 1:** `npm run dev`（5180）。打开 http://localhost:5180/，在「LLM 架构解剖（Transformer 内部）」枝下确认新增「位置编码」「归一化」：
  - 位置编码：4 个变体可切；正弦 PE 出热力图+位置滑块高亮行；RoPE 出旋转向量+位置滑块；YaRN 出 RoPE/YaRN 角度曲线；NoPE 出文字说明。底部参考文献列 PE/RoPE/NoPE/YaRN。
  - 归一化：4 个变体可切；LayerNorm/RMSNorm 出向量表（均值/RMS/余弦），RMSNorm 余弦≈1.000、LayerNorm 均值≈0；QKNorm 出 logit 对比曲线；Pre/Post 出层深幅度曲线。参考文献列 LayerNorm/RMSNorm/QKNorm/Pre·Post。
  - 在这两个与其它实验间来回切换，无白屏/崩溃。

- [ ] **Step 2:** 确认无误后打勾；有问题回对应 Task 修复并重跑其验证步骤。

---

## Self-Review（对照 spec）

- **覆盖**：落地 spec §3 表 #1 位置编码、#2 归一化。
- **占位符**：无；所有代码步骤含完整代码。
- **类型一致性**：`PosEncVariant`/`PosEncState`/`NormVariant`/`NormState` 在 Task 1 定义，被 builder/Viz 一致引用；`sinusoidalPE`/`ropeApply`/`ropeAngle`/`yarnAngle`/`yarnApply`/`dot` 命名在 fns/test/Viz 一致；`layerNorm`/`rmsNorm`/`mean`/`rms`/`cosine` 命名一致；DEMO key 与 papers.ts 键、TREE 叶子 key 一致（`pos-encoding`/`normalization`）。
- **family**：`pos-encoding`、`normalization` 已在 Batch 1 加入 Family 联合。
- **运行时安全**：单帧 + 空 metrics + metricKey "noop"，MetricCurve 用 `?? null` 容错（Batch 1 已验证），不崩。
```
