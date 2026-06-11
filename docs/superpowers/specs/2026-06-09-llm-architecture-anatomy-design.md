# 设计：LLM 架构解剖（Transformer 内部）板块

日期：2026-06-09
状态：已与用户确认，待转实现计划

## 1. 目标

在现有「算法过程动画 · 实验台」（ml-classics-lab / rl-lab）里新增一个顶级板块
**「LLM 架构解剖（Transformer 内部）」**，把现代大模型 Transformer block 的内部零件
按"对比"的方式逐类解剖、做成可交互的过程动画。

这是与现有"AI 通史"轴正交的一个**新维度**：现有板块横向铺开"古典 ML → 神经网络 →
CNN → Attention → Mamba → RAG → Agent"；本板块纵向深入"Transformer block 里到底
由哪些零件组成、每个零件从 X 演进到 Y 解决了什么"。

覆盖用户提供的全部 35 篇参考论文（见 §8 引用表）。

## 2. 已确认的关键决策

1. **范围**：8 大类全覆盖（约 35 篇论文）。
2. **呈现方式**：对比式分组 —— 8 个"对比实验"，每个内部用开关切换变体；每篇论文
   作为某个实验里的一个变体被引用。
3. **数据源**：混合。绝大多数组件用**浏览器端 TS 确定性玩具数据**；第 4 个（稀疏·
   窗口注意力）额外叠一版**本地真实小模型的真实注意力稀疏图**（本地 HF + forward
   hook，Python 预计算成帧 JSON）。
4. **技术路线**：复用现有"按 demo 绑 Viz + 帧播放器"，不改 Family 联合类型（分发是
   按 demo 的，每个实验直接绑自己的 Viz）。零新 UI 基建。
5. **DeepSeek**：不用。其聊天补全 API 拿不到内部注意力/路由，无法支撑内部结构可视化；
   "真实"那一小撮走本地 HF 预计算。
6. **极新/未公开论文**（Mamba-3 / DSA / mHC / AttnResidual / CSA-HCA）：实现前逐篇抓
   内容确认；抓不到实锤的，在该变体上明确标注「按公开描述近似演示」，不假装精确复现。

## 3. 板块结构：8 个对比实验

新建顶级树枝「LLM 架构解剖（Transformer 内部）」，置于"大模型时代"之后。

| # | key | 实验 | 变体（切换对比） | 动画轴 | 主要 Viz 思路 |
|---|---|---|---|---|---|
| 1 | `pos-encoding` | 位置编码 | 无 / 正弦PE / RoPE / NoPE / YaRN | 时间轴 = token 位置 | RoPE 演示 2D 查询/键向量随位置旋转；正弦 PE 的 sin/cos 热图；YaRN 演示旋转频率缩放以扩上下文；指标 = 相对位置点积保持度 |
| 2 | `normalization` | 归一化 | LayerNorm / RMSNorm / QK·KVNorm / Pre·Post-Norm | 静态（变体开关 + 滑块） | 玩具激活点云几何：LN 先中心化再投到球面、RMS 只缩放不去均值；QKNorm 看注意力 logits 稳定性；Pre vs Post 看残差流幅度随深度增长 |
| 3 | `attention-kv` | 注意力 KV 变体 | MHA / MQA / GQA / MLA | 静态（变体开关 + 头数滑块） | Q 头与 K/V 头的分组示意框；KV-cache 显存随变体缩小；MLA 演示 KV 压成低秩潜向量再上投；指标 = KV-cache 内存 |
| 4 | `sparse-attention` | 稀疏·窗口注意力 | 全连接 / Sparse / SWA / DSA / CSA·HCA | 静态（掩码图案）+ 真实切片 | L×L 注意力掩码图案（哪些格被计算）+ FLOPs/被注意格占比；**额外叠一版本地真实小模型的真实稀疏注意力图（HF 预计算成帧）** |
| 5 | `linear-seq` | 线性·SSM | Lightning / GatedDeltaNet / KDA / Mamba / Mamba-2 / Mamba-3 | 时间轴 = 扫描位置 | 固定大小状态递推扫描 vs 注意力 KV 增长的对比；各变体标注改了什么（门控 / delta 规则等）；与现有 mamba-ssm 风格呼应 |
| 6 | `moe` | 专家混合 | Dense / MoE(top-k) / Switch(top-1) / DeepSeekMoE | 时间轴 = token 流 | 路由器逐 token 点亮被选专家；激活参数/总参占比（稀疏度）、负载均衡；DeepSeekMoE 演示细粒度专家 + 共享专家 |
| 7 | `activations` | 激活函数 | ReLU / GELU / SiLU / Swish / SwiGLU | 静态（函数图 + β 滑块） | 函数曲线 + 导数曲线；Swish β 滑块；GLU/SwiGLU 门控示意 |
| 8 | `residual` | 残差连接 | Plain / RC / HC / mHC / AttnResidual | 时间轴 = 层深 | 信号/梯度幅度沿层深流动；有/无 skip 的差异；HC/mHC 多条并行残差流的学习混合；AttnResidual 演示对残差流做注意力 |

动画轴归类：
- **播放型（有自然时间轴）**：1 `pos-encoding`(位置)、5 `linear-seq`(扫描位置)、
  6 `moe`(token 流)、8 `residual`(层深)。
- **静态结构对比（交互在 Viz 内：变体开关 + 滑块，单帧或少帧）**：2 `normalization`、
  3 `attention-kv`、4 `sparse-attention`、7 `activations`。

两类都装在同一个帧播放器里：静态型用 1 帧、交互控件放进 Viz 组件内部（参照现有
`AttentionHeatmap` 的层/头下拉）；播放型由帧序列驱动、play 键动画。

## 4. 数据契约与组件

复用 A 的统一帧契约 `Trajectory = { meta, frames[] }`、`Frame = { iter, state, metrics }`。
每个实验用**自定义 state 形状**（在该实验自己的类型文件里声明），其对应 Viz 直接读。

变体选择放在 **Viz 内部的开关**：builder 把"所有变体"的数据都算进 state，Viz 按当前
选中的变体渲染，无需重建帧。播放型实验里，每一帧（某个位置/层深）的 state 仍包含
全部变体的数据，切变体即时切换、不打断播放。

### 文件结构
```
rl-lab/src/visualizers/arch/
  PosEncodingViz.tsx
  NormViz.tsx
  AttnKVViz.tsx
  SparseAttnViz.tsx
  LinearSeqViz.tsx
  MoEViz.tsx
  ActivationViz.tsx
  ResidualViz.tsx
  types.ts            # 8 个实验的 state 形状
rl-lab/src/algorithms/arch/
  pos-encoding.ts     # 确定性玩具 builder（纯 TS，零库）
  normalization.ts
  attention-kv.ts
  sparse-attention.ts # 玩具掩码 builder（真实切片走预计算 JSON）
  linear-seq.ts
  moe.ts
  activations.ts
  residual.ts
rl-lab/src/data/papers.ts          # PAPERS 引用表
rl-lab/src/data/frames/sparse-attention-real.json   # 第4个真实切片预计算帧
solvers/arch_sparse_attention.py   # 本地 HF + forward hook 导出真实稀疏注意力
```

### 接入 AlgorithmLab.tsx
- 8 条 DEMO（key/label/group/build/Viz/metricKey…）。
- TREE 新增顶级枝「LLM 架构解剖（Transformer 内部）」+ 8 个子节点。
- `sparse-attention` 进 ENGINE map 标 `pytorch`（因含真实 HF 预计算切片）；其余无引擎徽章。
- `BACKGROUNDS` 加 8 条"现实意义"文案；每个实验配小白教程（problem/intuition/watch/
  concepts）。

## 5. 参考文献引用机制（PAPERS）

新增 `src/data/papers.ts`：

```ts
export interface PaperRef { name: string; url: string; note?: string }
export const PAPERS: Record<string, PaperRef[]> = {
  "pos-encoding": [ {name:"PE (Attention Is All You Need)", url:"https://arxiv.org/abs/1706.03762v7"}, ... ],
  ...
}
```

在实验面板里加一个「参考文献」区块，列出该实验涉及的全部变体论文（名称 + arXiv 链接，
原样挂用户给的链接）。极新/近似演示的变体在此区块用 `note` 标注「按公开描述近似演示」。

## 6. 第 4 个：本地真实稀疏注意力切片

- `solvers/arch_sparse_attention.py`：本地加载一个小型真实 transformer（如 GPT-2 small
  或现有 IMDb tiny transformer），对一段示例输入跑前向，用 forward hook 抓某层多头的
  真实 L×L 注意力权重，导出为帧 JSON（沿用 `imdb-transformer` 的导出格式与
  `bridgeMlLab`/`AttentionHeatmap` 复用思路）。
- 前端：`SparseAttnViz` 在玩具掩码图案旁，提供一个「真实样本」切换，加载预计算的真实
  注意力图，直观对照"理论稀疏图案 vs 真实注意力的天然稀疏性"。
- 该实验因此带 `pytorch` 徽章。依赖：本地 `transformers` + 一个小模型权重（实现阶段确认
  环境，复用现有 conda `tianshou` env 或新建）。

## 7. 范围边界 / 非目标

- 不接 DeepSeek，不跑大模型在线推理。
- 不追求论文级精确复现；玩具演示以"讲清直觉"为第一目标，精确数值非目标。
- 不重构现有实验；仅新增本板块所需文件与 AlgorithmLab 的接入点。
- 极新论文按 §2.6 处理。

## 8. 参考论文清单（用户提供，原样挂链接）

- 位置编码：PE https://arxiv.org/abs/1706.03762v7 · RoPE https://arxiv.org/abs/2104.09864 · NoPE https://arxiv.org/abs/2305.19466 · YaRN https://arxiv.org/abs/2309.00071
- 归一化：LayerNorm https://arxiv.org/abs/1607.06450 · RMSNorm https://arxiv.org/abs/1910.07467 · QK/KVNorm https://arxiv.org/abs/2010.04245 · Pre/Post-Norm https://arxiv.org/abs/2002.04745
- 注意力 KV：MHA https://arxiv.org/abs/1706.03762v7 · MQA https://arxiv.org/abs/1911.02150 · GQA https://arxiv.org/abs/2305.13245 · MLA https://arxiv.org/abs/2405.04434
- 稀疏·窗口：Sparse Transformer https://arxiv.org/abs/1904.10509 · DSA https://arxiv.org/abs/2512.02556 · SWA https://arxiv.org/abs/2004.05150 · Gemma2-SWA https://arxiv.org/abs/2408.00118 · CSA/HCA https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
- 线性·SSM：KDA https://arxiv.org/abs/2510.26692 · Gated DeltaNet https://arxiv.org/abs/2412.06464 · Lightning Attention https://arxiv.org/abs/2401.04658 · Mamba https://arxiv.org/abs/2312.00752 · Mamba-2 https://arxiv.org/abs/2405.21060 · Mamba-3 https://arxiv.org/abs/2603.15569
- MoE：MoE https://arxiv.org/abs/1701.06538 · Switch Transformer https://arxiv.org/abs/2101.03961 · DeepSeekMoE https://arxiv.org/abs/2401.06066
- 激活：GELU https://arxiv.org/abs/1606.08415 · SiLU https://arxiv.org/abs/1702.03118 · Swish https://arxiv.org/abs/1710.05941v1
- 残差：RC (ResNet) https://arxiv.org/abs/1512.03385 · HC https://arxiv.org/abs/2409.19606 · mHC https://arxiv.org/abs/2512.24880 · AttnResidual https://arxiv.org/abs/2603.15031

## 9. 成功标准

- 8 个实验全部接入树形导航，可切换、可播放/交互、不崩。
- 每个实验：变体开关可对比；配小白教程 + 现实意义 + 参考文献区块。
- 第 4 个能加载并展示本地真实小模型的真实注意力稀疏图。
- 极新/近似变体均带「近似演示」标注。
- `tsc --noEmit` 干净、`vite build` 成功。

## 10. 分批实现建议（留给 writing-plans）

工作量较大，建议分 4 批，每批 2 个实验，先易后难：
1. 激活(7) + 注意力KV(3)（最直观、纯静态）
2. 位置编码(1) + 归一化(2)
3. MoE(6) + 残差(8)
4. 线性·SSM(5) + 稀疏注意力(4，含本地 HF 真实切片，最重，放最后）
配套：papers.ts 引用机制在第 1 批一并搭好；arch/types.ts 随用随加。
