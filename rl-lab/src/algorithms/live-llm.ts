// 大模型实验「真·实时」运行器：浏览器经 Vite 代理直接调本地 Ollama 的 Qwen，
// 边调用边产生帧（emit 实时预览），不再回放预计算 JSON。meta/教程复用已有 JSON。
import { Trajectory, Frame, RagState, AgentState, AgentStep, EmbeddingState } from "@/player/types";
import { embed, generate, cosine } from "@/lib/ollama";
import ragJson from "@/data/rag.json";
import agentJson from "@/data/agent-react.json";
import agenticJson from "@/data/agentic-rag.json";
import qwenMetaJson from "@/data/qwen-embeddings.json";
import { QWEN_EMB_META } from "./qwen-embeddings";
import { mulberry32 } from "./rng";

const EMB = "qwen3-embedding:0.6b";
const GEN = "qwen2.5:7b";

export interface Progress {
  label: string;
  step: number;
  total: number;
}
type Emit<S> = (state: S, p: Progress) => void;

const round = (v: number, n = 4) => Math.round(v * 10 ** n) / 10 ** n;

// ---------------- RAG（实时） ----------------
const RAG_QUERY = "什么动物跑得最快？它能跑多快？";
const RAG_DOCS = [
  "猎豹是陆地上奔跑最快的动物，短距离冲刺时速可达约 120 公里。",
  "蓝鲸是地球上现存最大的动物，体长可超过 30 米。",
  "大象是陆地上最大的哺乳动物，记忆力很强、寿命很长。",
  "蜂鸟是唯一能向后飞行的鸟，每秒振翅约 50 次。",
  "树懒动作极慢，大部分时间挂在树上睡觉。",
  "章鱼有三个心脏，血液是蓝色的。",
  "帝企鹅能潜到水下 500 米深处觅食。",
  "长颈鹿是世界上最高的陆地动物，脖子特别长。",
  "蝙蝠是唯一真正会飞的哺乳动物，靠回声定位捕食。",
  "游隼俯冲时速可超过 300 公里，是飞行最快的鸟。",
];

export async function runRagLive(emit: Emit<RagState>): Promise<Trajectory<RagState>> {
  const TOP = 3;
  emit({ query: RAG_QUERY, docs: [], shown: 0, topK: TOP, answer: "" }, { label: "用 qwen3-embedding 嵌入问题与知识库…", step: 0, total: 3 });
  const vecs = await embed(EMB, [RAG_QUERY, ...RAG_DOCS]);
  const qv = vecs[0];
  const scored = RAG_DOCS.map((t, i) => ({ text: t, score: round(cosine(qv, vecs[i + 1])) })).sort((a, b) => b.score - a.score);

  const frames: Frame<RagState>[] = [];
  for (let shown = 0; shown <= scored.length; shown++) {
    const st: RagState = { query: RAG_QUERY, docs: scored, shown, topK: TOP, answer: "" };
    frames.push({ iter: shown, state: st, metrics: { topScore: scored[0].score } });
    emit(st, { label: `检索排序中 ${shown}/${scored.length}`, step: 1, total: 3 });
    await new Promise((r) => setTimeout(r, 60));
  }

  emit({ query: RAG_QUERY, docs: scored, shown: scored.length, topK: TOP, answer: "" }, { label: "qwen2.5 基于检索资料生成回答…（约 20-30s）", step: 2, total: 3 });
  const ctx = scored.slice(0, TOP).map((d) => "- " + d.text).join("\n");
  const answer = await generate(GEN, `只根据下面的资料回答问题，不要编造。\n资料：\n${ctx}\n\n问题：${RAG_QUERY}\n回答：`, { temperature: 0.2, num_predict: 160 });
  const final: RagState = { query: RAG_QUERY, docs: scored, shown: scored.length, topK: TOP, answer };
  frames.push({ iter: scored.length + 1, state: final, metrics: { topScore: scored[0].score } });
  emit(final, { label: "完成", step: 3, total: 3 });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return { meta: (ragJson as any).meta, frames };
}

// ---------------- Agent ReAct（实时，计算器工具） ----------------
const AGENT_TASK =
  "一个班有 32 名学生，每人交 15 元班费，已全部收齐。班级买了 3 箱饮料，每箱 89 元；又买了 1 个篮球 128 元。剩下的钱平分给 4 个活动小组，每组能分到多少元？";
const AGENT_SYS = `你是一个会使用工具的智能体，用 ReAct 方式分步解题。
可用工具：calc[算式] —— 计算一个数学算式，例如 calc[32*15]。
严格按格式，每轮只输出一个 Thought 和（可选）一个 Action：
Thought: <思考>
Action: calc[<算式>]
我会把计算结果作为 Observation 返回。能给出答案时用：
Answer: <最终答案>`;

function calc(expr: string): string {
  const e = expr.replace(/×/g, "*").replace(/÷/g, "/").trim();
  if (!/^[0-9+\-*/.()\s]+$/.test(e)) return "（非法算式）";
  try {
    // eslint-disable-next-line no-new-func
    const v = Function(`"use strict";return (${e})`)();
    return String(round(v));
  } catch {
    return "（计算错误）";
  }
}

async function runReactLive(
  task: string,
  system: string,
  tool: (q: string) => Promise<string> | string,
  toolName: "calc" | "retrieve",
  meta: Trajectory<AgentState>["meta"],
  emit: Emit<AgentState>
): Promise<Trajectory<AgentState>> {
  const steps: AgentStep[] = [{ type: "task", text: task }];
  const show = (label: string, p = 0, t = 1) => emit({ task, steps: [...steps], shown: steps.length }, { label, step: p, total: t });
  show("智能体启动…");
  let transcript = `${system}\n\n问题：${task}\n`;
  const re = new RegExp(`${toolName}\\[(.+?)\\]`);

  for (let k = 0; k < 6; k++) {
    show(`第 ${k + 1} 轮：qwen2.5 思考中…（约 15-30s）`, k, 6);
    const out = await generate(GEN, transcript, { temperature: 0.15, num_predict: 220, stop: ["Observation:"] });
    const th = out.match(/Thought:\s*(.+)/);
    if (th) {
      steps.push({ type: "thought", text: th[1].trim() });
      show("");
    }
    const ans = out.match(/Answer:\s*([\s\S]+)/);
    if (ans) {
      steps.push({ type: "answer", text: ans[1].trim() });
      show("完成");
      break;
    }
    const act = out.match(re);
    if (act) {
      const obs = await tool(act[1]);
      steps.push({ type: "action", text: `${toolName}[${act[1]}]` });
      steps.push({ type: "observation", text: obs });
      show("读取工具结果…");
      transcript += `${out}\nObservation: ${obs}\n`;
    } else {
      transcript += out + "\n";
      if (!th) break;
    }
  }

  const frames: Frame<AgentState>[] = steps.map((_, i) => ({ iter: i, state: { task, steps, shown: i + 1 }, metrics: { step: i } }));
  return { meta, frames };
}

export function runAgentLive(emit: Emit<AgentState>): Promise<Trajectory<AgentState>> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return runReactLive(AGENT_TASK, AGENT_SYS, calc, "calc", (agentJson as any).meta, emit);
}

// ---------------- Agentic RAG（实时，检索工具，多跳） ----------------
const RAG_QUESTION = "鸿蒙科技的 CEO 毕业于哪所大学？";
const RAG_KB = [
  "鸿蒙科技的 CEO 是张伟。",
  "张伟本科毕业于清华大学计算机系。",
  "星河传媒的 CEO 是李娜。",
  "李娜毕业于北京大学新闻学院。",
  "鸿蒙科技主要做芯片设计。",
  "星河传媒主要做影视制作。",
  "清华大学位于北京市海淀区。",
];
const RAG_SYS = `你是一个会检索的智能体，用 ReAct 方式回答问题。
可用工具：retrieve[查询词] —— 从知识库检索最相关的一条资料。
资料分散，你可能需要检索多次、把线索串起来。每轮只输出一步：
Thought: <思考下一步查什么>
Action: retrieve[<查询词>]
我会把检索到的资料作为 Observation 返回。掌握足够信息后用：
Answer: <最终答案>`;

export async function runAgenticRagLive(emit: Emit<AgentState>): Promise<Trajectory<AgentState>> {
  let kbVecs: number[][] | null = null;
  const retrieve = async (q: string) => {
    if (!kbVecs) kbVecs = await embed(EMB, RAG_KB);
    const qv = (await embed(EMB, [q]))[0];
    let best = 0;
    let bv = -Infinity;
    kbVecs.forEach((v, i) => {
      const s = cosine(qv, v);
      if (s > bv) {
        bv = s;
        best = i;
      }
    });
    return RAG_KB[best];
  };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return runReactLive(RAG_QUESTION, RAG_SYS, retrieve, "retrieve", (agenticJson as any).meta, emit);
}

// ---------------- Qwen 语义嵌入（实时嵌入 + MDS） ----------------
const EMB_WORDS = (qwenMetaJson as { words: string[] }).words;
const EMB_GROUPS = (qwenMetaJson as { groups: number[] }).groups;

export async function runEmbeddingsLive(emit: Emit<EmbeddingState>): Promise<Trajectory<EmbeddingState>> {
  const n = EMB_WORDS.length;
  const rng = mulberry32(1234);
  const pos = Array.from({ length: n }, () => ({ x: (rng() * 2 - 1) * 0.6, y: (rng() * 2 - 1) * 0.6 }));
  emit({ words: EMB_WORDS, positions: pos.map((p) => ({ ...p })), groups: EMB_GROUPS }, { label: "用 qwen3-embedding 实时嵌入 25 个词…", step: 0, total: 2 });

  const vecs = await embed(EMB, EMB_WORDS);
  const norm = vecs.map((v) => {
    const m = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    return v.map((x) => x / m);
  });
  const D = norm.map((a) => norm.map((b) => Math.sqrt(Math.max(0, 2 - 2 * cosine(a, b)))));

  const frames: Frame<EmbeddingState>[] = [];
  const stress = () => {
    let s = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) s += (Math.hypot(pos[i].x - pos[j].x, pos[i].y - pos[j].y) - D[i][j]) ** 2;
    return s;
  };
  const snap = (it: number) => frames.push({ iter: it, state: { words: EMB_WORDS, positions: pos.map((p) => ({ ...p })), groups: EMB_GROUPS }, metrics: { stress: stress() } });
  snap(0);
  emit({ words: EMB_WORDS, positions: pos.map((p) => ({ ...p })), groups: EMB_GROUPS }, { label: "MDS 还原 2D 语义簇…", step: 1, total: 2 });

  for (let t = 1; t <= 90; t++) {
    const grad = pos.map(() => ({ x: 0, y: 0 }));
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const dx = pos[i].x - pos[j].x;
        const dy = pos[i].y - pos[j].y;
        const dist = Math.hypot(dx, dy) || 1e-6;
        const c = (2 * (dist - D[i][j])) / dist;
        grad[i].x += c * dx;
        grad[i].y += c * dy;
      }
    for (let i = 0; i < n; i++) {
      pos[i].x -= (0.1 * grad[i].x) / n;
      pos[i].y -= (0.1 * grad[i].y) / n;
    }
    if (t % 2 === 0) snap(t);
  }
  emit(frames[frames.length - 1].state, { label: "完成", step: 2, total: 2 });
  return { meta: QWEN_EMB_META, frames };
}
