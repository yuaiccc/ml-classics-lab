// DeerFlow 深度研究 —— 从本地 DeerFlow 数据库读出的一次真实研究轨迹（只读，未触碰其服务）。
// 数据文件存 { meta, steps }，这里展开成逐步揭示的帧（复用 AgentPlot）。
import { Trajectory, AgentState, AgentStep } from "@/player/types";
import data from "@/data/deerflow.json";

export function runDeerflow(): Trajectory<AgentState> {
  const steps = (data as { steps: AgentStep[] }).steps;
  const task = steps.find((s) => s.type === "task")?.text ?? "";
  const frames = steps.map((_, i) => ({
    iter: i,
    state: { task, steps, shown: i + 1 },
    metrics: { step: i },
  }));
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return { meta: (data as any).meta, frames };
}
