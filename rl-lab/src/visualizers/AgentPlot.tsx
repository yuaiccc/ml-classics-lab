// agent 家族：ReAct 推理轨迹，逐步展开成卡片（思考/行动/观察/答案）。
import { AgentState } from "@/player/types";
import { Brain, Wrench, Eye, CheckCircle2, HelpCircle } from "lucide-react";

const STYLE = {
  task: { icon: HelpCircle, label: "任务", color: "#94a3b8", bg: "rgba(148,163,184,0.1)", border: "rgba(148,163,184,0.3)" },
  thought: { icon: Brain, label: "思考", color: "#00e5ff", bg: "rgba(0,229,255,0.08)", border: "rgba(0,229,255,0.3)" },
  action: { icon: Wrench, label: "行动 · 调用工具", color: "#00ff88", bg: "rgba(0,255,136,0.08)", border: "rgba(0,255,136,0.3)" },
  observation: { icon: Eye, label: "观察 · 工具返回", color: "#ffab40", bg: "rgba(255,171,64,0.08)", border: "rgba(255,171,64,0.3)" },
  answer: { icon: CheckCircle2, label: "最终答案", color: "#b388ff", bg: "rgba(179,136,255,0.1)", border: "rgba(179,136,255,0.4)" },
} as const;

export default function AgentPlot({ state }: { state: AgentState }) {
  const { steps, shown } = state;
  const visible = steps.slice(0, shown);

  return (
    <div className="rounded-xl bg-[rgba(10,14,23,0.6)] p-5 flex flex-col gap-2.5 min-h-[360px]">
      {visible.map((step, i) => {
        const s = STYLE[step.type] ?? STYLE.thought;
        const Icon = s.icon;
        const isLast = i === visible.length - 1;
        return (
          <div
            key={i}
            className="rounded-lg p-3 flex gap-3 items-start animate-fade-in-up"
            style={{ background: s.bg, border: `1px solid ${s.border}`, boxShadow: isLast ? `0 0 16px ${s.bg}` : "none" }}
          >
            <Icon className="w-4 h-4 mt-0.5 shrink-0" style={{ color: s.color }} />
            <div className="min-w-0">
              <div className="text-[11px] font-semibold mb-0.5" style={{ color: s.color }}>{s.label}</div>
              <div className={`text-sm leading-relaxed ${step.type === "action" || step.type === "observation" ? "font-mono" : ""} text-slate-300 break-words`}>
                {step.text}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
