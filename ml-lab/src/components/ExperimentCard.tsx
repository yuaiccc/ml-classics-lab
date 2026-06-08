import { ExperimentLegacy } from "@/data/experiments";
import { Clock, Zap, ArrowRight } from "lucide-react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
} from "recharts";

const STATUS_MAP: Record<string, { label: string; cls: string }> = {
  success: { label: "✓ 成功", cls: "status-success" },
  failed: { label: "✗ 失败", cls: "status-failed" },
  partial: { label: "△ 部分", cls: "status-partial" },
};

const ALG_BADGE_MAP: Record<string, string> = {
  PPO: "badge-ppo",
  DQN: "badge-dqn",
  SAC: "badge-sac",
  "DQN + Reward Shaping": "badge-shaped",
};

const CURVE_COLORS: Record<string, string> = {
  PPO: "#5b7b9a",
  DQN: "#7a8b5a",
  SAC: "#cc785c",
  "DQN + Reward Shaping": "#c99a4e",
};

export default function ExperimentCard({ experiment }: { experiment: ExperimentLegacy }) {
  const status = STATUS_MAP[experiment.status];
  const badge = ALG_BADGE_MAP[experiment.algorithm] || "badge-ppo";
  const color = CURVE_COLORS[experiment.algorithm] || "#cc785c";

  return (
    <div className="glass glass-hover rounded-xl p-5 transition-all duration-300 group cursor-pointer h-full flex flex-col">
      <div className="flex items-start justify-between mb-3">
        <span
          className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${status.cls}`}
        >
          {status.label}
        </span>
        <span
          className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${badge}`}
        >
          {experiment.algorithm}
        </span>
      </div>

      <h3 className="text-base font-bold text-stone-800 mb-1 font-mono">
        {experiment.env}
      </h3>

      <p className="text-xs text-stone-500 mb-4 line-clamp-2 flex-grow">
        {experiment.description}
      </p>

      <div className="h-16 mb-3 -mx-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={experiment.curve}>
            <defs>
              <linearGradient id={`grad-${experiment.id}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.3} />
                <stop offset="100%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="epoch" hide />
            <YAxis hide domain={["dataMin - 50", "dataMax + 50"]} />
            <Area
              type="monotone"
              dataKey="reward"
              stroke={color}
              strokeWidth={1.5}
              fill={`url(#grad-${experiment.id})`}
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center justify-between text-xs text-stone-500">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <Zap className="w-3 h-3" />
            <span className="font-mono text-stone-700">
              {experiment.finalReward}
            </span>
          </span>
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            <span className="font-mono">{experiment.trainingTime}s</span>
          </span>
        </div>
        <ArrowRight className="w-3 h-3 text-stone-500 group-hover:text-[#cc785c] transition-colors" />
      </div>
    </div>
  );
}
