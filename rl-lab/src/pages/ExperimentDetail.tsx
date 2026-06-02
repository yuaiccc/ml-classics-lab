import { useParams, Link } from "react-router-dom";
import { useLabStore } from "@/hooks/useLabStore";
import { ArrowLeft, Clock, Zap, Layers, Target } from "lucide-react";
import CartPoleViz from "@/components/CartPoleViz";
import MountainCarViz from "@/components/MountainCarViz";
import PendulumViz from "@/components/PendulumViz";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";

const CURVE_COLORS: Record<string, string> = {
  PPO: "#00e5ff",
  DQN: "#b388ff",
  SAC: "#00ff88",
  "DQN + Reward Shaping": "#ffab40",
};

const STATUS_MAP: Record<string, { label: string; cls: string }> = {
  success: { label: "✓ 成功", cls: "status-success" },
  failed: { label: "✗ 失败", cls: "status-failed" },
  partial: { label: "△ 部分", cls: "status-partial" },
};

const MDP_ICONS = [
  { key: "S", label: "状态空间", emoji: "📊" },
  { key: "A", label: "动作空间", emoji: "🎮" },
  { key: "P", label: "转移函数", emoji: "🔄" },
  { key: "R", label: "奖励函数", emoji: "🏆" },
  { key: "gamma", label: "折扣因子", emoji: "⏳" },
];

export default function ExperimentDetail() {
  const { id } = useParams<{ id: string }>();
  const { experiments } = useLabStore();
  const exp = experiments.find((e) => e.id === id);

  if (!exp) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-20 text-center">
        <p className="text-slate-500">实验未找到</p>
        <Link to="/dashboard" className="text-[#00ff88] text-sm mt-2 inline-block">
          ← 返回总览
        </Link>
      </div>
    );
  }

  const color = CURVE_COLORS[exp.algorithm] || "#00ff88";
  const status = STATUS_MAP[exp.status];

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Link
        to="/dashboard"
        className="inline-flex items-center gap-1 text-sm text-slate-500 hover:text-[#00ff88] transition-colors mb-6"
      >
        <ArrowLeft className="w-4 h-4" />
        返回总览
      </Link>

      <div className="flex items-start justify-between mb-6">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <h1 className="text-2xl font-bold text-slate-200 font-mono">
              {exp.env}
            </h1>
            <span className={`text-xs font-mono px-2 py-0.5 rounded-full ${status.cls}`}>
              {status.label}
            </span>
          </div>
          <p className="text-sm text-slate-500 max-w-xl">{exp.description}</p>
          {exp.abstract && (
            <div className="mt-3 p-3 rounded-lg bg-[rgba(0,255,136,0.05)] border border-[rgba(0,255,136,0.1)]">
              <div className="text-[10px] text-[#00ff88]/60 font-mono mb-1">ABSTRACT</div>
              <p className="text-xs text-slate-400 leading-relaxed">{exp.abstract}</p>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-8">
        <div className="glass rounded-lg p-3">
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <Zap className="w-3 h-3" />
            最终奖励
          </div>
          <div className="text-lg font-bold font-mono" style={{ color }}>
            {exp.finalReward}
          </div>
        </div>
        <div className="glass rounded-lg p-3">
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <Target className="w-3 h-3" />
            目标奖励
          </div>
          <div className="text-lg font-bold font-mono text-slate-300">
            {exp.targetReward}
          </div>
        </div>
        <div className="glass rounded-lg p-3">
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <Clock className="w-3 h-3" />
            训练耗时
          </div>
          <div className="text-lg font-bold font-mono text-slate-300">
            {exp.trainingTime}s
          </div>
        </div>
        <div className="glass rounded-lg p-3">
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <Layers className="w-3 h-3" />
            训练轮次
          </div>
          <div className="text-lg font-bold font-mono text-slate-300">
            {exp.epochs}
          </div>
        </div>
      </div>

      {exp.id === "cartpole-ppo" && <CartPoleViz />}
      {(exp.id === "mountaincar-dqn" || exp.id === "mountaincar-ppo") && (
        <MountainCarViz variant={exp.id === "mountaincar-dqn" ? "dqn" : "shaped"} />
      )}
      {exp.id === "pendulum-sac" && <PendulumViz />}

      <div className="glass rounded-xl p-6 mb-8">
        <h2 className="text-sm font-bold text-slate-400 mb-4">📈 训练曲线</h2>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={exp.curve} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
              <defs>
                <linearGradient id={`detail-grad-${exp.id}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={color} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.15)" />
              <XAxis
                dataKey="epoch"
                tick={{ fill: "#64748b", fontSize: 11, fontFamily: "JetBrains Mono" }}
                axisLine={{ stroke: "rgba(100,116,139,0.2)" }}
              />
              <YAxis
                tick={{ fill: "#64748b", fontSize: 11, fontFamily: "JetBrains Mono" }}
                axisLine={{ stroke: "rgba(100,116,139,0.2)" }}
              />
              <Tooltip
                contentStyle={{
                  background: "rgba(15,23,42,0.9)",
                  border: "1px solid rgba(0,255,136,0.2)",
                  borderRadius: "8px",
                  fontSize: "12px",
                  fontFamily: "JetBrains Mono",
                }}
                labelStyle={{ color: "#94a3b8" }}
                itemStyle={{ color }}
                formatter={(value: number) => [value.toFixed(1), "奖励"]}
                labelFormatter={(label) => `Epoch ${label}`}
              />
              <ReferenceLine
                y={exp.targetReward}
                stroke="rgba(255,171,64,0.5)"
                strokeDasharray="5 5"
                label={{
                  value: `目标 ${exp.targetReward}`,
                  fill: "#ffab40",
                  fontSize: 10,
                  fontFamily: "JetBrains Mono",
                }}
              />
              <Area
                type="monotone"
                dataKey="reward"
                stroke={color}
                strokeWidth={2}
                fill={`url(#detail-grad-${exp.id})`}
                dot={{ r: 3, fill: color, stroke: "#0a0e17", strokeWidth: 2 }}
                activeDot={{ r: 5, fill: color, stroke: "#0a0e17", strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="glass rounded-xl p-6">
          <h2 className="text-sm font-bold text-slate-400 mb-4">⚙️ 超参数</h2>
          <div className="space-y-2">
            {Object.entries(exp.hyperparams).map(([key, val]) => (
              <div
                key={key}
                className="flex items-center justify-between py-1.5 border-b border-slate-800/50 last:border-0"
              >
                <span className="text-xs text-slate-500 font-mono">{key}</span>
                <span className="text-xs text-slate-300 font-mono font-medium">
                  {String(val)}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="glass rounded-xl p-6">
          <h2 className="text-sm font-bold text-slate-400 mb-4">🧩 MDP 五元组</h2>
          <div className="space-y-3">
            {MDP_ICONS.map(({ key, label, emoji }) => {
              const value = key === "gamma" ? String(exp.mdp.gamma) : exp.mdp[key as keyof typeof exp.mdp];
              return (
                <div key={key} className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-[rgba(0,255,136,0.1)] flex items-center justify-center text-sm shrink-0">
                    {emoji}
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-600 font-mono uppercase">
                      {label} ({key === "gamma" ? "γ" : key})
                    </div>
                    <div className="text-xs text-slate-300">{value}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="glass rounded-xl p-6">
        <h2 className="text-sm font-bold text-slate-400 mb-3">💡 算法原理</h2>
        <p className="text-sm text-slate-400 leading-relaxed">
          {exp.algorithmInsight}
        </p>
      </div>
    </div>
  );
}
