// 标量指标曲线（loss / inertia / reward ...），当前帧用竖线标出。
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  ReferenceLine,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { Frame } from "@/player/types";

interface Props {
  frames: Frame[];
  index: number;
  metricKey: string;
  label: string;
  color?: string;
}

export default function MetricCurve({
  frames,
  index,
  metricKey,
  label,
  color = "#00ff88",
}: Props) {
  const data = frames.map((f) => ({
    iter: f.iter,
    value: f.metrics?.[metricKey] ?? null,
  }));
  const currentIter = frames[index]?.iter ?? 0;

  return (
    <div className="glass rounded-xl p-4">
      <div className="text-xs text-slate-500 mb-2 font-mono">{label}</div>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="iter" stroke="#475569" fontSize={11} />
          <YAxis stroke="#475569" fontSize={11} width={48} />
          <Tooltip
            contentStyle={{
              background: "rgba(15,23,42,0.95)",
              border: "1px solid rgba(0,255,136,0.3)",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "#94a3b8" }}
          />
          <ReferenceLine x={currentIter} stroke={color} strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
