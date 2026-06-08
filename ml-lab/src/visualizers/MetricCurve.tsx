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
  color = "#cc785c",
}: Props) {
  const data = frames.map((f) => ({
    iter: f.iter,
    value: f.metrics?.[metricKey] ?? null,
  }));
  const currentIter = frames[index]?.iter ?? 0;

  return (
    <div className="glass rounded-xl p-4">
      <div className="text-xs text-stone-500 mb-2 font-mono">{label}</div>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
          <CartesianGrid stroke="rgba(20,20,19,0.06)" />
          <XAxis dataKey="iter" stroke="#908e85" fontSize={11} />
          <YAxis stroke="#908e85" fontSize={11} width={48} />
          <Tooltip
            contentStyle={{
              background: "rgba(255, 255, 255,0.97)",
              border: "1px solid rgba(204, 120, 92,0.35)",
              borderRadius: 8,
              fontSize: 12,
              color: "#141413",
            }}
            labelStyle={{ color: "#6b6a65" }}
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
