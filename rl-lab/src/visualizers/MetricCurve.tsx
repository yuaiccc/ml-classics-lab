// 标量指标曲线（loss / inertia / reward ...），当前帧用竖线标出。
// 支持可选的第二条曲线（如训练 vs 测试误差）。
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  ReferenceLine,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";
import { Frame } from "@/player/types";

interface Props {
  frames: Frame[];
  index: number;
  metricKey: string;
  label: string;
  color?: string;
  metricKey2?: string;
  label2?: string;
  color2?: string;
}

export default function MetricCurve({
  frames,
  index,
  metricKey,
  label,
  color = "#00ff88",
  metricKey2,
  label2,
  color2 = "#ff5252",
}: Props) {
  const data = frames.map((f) => ({
    iter: f.iter,
    value: f.metrics?.[metricKey] ?? null,
    value2: metricKey2 ? f.metrics?.[metricKey2] ?? null : null,
  }));
  const currentIter = frames[index]?.iter ?? 0;

  return (
    <div className="glass rounded-xl p-4">
      <div className="text-xs text-slate-500 mb-2 font-mono">{label}</div>
      <ResponsiveContainer width="100%" height={metricKey2 ? 200 : 180}>
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
          {metricKey2 && <Legend wrapperStyle={{ fontSize: 11 }} />}
          <ReferenceLine x={currentIter} stroke="#94a3b8" strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="value"
            name={label}
            stroke={color}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          {metricKey2 && (
            <Line
              type="monotone"
              dataKey="value2"
              name={label2 ?? metricKey2}
              stroke={color2}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
