import { useEffect, useMemo, useState } from "react";
import { ExperimentLegacy } from "@/data/experiments";

function formatMetric(value: number, ratioMode: boolean) {
  if (ratioMode) return `${(value * 100).toFixed(1)}%`;
  if (Math.abs(value) < 10) return value.toFixed(3);
  return value.toFixed(1);
}

export default function ExperimentMotion({ experiment }: { experiment: ExperimentLegacy }) {
  const frames = useMemo(() => {
    if (experiment.curve.length > 0) return experiment.curve;
    return Array.from({ length: Math.max(2, experiment.epochs || 2) }, (_, index) => ({
      epoch: index,
      reward: experiment.finalReward * (index / Math.max(1, (experiment.epochs || 2) - 1)),
    }));
  }, [experiment]);
  const [frameIndex, setFrameIndex] = useState(0);

  useEffect(() => {
    const id = window.setInterval(() => {
      setFrameIndex((index) => (index + 1) % frames.length);
    }, 650);
    return () => window.clearInterval(id);
  }, [frames.length]);

  const current = frames[frameIndex];
  const values = frames.map((frame) => frame.reward);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const progress = frames.length === 1 ? 1 : frameIndex / (frames.length - 1);
  const normalized = (current.reward - min) / range;
  // 数值都落在 [0,1] 时按比率(准确率)显示百分比；否则是 loss/分数等原始值
  const ratioMode = max <= 1.0001 && min >= -0.0001;
  const metricLabel =
    experiment.category === "rl"
      ? "reward"
      : ratioMode
        ? "accuracy"
        : experiment.category === "deep"
          ? "loss"
          : "metric";

  return (
    <div className="glass rounded-xl p-6 mb-8 overflow-hidden">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-5">
        <h2 className="text-sm font-bold text-stone-600">动态训练回放</h2>
        <div className="text-[10px] font-mono text-stone-500">
          epoch {current.epoch} · {metricLabel} {formatMetric(current.reward, ratioMode)}
        </div>
      </div>

      <div className="relative h-32 rounded-xl border border-stone-200 bg-stone-100/40 overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 bg-[#cc785c]/5 transition-all duration-500"
          style={{ width: `${Math.max(4, progress * 100)}%` }}
        />
        <div className="absolute inset-x-6 top-1/2 h-px bg-stone-200" />
        <div className="absolute inset-x-6 top-1/2">
          {frames.map((frame, index) => {
            const x = frames.length === 1 ? 0 : index / (frames.length - 1);
            const y = 1 - (frame.reward - min) / range;
            return (
              <span
                key={`${frame.epoch}-${index}`}
                className="absolute h-1.5 w-1.5 rounded-full bg-slate-700"
                style={{
                  left: `${x * 100}%`,
                  top: `${(y - 0.5) * 78}px`,
                }}
              />
            );
          })}
        </div>
        <div
          className="absolute top-1/2 h-8 w-8 -translate-x-1/2 -translate-y-1/2 rounded-full border border-[#5b7b9a]/40 bg-[#5b7b9a]/15 shadow-lg shadow-[#5b7b9a]/20 transition-all duration-500"
          style={{
            left: `calc(1.5rem + ${progress} * (100% - 3rem))`,
            transform: `translate(-50%, calc(-50% + ${(0.5 - normalized) * 78}px))`,
          }}
        >
          <span className="absolute inset-1 rounded-full bg-[#5b7b9a] animate-pulse" />
        </div>
        <div className="absolute bottom-3 left-5 right-5 flex justify-between text-[10px] font-mono text-stone-500">
          <span>start</span>
          <span>{experiment.algorithm}</span>
          <span>target {formatMetric(experiment.targetReward, ratioMode)}</span>
        </div>
        <div className="motion-scanline" />
      </div>
    </div>
  );
}
