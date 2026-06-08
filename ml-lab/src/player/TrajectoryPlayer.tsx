import { Play, Pause, RotateCcw, SkipBack, SkipForward, Gauge } from "lucide-react";
import { useTrajectory } from "./useTrajectory";
import { Trajectory } from "./types";
import ScatterBoundary from "@/visualizers/ScatterBoundary";
import Clusters from "@/visualizers/Clusters";
import Curves from "@/visualizers/Curves";
import MetricCurve from "@/visualizers/MetricCurve";
import AttentionHeatmap from "@/visualizers/AttentionHeatmap";
import ImageGrid from "@/visualizers/ImageGrid";

interface Props {
  trajectory: Trajectory;
}

const SPEED_OPTIONS = [0.25, 0.5, 1, 2, 4];
const CURVE_METRIC_PRIORITY = ["reward", "accuracy", "loss", "inertia"];

function pickMetricKey(metrics: Record<string, number>): string | null {
  const keys = Object.keys(metrics);
  if (keys.length === 0) return null;
  return CURVE_METRIC_PRIORITY.find((k) => keys.includes(k)) ?? keys[0];
}

export default function TrajectoryPlayer({ trajectory }: Props) {
  const {
    frame,
    currentFrame,
    totalFrames,
    playing,
    speed,
    setPlaying,
    setSpeed,
    goTo,
    stepForward,
    stepBackward,
    reset,
  } = useTrajectory(trajectory);

  return (
    <div className="glass rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="font-serif text-xl text-stone-900">收敛过程动画</h2>
        <div className="flex items-center gap-1">
          <Gauge className="w-3 h-3 text-stone-500" />
          {SPEED_OPTIONS.map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-all ${
                speed === s
                  ? "border-[#cc785c]/40 text-[#cc785c] bg-[#cc785c]/10"
                  : "border-stone-300/50 text-stone-500 hover:border-slate-500"
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      <div className="flex justify-center mb-4">
        {frame.state.family === "scatter-boundary" && (
          <ScatterBoundary state={frame.state.data} metrics={frame.metrics} />
        )}
        {frame.state.family === "clusters" && (
          <Clusters state={frame.state.data} metrics={frame.metrics} />
        )}
        {frame.state.family === "attention" && (
          <AttentionHeatmap state={frame.state.data} />
        )}
        {frame.state.family === "image-grid" && (
          <ImageGrid state={frame.state.data} />
        )}
        {frame.state.family === "curves" &&
          (frame.state.data.embedding && frame.state.data.embedding.length > 0 ? (
            <Curves state={frame.state.data} metrics={frame.metrics} />
          ) : (
            (() => {
              const metricKey = pickMetricKey(frame.metrics);
              if (!metricKey) return null;
              return (
                <div className="w-full max-w-[520px]">
                  <MetricCurve
                    frames={trajectory.frames.slice(0, currentFrame + 1)}
                    index={currentFrame}
                    metricKey={metricKey}
                    label={`${metricKey} · 第 ${frame.iter} 步`}
                  />
                </div>
              );
            })()
          ))}
      </div>

      <div className="flex items-center gap-2 mb-3">
        <input
          type="range"
          min={0}
          max={totalFrames - 1}
          value={currentFrame}
          onChange={(e) => goTo(Number(e.target.value))}
          className="flex-1 h-1.5 bg-stone-200 rounded-lg appearance-none cursor-pointer
            [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:w-3
            [&::-webkit-slider-thumb]:h-3
            [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-[#cc785c]
            [&::-webkit-slider-thumb]:shadow-[0_0_8px_rgba(204, 120, 92,0.4)]"
        />
        <span className="text-[10px] font-mono text-stone-500 w-20 text-right">
          {currentFrame + 1}/{totalFrames}
        </span>
      </div>

      <div className="flex items-center justify-center gap-2">
        <button
          onClick={reset}
          className="w-8 h-8 rounded-lg border border-stone-300 flex items-center justify-center text-stone-500 hover:text-[#cc785c] hover:border-[#cc785c]/30 transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
        <button
          onClick={stepBackward}
          className="w-8 h-8 rounded-lg border border-stone-300 flex items-center justify-center text-stone-500 hover:text-[#5b7b9a] hover:border-[#5b7b9a]/30 transition-colors"
        >
          <SkipBack className="w-4 h-4" />
        </button>
        <button
          onClick={() => setPlaying(!playing)}
          className="w-10 h-10 rounded-lg border border-[#cc785c]/30 bg-[#cc785c]/10 flex items-center justify-center text-[#cc785c] hover:bg-[#cc785c]/20 transition-colors"
        >
          {playing ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
        </button>
        <button
          onClick={stepForward}
          className="w-8 h-8 rounded-lg border border-stone-300 flex items-center justify-center text-stone-500 hover:text-[#5b7b9a] hover:border-[#5b7b9a]/30 transition-colors"
        >
          <SkipForward className="w-4 h-4" />
        </button>
      </div>

      {Object.keys(frame.metrics).length > 0 && (
        <div className="mt-4 flex items-center justify-center gap-4 text-xs text-stone-500">
          {Object.entries(frame.metrics).map(([key, val]) => (
            <span key={key}>
              {key}: <span className="font-mono text-stone-700">{typeof val === 'number' ? val.toFixed(4) : val}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
