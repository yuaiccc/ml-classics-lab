// 播放器 UI：纯展示组件，状态由 useTrajectory 提供。
import { Play, Pause, RotateCcw, SkipForward, SkipBack } from "lucide-react";

interface Props {
  index: number;
  last: number;
  playing: boolean;
  speed: number;
  iter: number;
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  onStepFwd: () => void;
  onStepBack: () => void;
  onSeek: (i: number) => void;
  onSpeed: (s: number) => void;
}

const SPEEDS = [0.5, 1, 2, 4];

export default function TrajectoryPlayer({
  index,
  last,
  playing,
  speed,
  iter,
  onPlay,
  onPause,
  onReset,
  onStepFwd,
  onStepBack,
  onSeek,
  onSpeed,
}: Props) {
  const btn =
    "w-9 h-9 rounded-lg flex items-center justify-center border border-slate-700 text-slate-300 hover:border-[#00ff88]/50 hover:text-[#00ff88] transition-all";

  return (
    <div className="glass rounded-xl p-4 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <button className={btn} onClick={onReset} title="重置">
          <RotateCcw className="w-4 h-4" />
        </button>
        <button className={btn} onClick={onStepBack} title="上一步">
          <SkipBack className="w-4 h-4" />
        </button>
        <button
          className="w-11 h-9 rounded-lg flex items-center justify-center bg-[rgba(0,255,136,0.15)] border border-[#00ff88]/40 text-[#00ff88] hover:bg-[rgba(0,255,136,0.25)] transition-all"
          onClick={playing ? onPause : onPlay}
          title={playing ? "暂停" : "播放"}
        >
          {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </button>
        <button className={btn} onClick={onStepFwd} title="下一步">
          <SkipForward className="w-4 h-4" />
        </button>

        <div className="ml-auto flex items-center gap-1">
          {SPEEDS.map((s) => (
            <button
              key={s}
              onClick={() => onSpeed(s)}
              className={`px-2 py-1 rounded text-xs font-mono border transition-all ${
                speed === s
                  ? "border-[#00e5ff]/50 text-[#00e5ff] bg-[rgba(0,229,255,0.1)]"
                  : "border-slate-800 text-slate-500 hover:border-slate-600"
              }`}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-3">
        <span className="font-mono text-xs text-slate-500 w-20 shrink-0">
          iter {iter}
        </span>
        <input
          type="range"
          min={0}
          max={last}
          value={index}
          onChange={(e) => onSeek(Number(e.target.value))}
          className="flex-1 accent-[#00ff88]"
        />
        <span className="font-mono text-xs text-slate-500 w-14 shrink-0 text-right">
          {index}/{last}
        </span>
      </div>
    </div>
  );
}
