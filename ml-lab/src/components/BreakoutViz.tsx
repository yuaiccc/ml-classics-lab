import { useEffect, useState } from "react";

interface BreakoutResult {
  finalReward: number;
  bestReward: number;
  actions: number[];
  frames: number[][];
}

const ACTION_LABELS = ["NOOP", "FIRE", "RIGHT", "LEFT"];

export default function BreakoutViz() {
  const [result, setResult] = useState<BreakoutResult | null>(null);
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    fetch("/results/breakout-random.json")
      .then((res) => res.json())
      .then(setResult)
      .catch(() => setResult(null));
  }, []);

  useEffect(() => {
    if (!result?.frames.length) return;
    const timer = window.setInterval(() => {
      setFrame((value) => (value + 1) % result.frames.length);
    }, 120);
    return () => window.clearInterval(timer);
  }, [result]);

  if (!result) {
    return (
      <div className="glass rounded-xl p-6 mb-8">
        <div className="text-xs font-mono text-stone-500">Loading Breakout frames...</div>
      </div>
    );
  }

  const pixels = result.frames[frame] ?? [];
  const action = result.actions[frame] ?? 0;

  return (
    <div className="glass rounded-xl p-6 mb-8">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-5">
        <div>
          <h2 className="text-sm font-bold text-stone-700">Atari Breakout 帧动画</h2>
          <p className="text-[11px] text-stone-500 mt-1">
            随机策略 baseline 的像素输入流，展示 Atari 从屏幕帧到动作的控制问题。
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
          <div className="rounded-md border border-[#c99a4e]/20 bg-[#c99a4e]/5 px-2 py-1">
            <div className="text-stone-500">MEAN</div>
            <div className="text-[#c99a4e]">{result.finalReward.toFixed(2)}</div>
          </div>
          <div className="rounded-md border border-[#5b7b9a]/20 bg-[#5b7b9a]/5 px-2 py-1">
            <div className="text-stone-500">BEST</div>
            <div className="text-[#5b7b9a]">{result.bestReward.toFixed(2)}</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[220px_1fr] gap-5 items-center">
        <div className="mx-auto rounded-lg border border-stone-200 bg-black p-3">
          <div className="grid h-[168px] w-[168px] grid-cols-[repeat(42,4px)] grid-rows-[repeat(42,4px)] overflow-hidden image-rendering-pixelated">
            {pixels.map((value, index) => (
              <div
                key={`${frame}-${index}`}
                style={{ backgroundColor: `rgb(${value},${Math.max(0, value - 16)},${Math.max(0, value - 28)})` }}
              />
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-lg border border-stone-200 bg-stone-100/40 p-4">
            <div className="flex items-center justify-between text-[10px] font-mono text-stone-500 mb-2">
              <span>FRAME</span>
              <span>{frame + 1}/{result.frames.length}</span>
            </div>
            <div className="h-2 rounded-full bg-stone-100 overflow-hidden">
              <div
                className="h-full rounded-full bg-[#5b7b9a] transition-all duration-150"
                style={{ width: `${((frame + 1) / result.frames.length) * 100}%` }}
              />
            </div>
          </div>

          <div className="grid grid-cols-4 gap-2">
            {ACTION_LABELS.map((label, index) => (
              <div
                key={label}
                className={`rounded-lg border px-2 py-3 text-center text-[10px] font-mono transition-all ${
                  action === index
                    ? "border-[#cc785c]/50 bg-[#cc785c]/15 text-[#cc785c]"
                    : "border-stone-200 bg-stone-100/30 text-stone-500"
                }`}
              >
                {label}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
