import { useEffect, useState } from "react";

interface FrozenLakeResult {
  successRate: number;
  finalReward: number;
  qTable: number[][];
}

const MAP = ["SFFF", "FHFH", "FFFH", "HFFG"];
const ACTIONS = ["←", "↓", "→", "↑"];

export default function FrozenLakeViz() {
  const [result, setResult] = useState<FrozenLakeResult | null>(null);

  useEffect(() => {
    let mounted = true;
    fetch("/results/frozenlake-qlearning.json")
      .then((res) => res.json())
      .then((data) => {
        if (mounted) setResult(data);
      })
      .catch(() => {
        if (mounted) setResult(null);
      });
    return () => {
      mounted = false;
    };
  }, []);

  if (!result) {
    return (
      <div className="glass rounded-xl p-6 mb-8">
        <div className="text-sm text-stone-500">FrozenLake 策略数据尚未加载。</div>
      </div>
    );
  }

  return (
    <div className="glass rounded-xl p-6 mb-8">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-5">
        <h2 className="text-sm font-bold text-stone-600">FrozenLake 学到的策略</h2>
        <div className="text-[10px] font-mono text-stone-500">
          success {(result.successRate * 100).toFixed(1)}% · return {result.finalReward.toFixed(3)}
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2 max-w-sm">
        {MAP.flatMap((row, r) =>
          row.split("").map((cell, c) => {
            const index = r * 4 + c;
            const q = result.qTable[index] ?? [0, 0, 0, 0];
            const action = ACTIONS[q.indexOf(Math.max(...q))] ?? "·";
            const terminal = cell === "H" || cell === "G";
            const label = cell === "S" ? "START" : cell === "G" ? "GOAL" : cell === "H" ? "HOLE" : "ICE";

            return (
              <div
                key={`${r}-${c}`}
                className={`aspect-square rounded-lg border p-2 flex flex-col items-center justify-center ${
                  cell === "G"
                    ? "border-[#cc785c]/40 bg-[#cc785c]/10"
                    : cell === "H"
                      ? "border-[#b04a3a]/35 bg-[#b04a3a]/10"
                      : "border-stone-200 bg-stone-100/40"
                }`}
              >
                <div className="text-[10px] font-mono text-stone-500">{label}</div>
                <div className={`text-2xl font-mono ${terminal ? "text-stone-500" : "text-[#5b7b9a]"}`}>
                  {terminal ? cell : action}
                </div>
                <div className="text-[10px] font-mono text-stone-400">s{index}</div>
              </div>
            );
          }),
        )}
      </div>
    </div>
  );
}
