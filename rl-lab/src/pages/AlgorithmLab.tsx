// M1 验证页：把帧契约 + 播放器 + 可视化器串起来，跑梯度下降 / K-Means 两个 demo。
// 新增算法 = 往 DEMOS 里加一条 builder（产出 Trajectory），无需改播放器。
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, RefreshCw, Lightbulb } from "lucide-react";
import { Trajectory } from "@/player/types";
import { runLinearRegressionGD } from "@/algorithms/gradient-descent";
import { runKMeans } from "@/algorithms/kmeans";
import { useTrajectory } from "@/player/useTrajectory";
import TrajectoryPlayer from "@/player/TrajectoryPlayer";
import RegressionPlot from "@/visualizers/RegressionPlot";
import ClustersPlot from "@/visualizers/ClustersPlot";
import MetricCurve from "@/visualizers/MetricCurve";

interface Demo {
  key: string;
  label: string;
  build: (seed?: number) => Trajectory;
  metricKey: string;
  metricLabel: string;
}

const DEMOS: Demo[] = [
  {
    key: "linreg",
    label: "梯度下降 · 线性回归",
    build: (seed) => runLinearRegressionGD({ seed }),
    metricKey: "loss",
    metricLabel: "MSE Loss",
  },
  {
    key: "kmeans",
    label: "K-Means 聚类",
    build: (seed) => runKMeans({ seed }),
    metricKey: "inertia",
    metricLabel: "Inertia（簇内平方和）",
  },
];

function Viz({ traj, index }: { traj: Trajectory; index: number }) {
  const state = traj.frames[index].state;
  if (traj.meta.family === "scatter-boundary") {
    return <RegressionPlot state={state as never} />;
  }
  if (traj.meta.family === "clusters") {
    return <ClustersPlot state={state as never} />;
  }
  return <div className="text-slate-500 text-sm">暂无该家族的可视化器</div>;
}

export default function AlgorithmLab() {
  const [demoKey, setDemoKey] = useState(DEMOS[0].key);
  const [seed, setSeed] = useState(1234);
  const demo = DEMOS.find((d) => d.key === demoKey)!;

  const traj = useMemo(() => demo.build(seed), [demo, seed]);
  const player = useTrajectory(traj);
  const meta = traj.meta;

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Link
        to="/"
        className="inline-flex items-center gap-1.5 text-sm text-slate-500 hover:text-[#00ff88] transition-colors mb-6"
      >
        <ArrowLeft className="w-4 h-4" /> 返回看板
      </Link>

      <header className="mb-6">
        <h1 className="text-2xl font-bold text-slate-200 mb-1">算法过程动画 · 实验台</h1>
        <p className="text-slate-500 text-sm">
          M1 脚手架：统一帧契约 + 通用播放器，浏览器端实时计算可交互
        </p>
      </header>

      {/* 算法切换 + 重新生成 */}
      <div className="flex items-center gap-2 mb-5 flex-wrap">
        {DEMOS.map((d) => (
          <button
            key={d.key}
            onClick={() => setDemoKey(d.key)}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-all ${
              demoKey === d.key
                ? "filter-btn-active"
                : "border-slate-700 text-slate-400 hover:border-slate-500"
            }`}
          >
            {d.label}
          </button>
        ))}
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="ml-auto inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border border-slate-700 text-slate-300 hover:border-[#00e5ff]/50 hover:text-[#00e5ff] transition-all"
          title="换一组随机数据重新计算"
        >
          <RefreshCw className="w-3.5 h-3.5" /> 重新生成数据
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="lg:col-span-2 flex flex-col gap-4">
          <Viz traj={traj} index={player.index} />
          <TrajectoryPlayer
            index={player.index}
            last={player.last}
            playing={player.playing}
            speed={player.speed}
            iter={player.frame.iter}
            onPlay={player.play}
            onPause={player.pause}
            onReset={player.reset}
            onStepFwd={player.stepFwd}
            onStepBack={player.stepBack}
            onSeek={player.seek}
            onSpeed={player.setSpeed}
          />
        </div>

        <div className="flex flex-col gap-4">
          <MetricCurve
            frames={traj.frames}
            index={player.index}
            metricKey={demo.metricKey}
            label={demo.metricLabel}
          />

          {meta.insight && (
            <div className="glass rounded-xl p-4">
              <div className="flex items-center gap-1.5 text-xs text-[#ffab40] mb-2">
                <Lightbulb className="w-3.5 h-3.5" /> 算法在做什么
              </div>
              <p className="text-sm text-slate-400 leading-relaxed">{meta.insight}</p>
            </div>
          )}

          {meta.hyperparams && (
            <div className="glass rounded-xl p-4">
              <div className="text-xs text-slate-500 mb-2 font-mono">超参数</div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
                {Object.entries(meta.hyperparams).map(([k, v]) => (
                  <div key={k} className="flex justify-between text-xs">
                    <span className="text-slate-500 font-mono">{k}</span>
                    <span className="text-slate-300 font-mono">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
