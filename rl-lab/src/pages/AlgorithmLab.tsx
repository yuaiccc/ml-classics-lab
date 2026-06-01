// 算法过程动画实验台：帧契约 + 通用播放器 + 各家族可视化器。
// 加新算法 = 往 DEMOS 加一条（builder 产出 Trajectory + 指定 Viz 组件），播放器自动复用。
import { ComponentType, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, RefreshCw, Lightbulb } from "lucide-react";
import { Trajectory, TrajectoryMeta } from "@/player/types";
import { runLinearRegressionGD } from "@/algorithms/gradient-descent";
import { runLogisticRegression } from "@/algorithms/logistic-regression";
import { runPerceptron } from "@/algorithms/perceptron";
import { runDecisionTree } from "@/algorithms/decision-tree";
import { runKNN } from "@/algorithms/knn";
import { runKMeans } from "@/algorithms/kmeans";
import { runDBSCAN } from "@/algorithms/dbscan";
import { runPCA } from "@/algorithms/pca";
import { runMLP } from "@/algorithms/mlp";
import { runSelectiveSSM } from "@/algorithms/mamba";
import cartpolePPO from "@/data/frames/cartpole-ppo.json";
import pendulumSAC from "@/data/frames/pendulum-sac.json";
import mountaincarPPO from "@/data/frames/mountaincar-ppo.json";
import cnnShapes from "@/data/frames/cnn-shapes.json";
import attentionReverse from "@/data/frames/attention-reverse.json";
// 算法源码（?raw 把文件当字符串导入，供前端「查看源码」展示）
import linregSrc from "@/algorithms/gradient-descent.ts?raw";
import logregSrc from "@/algorithms/logistic-regression.ts?raw";
import perceptronSrc from "@/algorithms/perceptron.ts?raw";
import dtreeSrc from "@/algorithms/decision-tree.ts?raw";
import knnSrc from "@/algorithms/knn.ts?raw";
import kmeansSrc from "@/algorithms/kmeans.ts?raw";
import dbscanSrc from "@/algorithms/dbscan.ts?raw";
import pcaSrc from "@/algorithms/pca.ts?raw";
import mlpSrc from "@/algorithms/mlp.ts?raw";
import mambaSrc from "@/algorithms/mamba.ts?raw";
import { useTrajectory } from "@/player/useTrajectory";
import TrajectoryPlayer from "@/player/TrajectoryPlayer";
import RegressionPlot from "@/visualizers/RegressionPlot";
import BoundaryPlot from "@/visualizers/BoundaryPlot";
import ClustersPlot from "@/visualizers/ClustersPlot";
import PCAPlot from "@/visualizers/PCAPlot";
import EnvPlot from "@/visualizers/EnvPlot";
import CnnPlot from "@/visualizers/CnnPlot";
import AttentionPlot from "@/visualizers/AttentionPlot";
import SsmPlot from "@/visualizers/SsmPlot";
import MetricCurve from "@/visualizers/MetricCurve";
import TutorialPanel from "@/components/TutorialPanel";
import CodeViewer from "@/components/CodeViewer";

interface Demo {
  key: string;
  label: string;
  group: string;
  build: (seed?: number) => Trajectory;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  Viz: ComponentType<{ state: any; meta?: TrajectoryMeta }>;
  metricKey: string;
  metricLabel: string;
  metricColor?: string;
  /** 算法源码（浏览器端 TS 实现），用于「查看源码」 */
  source?: { code: string; path: string };
}

const DEMOS: Demo[] = [
  {
    key: "linreg",
    label: "线性回归",
    group: "监督 · 回归",
    build: (seed) => runLinearRegressionGD({ seed }),
    Viz: RegressionPlot,
    metricKey: "loss",
    metricLabel: "MSE Loss",
    source: { code: linregSrc, path: "algorithms/gradient-descent.ts" },
  },
  {
    key: "logreg",
    label: "逻辑回归",
    group: "监督 · 分类",
    build: (seed) => runLogisticRegression({ seed }),
    Viz: BoundaryPlot,
    metricKey: "loss",
    metricLabel: "交叉熵 Loss",
    source: { code: logregSrc, path: "algorithms/logistic-regression.ts" },
  },
  {
    key: "perceptron",
    label: "感知机",
    group: "监督 · 分类",
    build: (seed) => runPerceptron({ seed }),
    Viz: BoundaryPlot,
    metricKey: "errors",
    metricLabel: "误分类点数",
    metricColor: "#ff5252",
    source: { code: perceptronSrc, path: "algorithms/perceptron.ts" },
  },
  {
    key: "dtree",
    label: "决策树",
    group: "监督 · 分类",
    build: (seed) => runDecisionTree({ seed }),
    Viz: BoundaryPlot,
    metricKey: "accuracy",
    metricLabel: "训练准确率",
    source: { code: dtreeSrc, path: "algorithms/decision-tree.ts" },
  },
  {
    key: "knn",
    label: "KNN",
    group: "监督 · 分类",
    build: (seed) => runKNN({ seed }),
    Viz: BoundaryPlot,
    metricKey: "accuracy",
    metricLabel: "留一法准确率",
    source: { code: knnSrc, path: "algorithms/knn.ts" },
  },
  {
    key: "kmeans",
    label: "K-Means",
    group: "无监督 · 聚类",
    build: (seed) => runKMeans({ seed }),
    Viz: ClustersPlot,
    metricKey: "inertia",
    metricLabel: "Inertia（簇内平方和）",
    source: { code: kmeansSrc, path: "algorithms/kmeans.ts" },
  },
  {
    key: "dbscan",
    label: "DBSCAN",
    group: "无监督 · 聚类",
    build: (seed) => runDBSCAN({ seed }),
    Viz: ClustersPlot,
    metricKey: "clusters",
    metricLabel: "已发现簇数",
    metricColor: "#b388ff",
    source: { code: dbscanSrc, path: "algorithms/dbscan.ts" },
  },
  {
    key: "pca",
    label: "PCA",
    group: "无监督 · 降维",
    build: (seed) => runPCA({ seed }),
    Viz: PCAPlot,
    metricKey: "variance",
    metricLabel: "主轴方向方差",
    metricColor: "#ffab40",
    source: { code: pcaSrc, path: "algorithms/pca.ts" },
  },
  {
    key: "mlp",
    label: "MLP 神经网络",
    group: "深度学习",
    build: (seed) => runMLP({ seed }),
    Viz: BoundaryPlot,
    metricKey: "loss",
    metricLabel: "交叉熵 Loss",
    metricColor: "#b388ff",
    source: { code: mlpSrc, path: "algorithms/mlp.ts" },
  },
  {
    key: "cnn-shapes",
    label: "CNN 卷积网络",
    group: "深度学习",
    build: () => cnnShapes as unknown as Trajectory,
    Viz: CnnPlot,
    metricKey: "accuracy",
    metricLabel: "训练准确率",
    metricColor: "#00e5ff",
  },
  {
    key: "attention-reverse",
    label: "Self-Attention",
    group: "深度学习",
    build: () => attentionReverse as unknown as Trajectory,
    Viz: AttentionPlot,
    metricKey: "accuracy",
    metricLabel: "token 准确率",
    metricColor: "#b388ff",
  },
  {
    key: "mamba-ssm",
    label: "Mamba (SSM)",
    group: "深度学习",
    build: (seed) => runSelectiveSSM({ seed }),
    Viz: SsmPlot,
    metricKey: "gate",
    metricLabel: "选择门 Δ（随扫描位置）",
    metricColor: "#00ff88",
    source: { code: mambaSrc, path: "algorithms/mamba.ts" },
  },
  {
    key: "cartpole-ppo",
    label: "CartPole · PPO",
    group: "强化学习 · env",
    build: () => cartpolePPO as unknown as Trajectory,
    Viz: EnvPlot,
    metricKey: "return",
    metricLabel: "累计回报 Return",
    metricColor: "#00ff88",
  },
  {
    key: "mountaincar-ppo",
    label: "MountainCar · PPO",
    group: "强化学习 · env",
    build: () => mountaincarPPO as unknown as Trajectory,
    Viz: EnvPlot,
    metricKey: "return",
    metricLabel: "累计回报 Return",
    metricColor: "#00e5ff",
  },
  {
    key: "pendulum-sac",
    label: "Pendulum · SAC",
    group: "强化学习 · env",
    build: () => pendulumSAC as unknown as Trajectory,
    Viz: EnvPlot,
    metricKey: "return",
    metricLabel: "累计回报 Return",
    metricColor: "#ffab40",
  },
];

const GROUPS = [...new Set(DEMOS.map((d) => d.group))];

export default function AlgorithmLab() {
  const [demoKey, setDemoKey] = useState(DEMOS[0].key);
  const [seed, setSeed] = useState(1234);
  const demo = DEMOS.find((d) => d.key === demoKey)!;

  const traj = useMemo(() => demo.build(seed), [demo, seed]);
  const player = useTrajectory(traj);
  const meta = traj.meta;
  const Viz = demo.Viz;

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Link
        to="/"
        className="inline-flex items-center gap-1.5 text-sm text-slate-500 hover:text-[#00ff88] transition-colors mb-6"
      >
        <ArrowLeft className="w-4 h-4" /> 返回看板
      </Link>

      <header className="mb-5">
        <h1 className="text-2xl font-bold text-slate-200 mb-1">算法过程动画 · 实验台</h1>
        <p className="text-slate-500 text-sm">
          每个经典算法配一段过程动画，边看边学。下面选一个算法，按 ▶️ 播放看它怎么一步步收敛。
        </p>
      </header>

      {/* 小白使用说明 */}
      <div className="glass rounded-xl p-4 mb-5 text-xs text-slate-400 leading-relaxed">
        <span className="text-[#00ff88] font-semibold">怎么用：</span>
        ① 选一个算法 → ② 按 <span className="text-slate-200">▶️ 播放</span> 看动画，或拖
        <span className="text-slate-200"> 进度条 </span>手动逐帧；③ 用
        <span className="text-slate-200"> ⏭ 下一步 </span>单步慢看；④ 点
        <span className="text-slate-200"> 重新生成数据 </span>换一组随机数据再看。
        每个算法下方都有「解决什么问题 / 直觉 / 看点 / 概念」的小白讲解。
      </div>

      {/* 算法切换（按类别分组） */}
      <div className="flex flex-col gap-2.5 mb-5">
        {GROUPS.map((g) => (
          <div key={g} className="flex items-center gap-2 flex-wrap">
            <span className="text-[11px] text-slate-600 font-mono w-24 shrink-0">{g}</span>
            {DEMOS.filter((d) => d.group === g).map((d) => (
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
          </div>
        ))}
      </div>

      <div className="flex items-end justify-between gap-4 mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">{meta.title}</h2>
          {meta.description && (
            <p className="text-sm text-slate-500 mt-0.5">{meta.description}</p>
          )}
        </div>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border border-slate-700 text-slate-300 hover:border-[#00e5ff]/50 hover:text-[#00e5ff] transition-all shrink-0"
          title="换一组随机数据重新计算"
        >
          <RefreshCw className="w-3.5 h-3.5" /> 重新生成数据
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="lg:col-span-2 flex flex-col gap-4">
          <Viz state={player.frame.state} meta={meta} />
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
            color={demo.metricColor}
          />

          {/* 没有完整教程时，退回显示简短 insight */}
          {!meta.tutorial && meta.insight && (
            <div className="glass rounded-xl p-4">
              <div className="flex items-center gap-1.5 text-xs text-[#ffab40] mb-2">
                <Lightbulb className="w-3.5 h-3.5" /> 算法在做什么
              </div>
              <p className="text-sm text-slate-400 leading-relaxed">{meta.insight}</p>
            </div>
          )}

          {meta.hyperparams && (
            <div className="glass rounded-xl p-4">
              <div className="text-xs text-slate-500 mb-2 font-mono">参数</div>
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

      {/* 小白教程（整页宽，长内容更好读） */}
      {meta.tutorial && (
        <div className="mt-5">
          <TutorialPanel tutorial={meta.tutorial} />
        </div>
      )}

      {/* 真实源码（浏览器端纯手写实现，无机器学习库） */}
      {demo.source && (
        <div className="mt-5">
          <CodeViewer code={demo.source.code} path={demo.source.path} language="tsx" />
        </div>
      )}
    </div>
  );
}
