// 算法过程动画实验台：帧契约 + 通用播放器 + 各家族可视化器。
// 加新算法 = 往 DEMOS 加一条（builder 产出 Trajectory + 指定 Viz 组件），播放器自动复用。
import { ComponentType, useMemo, useState } from "react";
import { RefreshCw, Lightbulb, Route } from "lucide-react";
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
import { runCartPoleControl } from "@/algorithms/cartpole-control";
import { runSVM } from "@/algorithms/svm";
import { runAdaBoost } from "@/algorithms/adaboost";
import { runQLearning } from "@/algorithms/qlearning";
import { runGAN } from "@/algorithms/gan";
import { runDiffusion } from "@/algorithms/diffusion";
import { runHopfield } from "@/algorithms/hopfield";
import { runRNN } from "@/algorithms/rnn";
import { runWord2Vec } from "@/algorithms/word2vec";
import cartpolePPO from "@/data/frames/cartpole-ppo.json";
import pendulumSAC from "@/data/frames/pendulum-sac.json";
import mountaincarPPO from "@/data/frames/mountaincar-ppo.json";
import mountaincarDQN from "@/data/frames/mountaincar-dqn.json";
import mountaincarShaped from "@/data/frames/mountaincar-shaped.json";
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
import cartpoleControlSrc from "@/algorithms/cartpole-control.ts?raw";
import svmSrc from "@/algorithms/svm.ts?raw";
import adaboostSrc from "@/algorithms/adaboost.ts?raw";
import qlearningSrc from "@/algorithms/qlearning.ts?raw";
import ganSrc from "@/algorithms/gan.ts?raw";
import diffusionSrc from "@/algorithms/diffusion.ts?raw";
import hopfieldSrc from "@/algorithms/hopfield.ts?raw";
import rnnSrc from "@/algorithms/rnn.ts?raw";
import word2vecSrc from "@/algorithms/word2vec.ts?raw";
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
import GridWorldPlot from "@/visualizers/GridWorldPlot";
import HopfieldPlot from "@/visualizers/HopfieldPlot";
import RnnPlot from "@/visualizers/RnnPlot";
import Word2VecPlot from "@/visualizers/Word2VecPlot";
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
    key: "svm",
    label: "SVM 支持向量机",
    group: "监督 · 分类",
    build: (seed) => runSVM({ seed }),
    Viz: BoundaryPlot,
    metricKey: "margin",
    metricLabel: "间隔宽度 margin",
    source: { code: svmSrc, path: "algorithms/svm.ts" },
  },
  {
    key: "adaboost",
    label: "AdaBoost 集成",
    group: "监督 · 分类",
    build: (seed) => runAdaBoost({ seed }),
    Viz: BoundaryPlot,
    metricKey: "accuracy",
    metricLabel: "训练准确率",
    source: { code: adaboostSrc, path: "algorithms/adaboost.ts" },
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
    key: "hopfield",
    label: "Hopfield 联想记忆",
    group: "联想记忆（1982）",
    build: (seed) => runHopfield({ seed }),
    Viz: HopfieldPlot,
    metricKey: "overlap",
    metricLabel: "与目标图案吻合度",
    metricColor: "#00ff88",
    source: { code: hopfieldSrc, path: "algorithms/hopfield.ts" },
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
    key: "rnn",
    label: "RNN 序列记忆",
    group: "深度学习",
    build: (seed) => runRNN({ seed }),
    Viz: RnnPlot,
    metricKey: "accuracy",
    metricLabel: "奇偶判断准确率",
    metricColor: "#00e5ff",
    source: { code: rnnSrc, path: "algorithms/rnn.ts" },
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
    key: "word2vec",
    label: "Word2Vec 词向量",
    group: "深度学习",
    build: (seed) => runWord2Vec({ seed }),
    Viz: Word2VecPlot,
    metricKey: "loss",
    metricLabel: "skip-gram Loss",
    metricColor: "#ffab40",
    source: { code: word2vecSrc, path: "algorithms/word2vec.ts" },
  },
  {
    key: "gan",
    label: "GAN 对抗生成",
    group: "生成模型",
    build: (seed) => runGAN({ seed }),
    Viz: ClustersPlot,
    metricKey: "realness",
    metricLabel: "判别器对假点的「像真」分",
    metricColor: "#00ff88",
    source: { code: ganSrc, path: "algorithms/gan.ts" },
  },
  {
    key: "diffusion",
    label: "Diffusion 扩散",
    group: "生成模型",
    build: (seed) => runDiffusion({ seed }),
    Viz: ClustersPlot,
    metricKey: "noiseLevel",
    metricLabel: "噪声水平（去噪进度）",
    metricColor: "#00e5ff",
    source: { code: diffusionSrc, path: "algorithms/diffusion.ts" },
  },
  {
    key: "cartpole-control",
    label: "CartPole · 经典控制",
    group: "经典控制（RL 之前）",
    build: (seed) => runCartPoleControl({ seed }),
    Viz: EnvPlot,
    metricKey: "angle",
    metricLabel: "杆偏离角度（°）",
    metricColor: "#ffab40",
    source: { code: cartpoleControlSrc, path: "algorithms/cartpole-control.ts" },
  },
  {
    key: "qlearning",
    label: "Q-Learning（GridWorld）",
    group: "强化学习 · 表格法",
    build: (seed) => runQLearning({ seed }),
    Viz: GridWorldPlot,
    metricKey: "greedyReturn",
    metricLabel: "贪心策略回报",
    metricColor: "#00ff88",
    source: { code: qlearningSrc, path: "algorithms/qlearning.ts" },
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
    key: "mountaincar-dqn",
    label: "MountainCar · DQN（失败）",
    group: "强化学习 · env",
    build: () => mountaincarDQN as unknown as Trajectory,
    Viz: EnvPlot,
    metricKey: "return",
    metricLabel: "累计回报 Return",
    metricColor: "#ff5252",
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
    key: "mountaincar-shaped",
    label: "MountainCar · Shaped",
    group: "强化学习 · env",
    build: () => mountaincarShaped as unknown as Trajectory,
    Viz: EnvPlot,
    metricKey: "return",
    metricLabel: "累计回报 Return",
    metricColor: "#ffab40",
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

// 学习路线：按 AI 发展史把实验排成一条课程线
const CURRICULUM: { stage: string; keys: string[] }[] = [
  { stage: "① 入门 · 拟合一条线", keys: ["linreg"] },
  { stage: "② 监督学习 · 分类", keys: ["logreg", "perceptron", "svm", "dtree", "adaboost", "knn"] },
  { stage: "③ 无监督学习", keys: ["kmeans", "dbscan", "pca"] },
  { stage: "④ 早期神经网络 · 联想记忆", keys: ["hopfield", "mlp"] },
  { stage: "⑤ 序列与卷积", keys: ["rnn", "cnn-shapes"] },
  { stage: "⑥ 表示学习", keys: ["word2vec"] },
  { stage: "⑦ 大模型时代", keys: ["attention-reverse", "mamba-ssm"] },
  { stage: "⑧ 生成模型", keys: ["gan", "diffusion"] },
  { stage: "⑨ 控制论（RL 之前）", keys: ["cartpole-control"] },
  { stage: "⑩ 强化学习", keys: ["qlearning", "cartpole-ppo", "mountaincar-dqn", "mountaincar-ppo", "mountaincar-shaped", "pendulum-sac"] },
];

const byKey = Object.fromEntries(DEMOS.map((d) => [d.key, d]));
let _n = 0;
const NUMBERED = CURRICULUM.map((s) => ({
  stage: s.stage,
  items: s.keys.filter((k) => byKey[k]).map((k) => ({ key: k, n: ++_n, demo: byKey[k] })),
}));

export default function AlgorithmLab() {
  const [demoKey, setDemoKey] = useState(DEMOS[0].key);
  const [seed, setSeed] = useState(1234);
  const demo = DEMOS.find((d) => d.key === demoKey)!;

  const traj = useMemo(() => demo.build(seed), [demo, seed]);
  const player = useTrajectory(traj);
  const meta = traj.meta;
  const Viz = demo.Viz;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <header className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">
          <span className="neon-text text-[#00ff88]">算法过程动画</span>{" "}
          <span className="text-slate-300">· 实验台</span>
        </h1>
        <p className="text-slate-500 text-sm mt-1.5">
          26 个经典算法，按 AI 发展史排成一条学习路线（左侧）—— 从感知机、SVM、聚类、深度学习、生成模型，到强化学习，一站式边看边学。选一个，按 ▶️ 看它怎么一步步收敛。
        </p>
      </header>

      <div className="flex gap-6 items-start">
        {/* 左侧：学习路线课程导航 */}
        <aside className="w-56 shrink-0 sticky top-6 max-h-[calc(100vh-3rem)] overflow-auto pr-1 hidden md:block">
          <div className="text-xs text-slate-500 mb-3 font-semibold flex items-center gap-1.5">
            <Route className="w-3.5 h-3.5 text-[#00ff88]" /> 学习路线
          </div>
          {NUMBERED.map((s) => (
            <div key={s.stage} className="mb-3">
              <div className="text-[11px] text-[#00e5ff]/80 font-mono mb-1">{s.stage}</div>
              <div className="flex flex-col gap-0.5">
                {s.items.map(({ key, n, demo: d }) => (
                  <button
                    key={key}
                    onClick={() => setDemoKey(key)}
                    className={`text-left text-xs px-2 py-1.5 rounded-md transition-all flex gap-2 border ${
                      demoKey === key
                        ? "bg-[rgba(0,255,136,0.12)] text-[#00ff88] border-[#00ff88]/30"
                        : "text-slate-400 hover:text-slate-200 hover:bg-white/[0.03] border-transparent"
                    }`}
                  >
                    <span className="text-slate-600 font-mono w-5 shrink-0 text-right">{n}</span>
                    <span>{d.label}</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </aside>

        {/* 主内容 */}
        <main className="flex-1 min-w-0">
          {/* 移动端：下拉选实验 */}
          <select
            value={demoKey}
            onChange={(e) => setDemoKey(e.target.value)}
            className="md:hidden w-full mb-4 bg-[rgba(15,23,42,0.7)] border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200"
          >
            {NUMBERED.flatMap((s) => s.items).map(({ key, n, demo: d }) => (
              <option key={key} value={key}>
                {n}. {d.label}
              </option>
            ))}
          </select>

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
        </main>
      </div>
    </div>
  );
}
