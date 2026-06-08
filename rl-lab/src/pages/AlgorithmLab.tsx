// 算法过程动画实验台：帧契约 + 通用播放器 + 各家族可视化器。
// 加新算法 = 往 DEMOS 加一条（builder 产出 Trajectory + 指定 Viz 组件），播放器自动复用。
import { ComponentType, useMemo, useState } from "react";
import { RefreshCw, Lightbulb, GitBranch } from "lucide-react";
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
import { runIrisSoftmax } from "@/algorithms/iris-softmax";
import { runIrisKNN } from "@/algorithms/iris-knn";
import { runOverfitting } from "@/algorithms/overfitting";
import { runRegularization } from "@/algorithms/regularization";
import { runROC } from "@/algorithms/roc";
import { runQwenEmbeddings } from "@/algorithms/qwen-embeddings";
import { runYoloVersions } from "@/algorithms/yolo-versions";
import { runDeerflow } from "@/algorithms/deerflow";
import { runActivations } from "@/algorithms/arch/activations";
import { runAttentionKV } from "@/algorithms/arch/attention-kv";
import { runPosEncoding } from "@/algorithms/arch/pos-encoding";
import { runRagLive, runAgentLive, runAgenticRagLive, runEmbeddingsLive, Progress } from "@/algorithms/live-llm";
import { ollamaReachable } from "@/lib/ollama";
import cartpolePPO from "@/data/frames/cartpole-ppo.json";
import pendulumSAC from "@/data/frames/pendulum-sac.json";
import mountaincarPPO from "@/data/frames/mountaincar-ppo.json";
import mountaincarDQN from "@/data/frames/mountaincar-dqn.json";
import mountaincarShaped from "@/data/frames/mountaincar-shaped.json";
import cnnShapes from "@/data/frames/cnn-shapes.json";
import agentReact from "@/data/agent-react.json";
import ragData from "@/data/rag.json";
import agenticRag from "@/data/agentic-rag.json";
import yoloData from "@/data/yolo.json";
import attentionReverse from "@/data/frames/attention-reverse.json";
import { bridgeMlLab } from "@/algorithms/ml-lab-bridge";
import mnistCnnKernels from "@/data/frames/mnist-cnn-kernels.json";
import mnistGan from "@/data/frames/mnist-gan.json";
import mnistVae from "@/data/frames/mnist-vae.json";
import mnistAutoencoder from "@/data/frames/mnist-autoencoder.json";
import imdbLstm from "@/data/frames/imdb-lstm.json";
import imdbTransformer from "@/data/frames/imdb-transformer.json";
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
import irisSoftmaxSrc from "@/algorithms/iris-softmax.ts?raw";
import irisKnnSrc from "@/algorithms/iris-knn.ts?raw";
import overfittingSrc from "@/algorithms/overfitting.ts?raw";
import regularizationSrc from "@/algorithms/regularization.ts?raw";
import rocSrc from "@/algorithms/roc.ts?raw";
import qwenEmbSrc from "@/algorithms/qwen-embeddings.ts?raw";
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
import MultiBoundaryPlot from "@/visualizers/MultiBoundaryPlot";
import CurveFitPlot from "@/visualizers/CurveFitPlot";
import RocPlot from "@/visualizers/RocPlot";
import AgentPlot from "@/visualizers/AgentPlot";
import RagPlot from "@/visualizers/RagPlot";
import YoloPlot from "@/visualizers/YoloPlot";
import VersionsPlot from "@/visualizers/VersionsPlot";
import ImageGrid from "@/visualizers/ImageGrid";
import AttentionHeatmap from "@/visualizers/AttentionHeatmap";
import Cifar10Plot from "@/visualizers/Cifar10Plot";
import ActivationViz from "@/visualizers/arch/ActivationViz";
import AttnKVViz from "@/visualizers/arch/AttnKVViz";
import PosEncodingViz from "@/visualizers/arch/PosEncodingViz";
import MetricCurve from "@/visualizers/MetricCurve";
import TutorialPanel from "@/components/TutorialPanel";
import CodeViewer from "@/components/CodeViewer";
import BackgroundPanel from "@/components/BackgroundPanel";
import PaperRefs from "@/components/PaperRefs";
import { PAPERS } from "@/data/papers";
import { BACKGROUNDS } from "@/data/backgrounds";

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
  /** 实时运行器：调用本地 Ollama，边跑边 emit 预览，返回完整 Trajectory */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  live?: (emit: (state: any, p: Progress) => void) => Promise<Trajectory>;
  metricKey2?: string;
  metricLabel2?: string;
  metricColor2?: string;
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
    key: "iris-softmax",
    label: "Iris · Softmax",
    group: "真实数据",
    build: () => runIrisSoftmax(),
    Viz: MultiBoundaryPlot,
    metricKey: "accuracy",
    metricLabel: "分类准确率",
    metricColor: "#00e5ff",
    source: { code: irisSoftmaxSrc, path: "algorithms/iris-softmax.ts" },
  },
  {
    key: "iris-knn",
    label: "Iris · KNN",
    group: "真实数据",
    build: () => runIrisKNN(),
    Viz: MultiBoundaryPlot,
    metricKey: "accuracy",
    metricLabel: "留一法准确率",
    metricColor: "#b388ff",
    source: { code: irisKnnSrc, path: "algorithms/iris-knn.ts" },
  },
  {
    key: "overfitting",
    label: "过拟合与泛化",
    group: "评估与泛化",
    build: (seed) => runOverfitting({ seed }),
    Viz: CurveFitPlot,
    metricKey: "testError",
    metricLabel: "误差（训练 vs 测试）",
    metricColor: "#ff5252",
    metricKey2: "trainError",
    metricLabel2: "训练误差",
    metricColor2: "#00e5ff",
    source: { code: overfittingSrc, path: "algorithms/overfitting.ts" },
  },
  {
    key: "regularization",
    label: "正则化 L2",
    group: "评估与泛化",
    build: (seed) => runRegularization({ seed }),
    Viz: CurveFitPlot,
    metricKey: "testError",
    metricLabel: "误差（训练 vs 测试）",
    metricColor: "#ff5252",
    metricKey2: "trainError",
    metricLabel2: "训练误差",
    metricColor2: "#00e5ff",
    source: { code: regularizationSrc, path: "algorithms/regularization.ts" },
  },
  {
    key: "roc",
    label: "ROC / 精确率-召回率",
    group: "评估与泛化",
    build: (seed) => runROC({ seed }),
    Viz: RocPlot,
    metricKey: "precision",
    metricLabel: "精确率 vs 召回率",
    metricColor: "#00ff88",
    metricKey2: "recall",
    metricLabel2: "召回率",
    metricColor2: "#ffab40",
    source: { code: rocSrc, path: "algorithms/roc.ts" },
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
    key: "rag",
    label: "RAG 检索增强",
    group: "大模型时代",
    build: () => ragData as unknown as Trajectory,
    live: runRagLive,
    Viz: RagPlot,
    metricKey: "topScore",
    metricLabel: "最高相似度",
    metricColor: "#00e5ff",
  },
  {
    key: "agent-react",
    label: "Agent · ReAct",
    group: "Agent 时代",
    build: () => agentReact as unknown as Trajectory,
    live: runAgentLive,
    Viz: AgentPlot,
    metricKey: "step",
    metricLabel: "推理步骤",
    metricColor: "#b388ff",
  },
  {
    key: "agentic-rag",
    label: "Agentic RAG · 多跳",
    group: "Agent 时代",
    build: () => agenticRag as unknown as Trajectory,
    live: runAgenticRagLive,
    Viz: AgentPlot,
    metricKey: "step",
    metricLabel: "推理步骤",
    metricColor: "#00ff88",
  },
  {
    key: "deerflow",
    label: "DeerFlow 深度研究",
    group: "Agent 时代",
    build: () => runDeerflow(),
    Viz: AgentPlot,
    metricKey: "step",
    metricLabel: "研究步骤",
    metricColor: "#ffab40",
  },
  {
    key: "yolo",
    label: "YOLO 目标检测",
    group: "深度学习",
    build: () => yoloData as unknown as Trajectory,
    Viz: YoloPlot,
    metricKey: "detections",
    metricLabel: "检测框数量（随阈值）",
    metricColor: "#00e5ff",
  },
  {
    key: "yolo-versions",
    label: "YOLO 版本演进",
    group: "深度学习",
    build: () => runYoloVersions(),
    Viz: VersionsPlot,
    metricKey: "map",
    metricLabel: "COCO mAP（随版本，近似）",
    metricColor: "#00e5ff",
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
    key: "mnist-cnn-kernels",
    label: "MNIST · CNN 卷积核",
    group: "深度学习",
    build: () => bridgeMlLab(mnistCnnKernels),
    Viz: ImageGrid,
    metricKey: "val_accuracy",
    metricLabel: "验证准确率",
    metricColor: "#00e5ff",
  },
  {
    key: "mnist-autoencoder",
    label: "MNIST · 自编码器",
    group: "深度学习",
    build: () => bridgeMlLab(mnistAutoencoder),
    Viz: ImageGrid,
    metricKey: "recon_loss",
    metricLabel: "重建损失",
    metricColor: "#ffab40",
  },
  {
    key: "mnist-vae",
    label: "MNIST · 变分自编码器 VAE",
    group: "生成模型",
    build: () => bridgeMlLab(mnistVae),
    Viz: ImageGrid,
    metricKey: "elbo_loss",
    metricLabel: "ELBO 损失",
    metricColor: "#b388ff",
  },
  {
    key: "mnist-gan",
    label: "MNIST · GAN 生成",
    group: "生成模型",
    build: () => bridgeMlLab(mnistGan),
    Viz: ImageGrid,
    metricKey: "g_loss",
    metricLabel: "生成器损失",
    metricColor: "#00ff88",
    metricKey2: "d_loss",
    metricLabel2: "判别器损失",
    metricColor2: "#ff5252",
  },
  {
    key: "imdb-transformer",
    label: "IMDb · Tiny Transformer 注意力",
    group: "深度学习",
    build: () => bridgeMlLab(imdbTransformer),
    Viz: AttentionHeatmap,
    metricKey: "val_accuracy",
    metricLabel: "验证准确率",
    metricColor: "#00e5ff",
  },
  {
    key: "imdb-lstm",
    label: "IMDb · LSTM 隐状态",
    group: "深度学习",
    build: () => bridgeMlLab(imdbLstm),
    Viz: ImageGrid,
    metricKey: "val_accuracy",
    metricLabel: "验证准确率",
    metricColor: "#ffab40",
  },
  {
    key: "cifar10-cnn",
    label: "CIFAR-10 · CNN 十分类",
    group: "深度学习",
    build: () => ({ meta: { id: "cifar10-cnn", title: "CIFAR-10 · CNN 十分类", family: "cnn", algorithm: "CNN" }, frames: [{ iter: 0, state: {}, metrics: {} }] }) as unknown as Trajectory,
    Viz: Cifar10Plot,
    metricKey: "accuracy",
    metricLabel: "测试准确率",
    metricColor: "#00e5ff",
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
    key: "qwen-embeddings",
    label: "本地 Qwen 语义嵌入",
    group: "大模型时代",
    build: (seed) => runQwenEmbeddings({ seed }),
    live: runEmbeddingsLive,
    Viz: Word2VecPlot,
    metricKey: "stress",
    metricLabel: "MDS 布局误差 stress",
    metricColor: "#b388ff",
    source: { code: qwenEmbSrc, path: "algorithms/qwen-embeddings.ts" },
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
  {
    key: "activations",
    label: "激活函数",
    group: "LLM 架构解剖",
    build: () => runActivations(),
    Viz: ActivationViz,
    metricKey: "noop",
    metricLabel: "（交互演示 · 无训练曲线）",
    metricColor: "#38bdf8",
  },
  {
    key: "attention-kv",
    label: "注意力 KV 变体",
    group: "LLM 架构解剖",
    build: () => runAttentionKV(),
    Viz: AttnKVViz,
    metricKey: "noop",
    metricLabel: "（交互演示 · 无训练曲线）",
    metricColor: "#38bdf8",
  },
  {
    key: "pos-encoding",
    label: "位置编码",
    group: "LLM 架构解剖",
    build: () => runPosEncoding(),
    Viz: PosEncodingViz,
    metricKey: "noop",
    metricLabel: "（交互演示 · 无训练曲线）",
    metricColor: "#38bdf8",
  },
];

// 发展脉络：按 AI 发展史组成一棵分叉树（分支=方法谱系，叶子=具体实验）
interface TreeNode {
  label?: string; // 分支名（无 key 时）
  era?: string; // 年代标注
  key?: string; // 叶子 = 某个 demo 的 key
  children?: TreeNode[];
}

const TREE: TreeNode[] = [
  {
    label: "统计 · 符号学习",
    children: [
      { label: "线性模型", children: [{ key: "linreg" }, { key: "logreg" }, { key: "perceptron", era: "1958" }] },
      { label: "间隔 · 核方法", children: [{ key: "svm", era: "1995" }] },
      { label: "实例 · 树 · 集成", children: [{ key: "knn" }, { key: "dtree" }, { key: "adaboost", era: "1995" }] },
      { label: "无监督", children: [{ key: "kmeans" }, { key: "dbscan" }, { key: "pca" }] },
      { label: "真实数据 · Iris", children: [{ key: "iris-softmax", era: "1936" }, { key: "iris-knn" }] },
    ],
  },
  {
    label: "数据 · 评估 · 泛化",
    children: [
      { label: "泛化", children: [{ key: "overfitting" }, { key: "regularization" }] },
      { label: "分类评估", children: [{ key: "roc" }] },
    ],
  },
  {
    label: "联结主义 · 神经网络",
    children: [
      { label: "联想记忆", children: [{ key: "hopfield", era: "1982" }] },
      { label: "前馈 · 反向传播", children: [{ key: "mlp", era: "1986" }] },
      { label: "序列", children: [{ key: "rnn", era: "1997" }] },
      { label: "视觉 · 卷积", children: [{ key: "cnn-shapes", era: "1998" }, { key: "yolo", era: "2016" }, { key: "yolo-versions", era: "v1→v12" }] },
      { label: "真实数据集 · 视觉", children: [{ key: "mnist-cnn-kernels", era: "MNIST" }, { key: "mnist-autoencoder" }, { key: "cifar10-cnn", era: "CIFAR-10" }] },
      { label: "真实数据集 · 文本", children: [{ key: "imdb-lstm", era: "IMDb" }, { key: "imdb-transformer" }] },
      { label: "表示学习", children: [{ key: "word2vec", era: "2013" }] },
    ],
  },
  {
    label: "大模型时代",
    children: [
      { label: "注意力", children: [{ key: "attention-reverse", era: "2017" }] },
      { label: "后注意力 · 线性时间", children: [{ key: "mamba-ssm", era: "2023" }] },
      { label: "嵌入 · 语义（本地 Qwen）", children: [{ key: "qwen-embeddings" }] },
      { label: "检索增强 RAG（本地 Qwen）", children: [{ key: "rag" }] },
    ],
  },
  {
    label: "Agent 时代",
    children: [
      { label: "ReAct · 工具调用（本地 Qwen）", children: [{ key: "agent-react" }, { key: "agentic-rag" }] },
      { label: "深度研究 Agent（本地 DeerFlow）", children: [{ key: "deerflow" }] },
    ],
  },
  {
    label: "生成模型",
    children: [
      { label: "对抗生成", children: [{ key: "gan", era: "2014" }, { key: "mnist-gan", era: "MNIST" }] },
      { label: "变分自编码", children: [{ key: "mnist-vae", era: "2013" }] },
      { label: "扩散", children: [{ key: "diffusion", era: "2020" }] },
    ],
  },
  {
    label: "决策与控制",
    children: [
      { label: "经典控制 · RL 之前", children: [{ key: "cartpole-control", era: "1960s" }] },
      {
        label: "强化学习",
        children: [
          { label: "价值法", children: [{ key: "qlearning", era: "1989" }, { key: "mountaincar-dqn", era: "2013" }] },
          { label: "策略法", children: [{ key: "cartpole-ppo" }, { key: "mountaincar-ppo" }] },
          { label: "Actor-Critic · 连续", children: [{ key: "pendulum-sac" }] },
          { label: "奖励工程", children: [{ key: "mountaincar-shaped" }] },
        ],
      },
    ],
  },
  {
    label: "LLM 架构解剖（Transformer 内部）",
    children: [
      { label: "FFN · 激活", children: [{ key: "activations" }] },
      { label: "注意力 · KV 共享", children: [{ key: "attention-kv" }] },
      { label: "位置编码", children: [{ key: "pos-encoding" }] },
    ],
  },
];

const byKey = Object.fromEntries(DEMOS.map((d) => [d.key, d]));

// 实验由什么驱动：tianshou=用清华 Tianshou 的 RL 算法；pytorch=纯 PyTorch 训练；
// 其余（不在表里的）都是浏览器端纯手写、零库。
const ENGINE: Record<string, "tianshou" | "pytorch" | "ollama" | "deerflow"> = {
  "qwen-embeddings": "ollama",
  "agent-react": "ollama",
  rag: "ollama",
  "agentic-rag": "ollama",
  deerflow: "deerflow",
  "cartpole-ppo": "tianshou",
  "mountaincar-dqn": "tianshou",
  "mountaincar-ppo": "tianshou",
  "mountaincar-shaped": "tianshou",
  "pendulum-sac": "tianshou",
  "cnn-shapes": "pytorch",
  "attention-reverse": "pytorch",
  yolo: "pytorch",
  "mnist-cnn-kernels": "pytorch",
  "mnist-autoencoder": "pytorch",
  "mnist-vae": "pytorch",
  "mnist-gan": "pytorch",
  "imdb-transformer": "pytorch",
  "imdb-lstm": "pytorch",
  "cifar10-cnn": "pytorch",
};

// 直接对应 Google ML 速成课「数据 / 评估 / 泛化」模块的实验
const GOOGLE_ML = new Set(["overfitting", "regularization", "roc"]);

// 定义/资料引自 Wikipedia 的实验 → 词条链接
const WIKI: Record<string, string> = {
  "yolo-versions": "https://en.wikipedia.org/wiki/You_Only_Look_Once",
  yolo: "https://en.wikipedia.org/wiki/You_Only_Look_Once",
};

// 把树拍平成叶子列表（供移动端下拉 + 计数）
const LEAVES: TreeNode[] = [];
(function walk(ns: TreeNode[]) {
  for (const n of ns) {
    if (n.key && byKey[n.key]) LEAVES.push(n);
    else if (n.children) walk(n.children);
  }
})(TREE);

export default function AlgorithmLab() {
  const [demoKey, setDemoKey] = useState(DEMOS[0].key);
  const [seed, setSeed] = useState(1234);
  const demo = DEMOS.find((d) => d.key === demoKey)!;

  const buildTraj = useMemo(() => demo.build(seed), [demo, seed]);

  // 实时运行（真·调用本地 Ollama Qwen）状态。把 demoKey 绑进结果，渲染时按 key 匹配，
  // 避免切换实验后的那次渲染用「旧 state + 新 Viz」而崩。
  const [liveResult, setLiveResult] = useState<{ key: string; traj: Trajectory } | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [livePreview, setLivePreview] = useState<{ key: string; state: any; p: Progress } | null>(null);
  const [running, setRunning] = useState<string | null>(null); // 正在跑的 demoKey
  const [liveError, setLiveError] = useState<{ key: string; msg: string } | null>(null);

  // 只采用属于「当前实验」的实时数据
  const myLiveTraj = liveResult && liveResult.key === demoKey ? liveResult.traj : null;
  const myPreview = livePreview && livePreview.key === demoKey ? livePreview : null;
  const isRunning = running === demoKey;
  const myError = liveError && liveError.key === demoKey ? liveError.msg : null;

  const traj = myLiveTraj ?? buildTraj;
  const player = useTrajectory(traj);
  const meta = traj.meta;
  const Viz = demo.Viz;

  const runLive = async () => {
    if (!demo.live || isRunning) return;
    const key = demoKey;
    setRunning(key);
    setLiveError(null);
    setLivePreview(null);
    setLiveResult(null);
    if (!(await ollamaReachable())) {
      setLiveError({ key, msg: "连不上本地 Ollama。请确认 Ollama 在 localhost:11434 运行，且本前端是用 npm run dev 起的（带 /ollama 代理）。" });
      setRunning((r) => (r === key ? null : r));
      return;
    }
    try {
      const t = await demo.live((state, p) => setLivePreview({ key, state, p }));
      setLiveResult({ key, traj: t });
    } catch (e) {
      setLiveError({ key, msg: e instanceof Error ? e.message : String(e) });
    } finally {
      setRunning((r) => (r === key ? null : r));
      setLivePreview((lp) => (lp && lp.key === key ? null : lp));
    }
  };

  // 递归渲染发展脉络树：分支=方法谱系，叶子=可点击的实验
  const renderTree = (nodes: TreeNode[], depth = 0) =>
    nodes.map((node, i) => {
      if (node.key) {
        const d = byKey[node.key];
        if (!d) return null;
        const active = demoKey === node.key;
        return (
          <button
            key={node.key}
            onClick={() => setDemoKey(node.key!)}
            className={`w-full text-left text-xs px-2 py-1.5 rounded-md transition-all flex items-center gap-2 border ${
              active
                ? "bg-[rgba(0,255,136,0.12)] text-[#00ff88] border-[#00ff88]/30"
                : "text-slate-400 hover:text-slate-200 hover:bg-white/[0.03] border-transparent"
            }`}
          >
            <span className={active ? "text-[#00ff88]" : "text-slate-600"}>•</span>
            <span className="flex-1">{d.label}</span>
            {ENGINE[node.key] === "tianshou" && (
              <span className="text-[9px] font-mono px-1 rounded bg-[rgba(0,229,255,0.15)] text-[#00e5ff] border border-[#00e5ff]/30">
                ts
              </span>
            )}
            {node.era && <span className="text-[10px] text-slate-600 font-mono">{node.era}</span>}
          </button>
        );
      }
      return (
        <div key={(node.label ?? "") + i} className={depth === 0 ? "mb-3" : "mt-1.5"}>
          <div
            className={
              depth === 0
                ? "text-[11px] text-[#00e5ff]/80 font-semibold mb-1"
                : "text-[11px] text-slate-500 mb-0.5"
            }
          >
            {node.label}
          </div>
          <div className="ml-1.5 pl-2 border-l border-slate-800 flex flex-col gap-0.5">
            {renderTree(node.children!, depth + 1)}
          </div>
        </div>
      );
    });

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <header className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">
          <span className="neon-text text-[#00ff88]">算法过程动画</span>{" "}
          <span className="text-slate-300">· 实验台</span>
        </h1>
        <p className="text-slate-500 text-sm mt-1.5">
          26 个经典算法，按 AI 发展脉络组成一棵树（左侧）—— 统计学习、联结主义、大模型、生成模型、强化学习各成一支，分支即方法谱系。选一个，按 ▶️ 看它怎么一步步收敛。
        </p>
      </header>

      <div className="flex gap-6 items-start">
        {/* 左侧：AI 发展脉络树 */}
        <aside className="w-60 shrink-0 sticky top-6 max-h-[calc(100vh-3rem)] overflow-auto pr-1 hidden md:block">
          <div className="text-xs text-slate-500 mb-3 font-semibold flex items-center gap-1.5">
            <GitBranch className="w-3.5 h-3.5 text-[#00ff88]" /> AI 发展脉络
          </div>
          {renderTree(TREE)}
        </aside>

        {/* 主内容 */}
        <main className="flex-1 min-w-0">
          {/* 移动端：下拉选实验 */}
          <select
            value={demoKey}
            onChange={(e) => setDemoKey(e.target.value)}
            className="md:hidden w-full mb-4 bg-[rgba(15,23,42,0.7)] border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200"
          >
            {LEAVES.map((node) => (
              <option key={node.key} value={node.key}>
                {byKey[node.key!].label}
              </option>
            ))}
          </select>

          <div className="flex items-end justify-between gap-4 mb-4">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-lg font-bold text-slate-100">{meta.title}</h2>
            {ENGINE[demoKey] === "tianshou" && (
              <a
                href="https://github.com/thu-ml/tianshou"
                target="_blank"
                rel="noreferrer"
                title="该实验的 RL 算法由清华 Tianshou 提供（PPO/DQN/SAC + 训练基建）"
                className="inline-flex items-center gap-1 text-[11px] font-mono px-1.5 py-0.5 rounded bg-[rgba(0,229,255,0.12)] text-[#00e5ff] border border-[#00e5ff]/30 hover:bg-[rgba(0,229,255,0.2)] transition-all"
              >
                ⚡ Tianshou 驱动
              </a>
            )}
            {ENGINE[demoKey] === "pytorch" && (
              <span
                title="该实验由纯 PyTorch 训练（非 Tianshou）"
                className="inline-flex items-center gap-1 text-[11px] font-mono px-1.5 py-0.5 rounded bg-[rgba(255,171,64,0.12)] text-[#ffab40] border border-[#ffab40]/30"
              >
                🔥 PyTorch
              </span>
            )}
            {ENGINE[demoKey] === "ollama" && (
              <span
                title="数据来自本地 Ollama 的 Qwen 模型（真实大模型嵌入）"
                className="inline-flex items-center gap-1 text-[11px] font-mono px-1.5 py-0.5 rounded bg-[rgba(179,136,255,0.14)] text-[#b388ff] border border-[#b388ff]/30"
              >
                🦙 本地 Qwen
              </span>
            )}
            {ENGINE[demoKey] === "deerflow" && (
              <span
                title="从本地 DeerFlow 数据库读出的真实研究轨迹（DeepSeek 驱动，只读未触碰服务）"
                className="inline-flex items-center gap-1 text-[11px] font-mono px-1.5 py-0.5 rounded bg-[rgba(255,171,64,0.14)] text-[#ffab40] border border-[#ffab40]/30"
              >
                🦌 DeerFlow · 真实运行
              </span>
            )}
            {!ENGINE[demoKey] && (
              <span
                title="浏览器端纯手写实现，未使用任何机器学习库"
                className="inline-flex items-center gap-1 text-[11px] font-mono px-1.5 py-0.5 rounded bg-[rgba(0,255,136,0.1)] text-[#00ff88]/80 border border-[#00ff88]/25"
              >
                🌐 浏览器手写
              </span>
            )}
            {GOOGLE_ML.has(demoKey) && (
              <a
                href="https://developers.google.com/machine-learning/crash-course?hl=zh-cn"
                target="_blank"
                rel="noreferrer"
                title="对应 Google 机器学习速成课的「数据 / 评估 / 泛化」模块"
                className="inline-flex items-center gap-1 text-[11px] font-mono px-1.5 py-0.5 rounded bg-[rgba(66,133,244,0.15)] text-[#4285f4] border border-[#4285f4]/40 hover:bg-[rgba(66,133,244,0.25)] transition-all"
              >
                📘 Google ML 速成课
              </a>
            )}
            {WIKI[demoKey] && (
              <a
                href={WIKI[demoKey]}
                target="_blank"
                rel="noreferrer"
                title="定义/资料引自 Wikipedia"
                className="inline-flex items-center gap-1 text-[11px] font-mono px-1.5 py-0.5 rounded bg-[rgba(148,163,184,0.12)] text-slate-300 border border-slate-500/40 hover:bg-[rgba(148,163,184,0.2)] transition-all"
              >
                📚 Wikipedia
              </a>
            )}
          </div>
          {meta.description && (
            <p className="text-sm text-slate-500 mt-0.5">{meta.description}</p>
          )}
        </div>
        {demo.live ? (
          <button
            onClick={runLive}
            disabled={isRunning}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border transition-all shrink-0 ${
              isRunning
                ? "border-slate-700 text-slate-500 cursor-wait"
                : "border-[#ff5252]/50 text-[#ff5252] hover:bg-[rgba(255,82,82,0.1)]"
            }`}
            title="真的调用你本地 Ollama 的 Qwen 跑一遍（非预计算）"
          >
            <span className={isRunning ? "animate-pulse" : ""}>🔴</span>{" "}
            {isRunning ? "运行中…" : myLiveTraj ? "再跑一次" : "实时运行（本地 Qwen）"}
          </button>
        ) : (
          <button
            onClick={() => setSeed((s) => s + 1)}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border border-slate-700 text-slate-300 hover:border-[#00e5ff]/50 hover:text-[#00e5ff] transition-all shrink-0"
            title="换一组随机数据重新计算"
          >
            <RefreshCw className="w-3.5 h-3.5" /> 重新生成数据
          </button>
        )}
      </div>

      {/* 实时运行提示条 */}
      {demo.live && (
        <div className="mb-4 text-xs rounded-lg px-3 py-2 bg-[rgba(255,82,82,0.06)] border border-[#ff5252]/20 text-slate-400">
          {myError ? (
            <span className="text-[#ff5252]">⚠ {myError}</span>
          ) : isRunning && myPreview ? (
            <span>
              <span className="text-[#ff5252] animate-pulse">🔴 实时运行中</span> · {myPreview.p.label}
              {myPreview.p.total > 1 && ` （${myPreview.p.step}/${myPreview.p.total}）`}
            </span>
          ) : myLiveTraj ? (
            <span className="text-[#00ff88]">✓ 以上是刚才用本地 Qwen 实时跑出来的结果（非预计算）。点「再跑一次」重跑。</span>
          ) : (
            <span>下方是示例结果。点右上角 <span className="text-[#ff5252]">🔴 实时运行</span> 用你本地 Ollama 的 Qwen 真的跑一遍（生成类约 20-30s/步）。</span>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="lg:col-span-2 flex flex-col gap-4">
          <Viz state={isRunning && myPreview ? myPreview.state : player.frame.state} meta={meta} />
          {!(isRunning && myPreview) && (
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
          )}
        </div>

        <div className="flex flex-col gap-4">
          <MetricCurve
            frames={traj.frames}
            index={player.index}
            metricKey={demo.metricKey}
            label={demo.metricLabel}
            color={demo.metricColor}
            metricKey2={demo.metricKey2}
            label2={demo.metricLabel2}
            color2={demo.metricColor2}
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

          {/* 现实意义 + 小白教程：宽屏并排，行长更舒适、减少纵向滚动 */}
          <div className="mt-5 grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
            {BACKGROUNDS[demoKey] && <BackgroundPanel bg={BACKGROUNDS[demoKey]} />}
            {PAPERS[demoKey] && <PaperRefs refs={PAPERS[demoKey]} />}
            {meta.tutorial && <TutorialPanel tutorial={meta.tutorial} />}
          </div>

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
