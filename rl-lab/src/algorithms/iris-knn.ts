// Iris · KNN 多分类 —— 真实数据集。横轴是 k，看近邻数对三类边界的影响。
import { Trajectory, MultiBoundaryState, Frame } from "@/player/types";
import { IRIS, IRIS_CLASSES, IRIS_FEATURES } from "@/data/iris";
import { computeBounds, makeGrid } from "./grid";

const FX = 2;
const FY = 3;
const K = 3;

export interface IrisKNNOptions {
  kValues?: number[];
}

export function runIrisKNN(opts: IrisKNNOptions = {}): Trajectory<MultiBoundaryState> {
  const { kValues = [1, 3, 5, 9, 15, 25, 41] } = opts;
  const pts = IRIS.map((d) => ({ x: d.x[FX], y: d.x[FY], label: d.y }));
  const bounds = computeBounds(pts);

  // 标准化（让两个特征量纲一致再算距离）
  const mean = [avg(pts.map((p) => p.x)), avg(pts.map((p) => p.y))];
  const std = [stdv(pts.map((p) => p.x), mean[0]), stdv(pts.map((p) => p.y), mean[1])];
  const nx = (v: number, i: number) => (v - mean[i]) / std[i];

  const voteClass = (x: number, y: number, k: number, exclude = -1) => {
    const qx = nx(x, 0);
    const qy = nx(y, 1);
    const d = pts
      .map((p, i) => ({ i, label: p.label, d2: (nx(p.x, 0) - qx) ** 2 + (nx(p.y, 1) - qy) ** 2 }))
      .filter((o) => o.i !== exclude)
      .sort((a, b) => a.d2 - b.d2);
    const cnt = new Array(K).fill(0);
    for (let i = 0; i < k && i < d.length; i++) cnt[d[i].label]++;
    return cnt.indexOf(Math.max(...cnt));
  };

  const frames: Frame<MultiBoundaryState>[] = [];
  for (const k of kValues) {
    const grid = makeGrid(bounds, 48, 38, (x, y) => voteClass(x, y, k));
    let correct = 0;
    for (let i = 0; i < pts.length; i++) if (voteClass(pts[i].x, pts[i].y, k, i) === pts[i].label) correct++;
    frames.push({
      iter: k,
      state: { points: pts, grid, classNames: IRIS_CLASSES, xName: IRIS_FEATURES[FX], yName: IRIS_FEATURES[FY] },
      metrics: { accuracy: correct / pts.length },
    });
  }

  return {
    meta: {
      id: "iris-knn",
      title: "Iris · KNN 多分类",
      family: "multiclass",
      algorithm: "KNN (multiclass)",
      description: "真实数据集。横轴是 k：近邻数越大，三类边界越平滑。",
      tutorial: {
        problem: "用 KNN 在真实的鸢尾花数据上做三分类，并观察 k 怎么影响边界。",
        intuition:
          "判断一朵花属于哪类，就看花瓣尺寸最接近的 k 朵已知花里哪类最多。k 小则边界贴着每个点、易受噪声影响；k 大则边界平滑但可能糊掉细节。",
        watch: [
          "三色点是真实三类鸢尾花，背景是 KNN 的三类决策区",
          "k=1 边界锯齿、可能有孤立小块；k 增大边界变平滑",
          "右侧留一法准确率随 k 变化，存在一个最优 k",
        ],
        concepts: [
          { term: "多数投票", explain: "最近 k 个邻居里票数最多的类别即预测" },
          { term: "特征标准化", explain: "花瓣长宽量纲不同，先标准化再算距离才公平" },
          { term: "偏差-方差", explain: "k 小方差大、k 大偏差大，需折中" },
        ],
        tryThis: "拖到 k=1 和 k=41 对比边界锯齿 vs 平滑；找准确率最高的那个 k。",
      },
      hyperparams: { features: "花瓣长/宽", kRange: `${kValues[0]}–${kValues[kValues.length - 1]}`, samples: IRIS.length },
    },
    frames,
  };
}

const avg = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
const stdv = (a: number[], m: number) => Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length) || 1;
