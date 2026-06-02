// 分类评估：ROC 曲线 + 精确率/召回率 + 混淆矩阵 —— 拖动判定阈值看指标怎么变。
// 谷歌 ML 速成课重点：分类好坏不能只看准确率。
import { Trajectory, RocState, Frame } from "@/player/types";
import { mulberry32, gaussian } from "./rng";

export interface RocOptions {
  seed?: number;
  steps?: number;
}

export function runROC(opts: RocOptions = {}): Trajectory<RocState> {
  const { seed = (Date.now() & 0xffff) >>> 0, steps = 30 } = opts;
  const rng = mulberry32(seed);

  // 正/负两类分数分布（重叠的高斯，所以阈值很重要）
  const pos = Array.from({ length: 100 }, () => gaussian(rng, 1.1, 1.0));
  const neg = Array.from({ length: 100 }, () => gaussian(rng, -0.9, 1.0));
  const all = [...pos, ...neg];
  const lo = Math.min(...all) - 0.3;
  const hi = Math.max(...all) + 0.3;

  const at = (thr: number) => {
    const tp = pos.filter((s) => s >= thr).length;
    const fn = pos.length - tp;
    const fp = neg.filter((s) => s >= thr).length;
    const tn = neg.length - fp;
    const tpr = tp / (tp + fn || 1);
    const fpr = fp / (fp + tn || 1);
    const precision = tp / (tp + fp || 1);
    const recall = tpr;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return { tp, fn, fp, tn, tpr, fpr, precision, recall, f1 };
  };

  // 完整 ROC 曲线（细扫阈值）+ AUC（梯形法）
  const sweep = Array.from({ length: 200 }, (_, i) => hi - ((hi - lo) * i) / 199).map((thr) => at(thr));
  const roc = sweep.map((m) => ({ fpr: m.fpr, tpr: m.tpr }));
  let auc = 0;
  for (let i = 1; i < roc.length; i++) auc += ((roc[i].fpr - roc[i - 1].fpr) * (roc[i].tpr + roc[i - 1].tpr)) / 2;
  auc = Math.abs(auc);

  const frames: Frame<RocState>[] = [];
  for (let i = 0; i < steps; i++) {
    const thr = hi - ((hi - lo) * i) / (steps - 1); // 从高阈值（严格）到低阈值（宽松）
    const m = at(thr);
    frames.push({
      iter: i,
      state: {
        threshold: thr,
        pos,
        neg,
        roc,
        current: { fpr: m.fpr, tpr: m.tpr },
        confusion: { tp: m.tp, fp: m.fp, fn: m.fn, tn: m.tn },
        precision: m.precision,
        recall: m.recall,
        f1: m.f1,
        auc,
      },
      metrics: { precision: m.precision, recall: m.recall, f1: m.f1 },
    });
  }

  return {
    meta: {
      id: "roc",
      title: "分类评估 · ROC / 精确率-召回率",
      family: "roc",
      algorithm: "ROC & PR",
      description: "正负两类分数有重叠。拖动判定阈值，看混淆矩阵、精确率、召回率、ROC 怎么联动变化。",
      tutorial: {
        problem: "分类器好不好，只看准确率够吗？不够。这个实验展示阈值、ROC、精确率-召回率的关系。",
        intuition:
          "分类器对每个样本输出一个“分数”，超过阈值就判正类。阈值高=很严格（精确率高但漏掉很多，召回率低）；阈值低=很宽松（召回率高但误报多，精确率低）。ROC 曲线把所有阈值下的(假阳率, 真阳率)连起来，曲线下面积 AUC 衡量整体区分能力（1=完美，0.5=瞎猜）。",
        watch: [
          "上方分数轴：绿=正类、橙=负类，竖线是当前阈值；两类重叠区是难点",
          "阈值从高往低移：召回率上升、精确率下降，混淆矩阵随之变化",
          "ROC 曲线上的点随阈值滑动；AUC 不随阈值变（是整体指标）",
        ],
        concepts: [
          { term: "混淆矩阵", explain: "TP/FP/FN/TN 四格，所有分类指标都从它算出来" },
          { term: "精确率 / 召回率", explain: "精确率=判正里多少真对；召回率=真正类里抓回多少。常此消彼长" },
          { term: "ROC / AUC", explain: "各阈值下(假阳率,真阳率)的曲线；AUC=曲线下面积，越大越好" },
        ],
        tryThis: "拖时间轴移动阈值，盯着精确率和召回率怎么一升一降；找 F1 最高的阈值。",
      },
      hyperparams: { posN: pos.length, negN: neg.length, steps },
    },
    frames,
  };
}
