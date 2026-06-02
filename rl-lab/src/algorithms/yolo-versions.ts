// YOLO 版本演进对比（v1 → v12）。数据核对自 Wikipedia「You Only Look Once」词条 + 各版论文。
// 时间轴 = 版本：拖动看每一代的关键创新、作者、年代与 COCO mAP 趋势。
import { Trajectory, VersionsState, Frame } from "@/player/types";

const WIKI_URL = "https://en.wikipedia.org/wiki/You_Only_Look_Once";
const WIKI_DEF =
  "You Only Look Once (YOLO) 是一系列基于卷积神经网络的实时目标检测系统，2015 年由 Joseph Redmon 等人首次提出，经过多次迭代，成为最流行的目标检测框架之一。（定义来自 Wikipedia）";

// COCO mAP 为「最大模型」的近似代表值，用于展示趋势（不同尺寸/设置会有差异）
const ITEMS = [
  { name: "YOLOv1", year: "2016", org: "Joseph Redmon", innovation: "开山之作：把图切成网格，一次前向(You Only Look Once)同时预测所有框+类别，开创实时检测", metric: 28 },
  { name: "YOLOv2 / YOLO9000", year: "2016", org: "Joseph Redmon", innovation: "引入锚框(anchor)、批归一化、高分辨率分类器；YOLO9000 号称能识别 9000 类", metric: 30 },
  { name: "YOLOv3", year: "2018", org: "Joseph Redmon", innovation: "多尺度检测 + Darknet-53 主干，小目标明显变准（原作者最后一版）", metric: 33 },
  { name: "YOLOv4", year: "2020", org: "Alexey Bochkovskiy", innovation: "CSPDarknet + 大量训练技巧(Bag of Freebies/Specials)，精度速度双升", metric: 43 },
  { name: "YOLOv5", year: "2020", org: "Ultralytics", innovation: "首个 PyTorch 实现、极易用，工程界爆火（无正式论文，存在命名争议）", metric: 50 },
  { name: "YOLOv6", year: "2022", org: "美团 Meituan", innovation: "面向工业部署：anchor-free + 重参数化，端侧友好", metric: 52 },
  { name: "YOLOv7", year: "2022", org: "Wang 等", innovation: "E-ELAN 结构 + 可训练的免费技巧，当时速度/精度 SOTA", metric: 56 },
  { name: "YOLOv8", year: "2023", org: "Ultralytics", innovation: "anchor-free + 统一框架：检测/分割/姿态/分类一套搞定", metric: 53 },
  { name: "YOLOv9", year: "2024", org: "Wang 等", innovation: "PGI 可编程梯度信息 + GELAN，缓解深层信息瓶颈", metric: 55 },
  { name: "YOLOv10", year: "2024", org: "清华大学", innovation: "去掉 NMS 后处理，端到端检测，更快更省", metric: 54, tag: "NMS-free" },
  { name: "YOLOv11", year: "2024", org: "Ultralytics", innovation: "C3k2 模块，效率与精度再优化，多任务支持", metric: 55 },
  { name: "YOLOv12", year: "2025", org: "Tian 等", innovation: "以注意力为中心(Area Attention)，把 Transformer 思想引入 YOLO", metric: 56 },
];

export function runYoloVersions(): Trajectory<VersionsState> {
  const frames: Frame<VersionsState>[] = ITEMS.map((_, i) => ({
    iter: i,
    state: { items: ITEMS, current: i, metricLabel: "COCO mAP（近似）", wikiDef: WIKI_DEF, wikiUrl: WIKI_URL },
    metrics: { map: ITEMS[i].metric, year: Number(ITEMS[i].year) },
  }));

  return {
    meta: {
      id: "yolo-versions",
      title: "YOLO 版本演进（v1 → v12）",
      family: "versions",
      algorithm: "YOLO Family",
      description: "从 2016 的 v1 到 2025 的 v12，拖动时间轴看每一代的关键创新、作者和精度趋势。定义引自 Wikipedia。",
      tutorial: {
        problem: "你以为 YOLO 到 v10 了？其实已经到 v11(2024)、v12(2025)。这条线梳理它十年的进化：每代解决了什么。",
        intuition:
          "YOLO 不是一个模型，而是一个家族。前三代(v1–v3)由原作者 Joseph Redmon 做，奠定「单次检测」范式；之后由不同团队接力(Ultralytics、美团、清华…)，主线是：精度↑、速度↑、再到去掉 NMS 做端到端、最后把注意力机制引进来。注意 mAP 到 v7 后基本见顶，创新转向「更快/更省/更易用/端到端」。",
        watch: [
          "时间轴上每个点是一代 YOLO，拖动看它的年代、作者、关键创新",
          "右侧 mAP 曲线：前期快速上升，v7 后趋于平台——精度见顶，创新转向效率",
          "留意 v10 的「NMS-free」标记：去掉后处理是近年的重要方向",
        ],
        concepts: [
          { term: "单次检测 YOLO", explain: "一次前向同时出所有框，区别于两阶段(R-CNN)的慢" },
          { term: "锚框 anchor", explain: "预设若干框形状再回归微调；v8 起转向 anchor-free" },
          { term: "NMS-free", explain: "去掉非极大值抑制后处理，做到端到端(v10)" },
        ],
        tryThis: "拖时间轴从 v1 走到 v12，对照右侧 mAP 看「精度见顶、创新转向效率」的拐点；点上方蓝标可跳 Wikipedia 原文。",
      },
      hyperparams: { versions: ITEMS.length, span: "2016 → 2025", source: "Wikipedia + 各版论文" },
    },
    frames,
  };
}
