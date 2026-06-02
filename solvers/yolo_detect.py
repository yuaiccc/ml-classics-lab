"""视觉 · YOLO 实时目标检测 —— 用 ultralytics 的 yolov8n 真实跑一张图。
导出所有检测框（含置信度），前端做「置信度阈值」动画：阈值升高时低置信的框逐个消失。

运行：  conda run -n tianshou python solvers/yolo_detect.py
输出：  rl-lab/public/yolo/scene.jpg + rl-lab/src/data/yolo.json
"""

import json
import os
import shutil

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, "..", "rl-lab", "public", "yolo")
os.makedirs(PUB, exist_ok=True)


def main():
    model = YOLO("yolov8n.pt")  # 首次运行自动下载 ~6MB

    # ultralytics 自带测试图（街景：人 + 公交车）
    import ultralytics

    asset = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")
    img_out = os.path.join(PUB, "scene.jpg")
    shutil.copy(asset, img_out)

    res = model(asset, conf=0.05, verbose=False)[0]
    H, W = res.orig_shape  # (h, w)
    names = res.names

    boxes = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        boxes.append(
            {
                "x": round(x1 / W, 4),
                "y": round(y1 / H, 4),
                "w": round((x2 - x1) / W, 4),
                "h": round((y2 - y1) / H, 4),
                "label": names[int(b.cls[0])],
                "conf": round(float(b.conf[0]), 4),
            }
        )
    boxes.sort(key=lambda d: -d["conf"])
    print(f"图 {W}x{H} | 检测到 {len(boxes)} 个框：")
    for bx in boxes:
        print(f"  {bx['conf']:.2f}  {bx['label']}")

    # 帧：置信度阈值从低到高扫，低置信框逐个消失
    steps = 22
    frames = []
    for i in range(steps):
        thr = 0.05 + (0.82 - 0.05) * i / (steps - 1)
        shown = sum(1 for b in boxes if b["conf"] >= thr)
        frames.append(
            {
                "iter": i,
                "state": {"image": "/yolo/scene.jpg", "imgW": W, "imgH": H, "threshold": round(thr, 3), "boxes": boxes},
                "metrics": {"threshold": round(thr, 3), "detections": shown},
            }
        )

    meta = {
        "id": "yolo",
        "title": "YOLO · 实时目标检测",
        "family": "yolo",
        "algorithm": "YOLOv8 (ultralytics)",
        "envId": "yolov8n",
        "description": "真实跑 yolov8n 检测一张街景：一次前向就框出所有物体。拖动置信度阈值，看低置信的框如何被筛掉。",
        "insight": "YOLO = You Only Look Once：把图切成网格，一次前向就同时预测所有物体的框和类别，所以能实时。置信度阈值决定保留哪些框——太低误检多，太高漏检。",
        "tutorial": {
            "problem": "怎么让机器实时「看见」并框出画面里所有物体？这是自动驾驶、安防、质检的基础。",
            "intuition": "传统检测要在图上滑很多窗口、跑很多次很慢。YOLO（You Only Look Once）把整张图切成网格，一次前向传播就同时预测每个位置的边界框 + 类别 + 置信度，因此能做到实时（几十帧/秒）。每个框都带一个「置信度」，最后用阈值筛掉不靠谱的框。",
            "watch": [
                "图上的彩色框是 yolov8n 真实检测出的物体（人、公交车等），数字是置信度",
                "拖动时间轴=提高置信度阈值：低置信的框逐个消失",
                "阈值太低→误检（框乱七八糟）；太高→漏检（真物体也没框）——需要折中",
            ],
            "concepts": [
                {"term": "单次检测 (You Only Look Once)", "explain": "一次前向同时输出所有框，所以快、能实时"},
                {"term": "置信度阈值", "explain": "保留置信度高于阈值的框，权衡误检与漏检"},
                {"term": "边界框 + 类别", "explain": "每个检测=位置(框)+ 是什么(类别)+ 多确信(置信度)"},
            ],
            "tryThis": "拖时间轴把阈值从 0.05 一路拉高，数一数检测框怎么从一堆减到只剩最确信的几个（公交车 + 几个人）。",
        },
        "hyperparams": {"model": "yolov8n", "minConf": 0.05, "classes": "COCO 80 类"},
    }

    out = os.path.join(HERE, "..", "rl-lab", "src", "data", "yolo.json")
    json.dump({"meta": meta, "frames": frames}, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("写出", os.path.relpath(out, os.path.join(HERE, "..")), "| 帧", len(frames))


if __name__ == "__main__":
    main()
