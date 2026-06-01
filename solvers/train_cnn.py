"""M4 · CNN 卷积核可视化（纯 PyTorch，无需 torchvision / 数据下载）。

合成数据：12×12 灰度图，类 0 = 横线、类 1 = 竖线（随机位置/粗细 + 噪声）。
训练一个小 CNN，每个 epoch 导出：第一层卷积核 + 一个样本的激活图 + loss/准确率。
动画看点：卷积核从随机噪声，逐渐学成“横边 / 竖边检测器”。

运行：  conda run -n tianshou python solvers/train_cnn.py
输出：  rl-lab/src/data/frames/cnn-shapes.json
"""

import numpy as np
import torch
from torch import nn

from frames_io import write_trajectory

SIZE = 12
N_FILTERS = 6
SEED = 0


def make_dataset(n_per_class: int = 300, seed: int = SEED):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label in (0, 1):  # 0=横线, 1=竖线
        for _ in range(n_per_class):
            img = rng.normal(0.0, 0.06, size=(SIZE, SIZE)).astype(np.float32)
            thick = rng.integers(1, 3)
            pos = rng.integers(1, SIZE - thick - 1)
            if label == 0:
                img[pos : pos + thick, :] += 1.0  # 横线
            else:
                img[:, pos : pos + thick] += 1.0  # 竖线
            X.append(img)
            y.append(label)
    X = np.stack(X)[:, None, :, :]  # [N,1,H,W]
    y = np.array(y, dtype=np.int64)
    idx = rng.permutation(len(y))
    return torch.tensor(X[idx]), torch.tensor(y[idx])


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, N_FILTERS, kernel_size=3)  # -> 6×10×10
        self.pool = nn.MaxPool2d(2)  # -> 6×5×5
        self.fc = nn.Linear(N_FILTERS * 5 * 5, 2)

    def features(self, x):
        return torch.relu(self.conv1(x))  # 激活图（ReLU 后）

    def forward(self, x):
        a = self.pool(self.features(x))
        return self.fc(a.flatten(1))


def grid(t: torch.Tensor):
    """torch 张量 -> 嵌套 python list（前端直接用）"""
    return [[round(float(v), 4) for v in row] for row in t]


def main():
    torch.manual_seed(SEED)
    X, y = make_dataset()
    Xtr, ytr = X[:400], y[:400]

    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    lossf = nn.CrossEntropyLoss()

    # 固定挑一个“竖线”样本，观察它的激活图如何被竖边卷积核点亮
    sample_idx = int((y == 1).nonzero()[0])
    sample = X[sample_idx : sample_idx + 1]
    sample_label = int(y[sample_idx])

    epochs = 24
    frames = []
    for e in range(epochs + 1):
        # 记录当前帧
        with torch.no_grad():
            logits = model(Xtr)
            loss = lossf(logits, ytr).item()
            acc = (logits.argmax(1) == ytr).float().mean().item()
            acts = model.features(sample)[0]  # [6,10,10]
            filters = model.conv1.weight.detach()[:, 0]  # [6,3,3]
            pred = int(model(sample).argmax(1))
        frames.append(
            {
                "iter": e,
                "state": {
                    "input": grid(sample[0, 0]),  # 12×12
                    "filters": [grid(f) for f in filters],  # 6×(3×3)
                    "activations": [grid(a) for a in acts],  # 6×(10×10)
                    "label": sample_label,
                    "pred": pred,
                },
                "metrics": {"loss": round(loss, 4), "accuracy": round(acc, 4)},
            }
        )
        if e == epochs:
            break
        # 训练一个 epoch（全批量）
        opt.zero_grad()
        lossf(model(Xtr), ytr).backward()
        opt.step()

    meta = {
        "id": "cnn-shapes",
        "title": "CNN · 卷积核可视化",
        "family": "cnn",
        "algorithm": "CNN",
        "description": "合成数据（横线 vs 竖线）。小 CNN 的第一层卷积核随训练逐渐变成边缘检测器。",
        "insight": "卷积核从随机噪声出发，训练后变成“横边/竖边检测器”；竖线样本经过竖边核会被强烈激活。",
        "tutorial": {
            "problem": "图像分类：让网络自己学会“看”出横线和竖线的区别，而不用人工设计特征。",
            "intuition": "CNN 用一组小卷积核（3×3）在整张图上滑动做模式匹配。训练会让这些核自动变成边缘检测器——竖边核遇到竖线就强烈响应。这就是 CNN“自动学特征”的威力。",
            "watch": [
                "左侧是输入样本（一条竖线）",
                "中间 6 个卷积核：从随机噪点，逐 epoch 变成有方向的边缘模板",
                "右侧激活图：竖线被对应方向的核点亮（越亮=响应越强）",
                "底部准确率随训练上升到接近 100%",
            ],
            "concepts": [
                {"term": "卷积核 / 滤波器", "explain": "3×3 的小权重窗口，在整张图滑动做局部模式匹配"},
                {"term": "激活图 feature map", "explain": "一个卷积核扫过整图后的响应图，亮处=匹配上了"},
                {"term": "权值共享", "explain": "同一个核在图上各处复用，参数远少于全连接"},
            ],
            "tryThis": "拖时间轴看卷积核从噪声变成边缘模板，对照右侧激活图怎么逐渐点亮竖线。",
        },
        "hyperparams": {"filters": N_FILTERS, "kernel": "3×3", "lr": 0.01, "epochs": epochs},
    }
    write_trajectory("cnn-shapes", meta, frames)
    print(f"CNN 训练完成，最终准确率 {frames[-1]['metrics']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
