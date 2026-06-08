"""ML-Lab M4 切片 2：MNIST CNN 卷积核 + 激活图，逐 epoch 导出。

复用 solve_mnist_cnn.py 的 SmallCNN 架构（conv1 1->12 5x5 -> conv2 -> pool -> 分类器），
在 MNIST CSV 子集上训练。每个 epoch 抓两组可视化：
  - 卷积核：conv1 的 12 个 5x5 核（diverging colormap，显正负权重）
  - 激活图：固定探针数字经 conv1+ReLU+pool 后的 12 张 14x14 特征图（sequential）

帧契约：state.family = "image-grid"
        state.data    = { groups: [{ title, w, h, colormap, images: [[float]] }] }

运行：poetry run python runners/deep/mnist_cnn_kernels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners" / "common"))
from trajectory import Meta, Trajectory  # noqa: E402
from capture_logger import frames_dir  # noqa: E402

OUTPUT = frames_dir() / "mnist-cnn-kernels.json"
DATA_DIR = REPO_ROOT / "data" / "kaggle" / "mnist"

# ==================== 超参数 ====================
SEED = 42
N_TRAIN = 8000
N_VAL = 2000
EPOCHS = 6
BATCH = 256
LR = 1e-3
PROBE_DIGIT = 7


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Linear(24 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 10)
        )

    def conv1_activation(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.relu(self.conv1(x)))  # (B,12,14,14)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return self.classifier(x.flatten(1))


def load_subset(path: Path, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(path, nrows=n)
    y = torch.tensor(df["label"].to_numpy(np.int64), dtype=torch.long)
    x = torch.tensor(df.drop(columns=["label"]).to_numpy(np.float32) / 255.0)
    return x.reshape(-1, 1, 28, 28), y


def kernels_group(model: SmallCNN) -> dict:
    """conv1 的 12 个 5x5 核，按全局 maxabs 对称归一化到 [0,1]（diverging）。"""
    w = model.conv1.weight[:, 0].detach()  # (12,5,5)
    maxabs = float(w.abs().max().item()) or 1.0
    images = [[round(0.5 + 0.5 * float(v) / maxabs, 4) for v in k.flatten()] for k in w]
    return {"title": "conv1 卷积核 (12×5×5)", "w": 5, "h": 5, "colormap": "diverging", "images": images}


def activations_group(model: SmallCNN, probe: torch.Tensor) -> dict:
    """探针经 conv1+ReLU+pool 后的 12 张 14x14 特征图，每张按自身 max 归一化（sequential）。"""
    with torch.no_grad():
        act = model.conv1_activation(probe)[0].detach()  # (12,14,14)
    images = []
    for fmap in act:
        m = float(fmap.max().item()) or 1.0
        images.append([round(float(v) / m, 4) for v in fmap.flatten()])
    return {"title": "conv1 激活图 (探针数字, 12×14×14)", "w": 14, "h": 14, "colormap": "sequential", "images": images}


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("加载 MNIST ...")
    x_train, y_train = load_subset(DATA_DIR / "mnist_train.csv", N_TRAIN)
    x_val, y_val = load_subset(DATA_DIR / "mnist_test.csv", N_VAL)

    # 固定探针：验证集中第一张 PROBE_DIGIT
    probe_idx = int(torch.nonzero(y_val == PROBE_DIGIT).flatten()[0].item())
    probe = x_val[probe_idx : probe_idx + 1]
    probe_pixels = [round(float(v), 4) for v in probe[0, 0].flatten()]
    print(f"探针: 验证集 #{probe_idx} (数字 {PROBE_DIGIT})")

    model = SmallCNN()
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def val_acc() -> float:
        model.eval()
        correct = 0
        for i in range(0, len(x_val), 512):
            logits = model(x_val[i : i + 512])
            correct += int((logits.argmax(1) == y_val[i : i + 512]).sum().item())
        return correct / len(x_val)

    meta = Meta(
        id="mnist-cnn-kernels",
        title="MNIST · CNN 卷积核与激活",
        algorithm="Small CNN",
        category="deep",
        source="python",
        abstract="观察 CNN 第一层卷积核如何从随机噪点逐渐学成边缘/笔画检测器，以及固定探针数字经过这些核后的激活图如何随训练变清晰。",
        description=f"复用 LeNet 风格 SmallCNN（conv1 1→12 5×5）在 MNIST 子集训练 {EPOCHS} 轮。每 epoch 导出 conv1 卷积核与探针数字的激活特征图。",
        insight="卷积核是 CNN 学到的『特征模板』。训练初期权重随机（噪点），随着分类目标驱动，卷积核逐渐形成方向性边缘/笔画检测器；探针的激活图也从模糊变得轮廓分明。这就是卷积归纳偏置的可视化。",
        hyperparams={
            "lr": LR,
            "batch_size": BATCH,
            "conv_channels": "12,24",
            "kernel_size": 5,
            "n_train": N_TRAIN,
        },
    )
    traj = Trajectory(meta)

    # epoch 0：训练前的随机卷积核（展示起点是噪点）
    traj.add_frame(
        iter=0,
        metrics={"loss": 0.0, "val_accuracy": round(val_acc(), 4)},
        family="image-grid",
        data={
            "probe": {"w": 28, "h": 28, "pixels": probe_pixels, "label": PROBE_DIGIT},
            "groups": [kernels_group(model), activations_group(model, probe)],
        },
    )

    print("开始训练 ...")
    n = len(x_train)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            logits = model(x_train[idx])
            loss = loss_fn(logits, y_train[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n
        acc = val_acc()
        traj.add_frame(
            iter=epoch,
            metrics={"loss": round(train_loss, 4), "val_accuracy": round(acc, 4)},
            family="image-grid",
            data={
                "probe": {"w": 28, "h": 28, "pixels": probe_pixels, "label": PROBE_DIGIT},
                "groups": [kernels_group(model), activations_group(model, probe)],
            },
        )
        print(f"epoch {epoch}: loss={train_loss:.4f} val_acc={acc:.4f}")

    traj.write(OUTPUT)
    print(f"已写出 {len(traj.frames)} 帧 -> {OUTPUT}")


if __name__ == "__main__":
    main()
