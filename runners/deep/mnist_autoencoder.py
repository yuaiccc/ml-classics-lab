"""ML-Lab M4：MNIST 自编码器重建，逐 epoch 导出（复用 image-grid 家族）。

小 MLP 自编码器 784->128->32->128->784。每个 epoch 对固定探针数字导出
「输入」与「重建」两组图，观察重建从噪声逐渐还原。

运行：poetry run python runners/deep/mnist_autoencoder.py
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

OUTPUT = frames_dir() / "mnist-autoencoder.json"
DATA_DIR = REPO_ROOT / "data" / "kaggle" / "mnist"

SEED = 42
N_TRAIN = 8000
N_VAL = 2000
EPOCHS = 8
BATCH = 256
LR = 1e-3
LATENT = 32


class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, LATENT))
        self.dec = nn.Sequential(nn.Linear(LATENT, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


def load_subset(path: Path, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(path, nrows=n)
    y = torch.tensor(df["label"].to_numpy(np.int64), dtype=torch.long)
    x = torch.tensor(df.drop(columns=["label"]).to_numpy(np.float32) / 255.0)
    return x, y


def img_list(flat: torch.Tensor) -> list[list[float]]:
    return [[round(float(v), 3) for v in row] for row in flat]


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("加载 MNIST ...")
    x_train, _ = load_subset(DATA_DIR / "mnist_train.csv", N_TRAIN)
    x_val, y_val = load_subset(DATA_DIR / "mnist_test.csv", N_VAL)

    # 固定探针：数字 0-5 各一张
    probe_idx = [int(torch.nonzero(y_val == d).flatten()[0].item()) for d in range(6)]
    probe = x_val[probe_idx]  # (8,784)
    inputs = img_list(probe)

    model = AutoEncoder()
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    def recon() -> list[list[float]]:
        model.eval()
        with torch.no_grad():
            return img_list(model(probe))

    meta = Meta(
        id="mnist-autoencoder",
        title="MNIST · 自编码器重建",
        algorithm="AutoEncoder",
        category="deep",
        source="python",
        abstract="自编码器把 784 维数字压到 32 维潜向量再重建。观察重建图随训练从模糊噪声逐渐还原出清晰数字——无监督地学到了数据的紧凑表示。",
        description=f"MLP 自编码器 784→128→{LATENT}→128→784，在 MNIST 子集训练 {EPOCHS} 轮，逐 epoch 导出固定探针的输入与重建。",
        insight="自编码器用『重建自身』作为无监督目标，瓶颈层强迫网络丢弃噪声、保留最有信息量的结构。潜空间维度越小，重建越抽象。这是表示学习与降维的桥梁。",
        hyperparams={"lr": LR, "batch_size": BATCH, "latent_dim": LATENT, "n_train": N_TRAIN},
    )
    traj = Trajectory(meta)

    def add(epoch: int, loss: float) -> None:
        traj.add_frame(
            iter=epoch,
            metrics={"recon_loss": round(loss, 4)},
            family="image-grid",
            data={"groups": [
                {"title": "输入 (探针数字 0-5)", "w": 28, "h": 28, "colormap": "sequential", "images": inputs},
                {"title": "重建", "w": 28, "h": 28, "colormap": "sequential", "images": recon()},
            ]},
        )

    add(0, float(loss_fn(model(probe), probe).item()))
    print("开始训练 ...")
    n = len(x_train)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            xb = x_train[perm[i : i + BATCH]]
            out = model(xb)
            loss = loss_fn(out, xb)
            optim.zero_grad(); loss.backward(); optim.step()
            total += loss.item() * len(xb)
        tl = total / n
        add(epoch, tl)
        print(f"epoch {epoch}: recon_loss={tl:.4f}")

    traj.write(OUTPUT)
    print(f"已写出 {len(traj.frames)} 帧 -> {OUTPUT}")


if __name__ == "__main__":
    main()
