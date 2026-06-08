"""ML-Lab：MNIST 变分自编码器(VAE)，逐 epoch 导出（复用 image-grid 家族）。

VAE 在自编码器基础上让潜空间服从正态分布，因而既能重建、也能从 N(0,1)
采样生成新数字。每个 epoch 导出三组：输入 / 重建 / 从潜空间生成。

运行：poetry run python runners/deep/mnist_vae.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners" / "common"))
from trajectory import Meta, Trajectory  # noqa: E402
from capture_logger import frames_dir  # noqa: E402

OUTPUT = frames_dir() / "mnist-vae.json"
DATA_DIR = REPO_ROOT / "data" / "kaggle" / "mnist"

SEED = 42
N_TRAIN = 8000
EPOCHS = 8
BATCH = 256
LR = 1e-3
LATENT = 16
N_GEN = 8


class VAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, LATENT)
        self.fc_lv = nn.Linear(256, LATENT)
        self.dec = nn.Sequential(nn.Linear(LATENT, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def encode(self, x: torch.Tensor):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_lv(h)

    def forward(self, x: torch.Tensor):
        mu, lv = self.encode(x)
        std = torch.exp(0.5 * lv)
        z = mu + std * torch.randn_like(std)
        return self.dec(z), mu, lv


def load_x(path: Path, n: int) -> tuple[torch.Tensor, torch.Tensor]:
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
    x_train, _ = load_x(DATA_DIR / "mnist_train.csv", N_TRAIN)
    x_test, y_test = load_x(DATA_DIR / "mnist_test.csv", 2000)
    probe_idx = [int(torch.nonzero(y_test == d).flatten()[0].item()) for d in range(6)]
    probe = x_test[probe_idx]
    inputs = img_list(probe)
    fixed_z = torch.randn(N_GEN, LATENT)

    model = VAE()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    def vae_loss(recon, x, mu, lv):
        bce = F.binary_cross_entropy(recon, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        return (bce + kld) / len(x)

    def recon_imgs():
        model.eval()
        with torch.no_grad():
            r, _, _ = model(probe)
        return img_list(r)

    def gen_imgs():
        model.eval()
        with torch.no_grad():
            return img_list(model.dec(fixed_z))

    meta = Meta(
        id="mnist-vae",
        title="MNIST · 变分自编码器 (VAE)",
        algorithm="VAE",
        category="deep",
        source="python",
        abstract="VAE 让潜空间服从正态分布，因而既能重建输入、也能从 N(0,1) 采样生成新数字。逐帧观察重建变清晰、随机潜向量解码出的图像逐渐成形为数字。",
        description=f"MLP VAE（潜维 {LATENT}）在 MNIST 子集训练 {EPOCHS} 轮，每 epoch 导出输入/重建/从潜空间生成三组图。",
        insight="VAE = 自编码器 + 正态潜空间约束(KL 项)。重参数化技巧让采样可微。相比普通自编码器，VAE 的潜空间连续且规整，因此能生成；相比 GAN，VAE 训练稳定但生成更模糊。",
        hyperparams={"lr": LR, "batch_size": BATCH, "latent_dim": LATENT, "n_train": N_TRAIN},
    )
    traj = Trajectory(meta)

    def add(epoch: int, loss: float) -> None:
        traj.add_frame(
            iter=epoch,
            metrics={"elbo_loss": round(loss, 3)},
            family="image-grid",
            data={"groups": [
                {"title": "输入 (探针数字 0-5)", "w": 28, "h": 28, "colormap": "sequential", "images": inputs},
                {"title": "重建", "w": 28, "h": 28, "colormap": "sequential", "images": recon_imgs()},
                {"title": f"从潜空间 N(0,1) 生成的 {N_GEN} 张", "w": 28, "h": 28, "colormap": "sequential", "images": gen_imgs()},
            ]},
        )

    add(0, 0.0)
    print("开始训练 VAE ...")
    n = len(x_train)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            xb = x_train[perm[i : i + BATCH]]
            recon, mu, lv = model(xb)
            loss = vae_loss(recon, xb, mu, lv)
            optim.zero_grad(); loss.backward(); optim.step()
            total += loss.item() * len(xb)
        tl = total / n
        add(epoch, tl)
        print(f"epoch {epoch}: elbo_loss={tl:.3f}")

    traj.write(OUTPUT)
    print(f"已写出 {len(traj.frames)} 帧 -> {OUTPUT}")


if __name__ == "__main__":
    main()
