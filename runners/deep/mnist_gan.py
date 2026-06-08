"""ML-Lab M4：MNIST 极小 GAN 生成，逐 epoch 导出（复用 image-grid 家族）。

MLP GAN：生成器 z(64)->256->784，判别器 784->256->1。每个 epoch 用一组固定
噪声生成 16 张图，观察生成分布从噪声逐渐逼近真实数字。

运行：poetry run python runners/deep/mnist_gan.py
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

OUTPUT = frames_dir() / "mnist-gan.json"
DATA_DIR = REPO_ROOT / "data" / "kaggle" / "mnist"

SEED = 42
N_TRAIN = 8000
Z_DIM = 64
EPOCHS = 30
BATCH = 128
LR = 2e-4
N_SHOW = 9


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 784), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_x(path: Path, n: int) -> torch.Tensor:
    df = pd.read_csv(path, nrows=n)
    return torch.tensor(df.drop(columns=["label"]).to_numpy(np.float32) / 255.0)


def img_list(flat: torch.Tensor) -> list[list[float]]:
    return [[round(float(v), 3) for v in row] for row in flat]


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("加载 MNIST ...")
    x_train = load_x(DATA_DIR / "mnist_train.csv", N_TRAIN)

    gen = Generator()
    disc = Discriminator()
    opt_g = torch.optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(N_SHOW, Z_DIM)

    def sample() -> list[list[float]]:
        gen.eval()
        with torch.no_grad():
            return img_list(gen(fixed_z))

    meta = Meta(
        id="mnist-gan",
        title="MNIST · GAN 生成",
        algorithm="GAN",
        category="deep",
        source="python",
        abstract="生成对抗网络：生成器与判别器博弈。固定一组噪声向量，观察生成的 16 张图随训练从纯噪声逐渐逼近真实手写数字——生成分布在向数据分布收敛。",
        description=f"MLP GAN（z={Z_DIM}），在 MNIST 子集训练 {EPOCHS} 轮，每 epoch 用固定噪声生成 {N_SHOW} 张图。",
        insight="GAN 把生成问题变成两个网络的对抗：判别器学着区分真假，生成器学着骗过判别器。纳什均衡处生成分布逼近真实分布。固定噪声让我们看到同一批『种子』如何逐帧长成数字。",
        hyperparams={"lr": LR, "batch_size": BATCH, "z_dim": Z_DIM, "n_train": N_TRAIN},
    )
    traj = Trajectory(meta)

    def add(epoch: int, d_loss: float, g_loss: float) -> None:
        traj.add_frame(
            iter=epoch,
            metrics={"d_loss": round(d_loss, 4), "g_loss": round(g_loss, 4)},
            family="image-grid",
            data={"groups": [
                {"title": f"固定噪声生成的 {N_SHOW} 张图", "w": 28, "h": 28, "colormap": "sequential", "images": sample()},
            ]},
        )

    add(0, 0.0, 0.0)
    print("开始训练 GAN ...")
    n = len(x_train)
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(n)
        dl_sum = gl_sum = 0.0
        steps = 0
        for i in range(0, n - BATCH, BATCH):
            real = x_train[perm[i : i + BATCH]]
            bs = len(real)
            # 训练判别器
            z = torch.randn(bs, Z_DIM)
            fake = gen(z).detach()
            d_real = disc(real)
            d_fake = disc(fake)
            d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            # 训练生成器
            z = torch.randn(bs, Z_DIM)
            gen_imgs = gen(z)
            d_out = disc(gen_imgs)
            g_loss = bce(d_out, torch.ones_like(d_out))
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
            dl_sum += d_loss.item(); gl_sum += g_loss.item(); steps += 1
        add(epoch, dl_sum / steps, gl_sum / steps)
        print(f"epoch {epoch}: d_loss={dl_sum/steps:.4f} g_loss={gl_sum/steps:.4f}")

    traj.write(OUTPUT)
    print(f"已写出 {len(traj.frames)} 帧 -> {OUTPUT}")


if __name__ == "__main__":
    main()
