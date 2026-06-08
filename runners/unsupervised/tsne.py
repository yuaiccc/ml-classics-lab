"""ML-Lab：t-SNE 降维逐帧展开（curves 家族）。

对 digits 数据(64维)做 t-SNE，沿优化路径逐帧导出 2D 投影，观察高维点从
随机初始逐渐展开成按数字聚集的簇。同一 seed + PCA 初始化保证各帧在同一条
优化轨迹上（截断到不同 max_iter）。

运行：poetry run python runners/unsupervised/tsne.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners" / "common"))
from trajectory import Meta, Trajectory  # noqa: E402
from capture_logger import frames_dir  # noqa: E402
from grid import embedding_state  # noqa: E402

OUTPUT = frames_dir() / "tsne.json"


def main() -> None:
    digits = load_digits()
    rng = np.random.RandomState(42)
    idx = rng.choice(len(digits.data), 500, replace=False)
    X = digits.data[idx]
    labels = digits.target[idx]

    iters = [300, 400, 550, 750, 1000, 1300]

    meta = Meta(
        id="tsne",
        title="t-SNE 降维展开",
        algorithm="t-SNE",
        category="unsupervised",
        source="python",
        abstract="t-SNE 把高维数据投影到 2D 用于可视化，保持局部邻域结构。对 8×8 手写数字(64维)逐帧展开，观察同一数字的点如何从混沌逐渐聚成清晰的簇。",
        description="对 digits 数据沿同一优化轨迹截取不同迭代步的 2D 投影（curves 家族 embedding），展示高维点逐帧展开。",
        insight="t-SNE 把高维相似度建模为概率，最小化高/低维分布的 KL 散度。它强调局部结构（邻居保持），擅长可视化簇，但全局距离不可信、且非确定性、对超参敏感。",
        hyperparams={"perplexity": 30, "init": "pca", "n_samples": len(X), "dim_in": 64},
    )
    traj = Trajectory(meta)
    for i, mi in enumerate(iters):
        ts = TSNE(n_components=2, perplexity=30, init="pca", random_state=42, max_iter=mi)
        emb = ts.fit_transform(X)
        traj.add_frame(iter=mi, metrics={"max_iter": mi, "kl_divergence": round(float(ts.kl_divergence_), 4)},
                       family="curves", data=embedding_state(emb, labels))
        print(f"max_iter={mi}: kl={ts.kl_divergence_:.4f}")
    traj.write(OUTPUT)
    print(f"已写出 {len(traj.frames)} 帧 -> {OUTPUT}")


if __name__ == "__main__":
    main()
