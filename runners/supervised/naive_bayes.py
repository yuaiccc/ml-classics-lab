"""ML-Lab：高斯朴素贝叶斯决策边界随样本增加成形（scatter-boundary）。

运行：poetry run python runners/supervised/naive_bayes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners" / "common"))
from trajectory import Meta, Trajectory  # noqa: E402
from capture_logger import frames_dir  # noqa: E402
from grid import scatter_boundary_state  # noqa: E402

OUTPUT = frames_dir() / "naive-bayes.json"


def main() -> None:
    X, y = make_blobs(n_samples=180, centers=3, cluster_std=1.6, random_state=3)
    xr = (float(X[:, 0].min()) - 1, float(X[:, 0].max()) + 1)
    yr = (float(X[:, 1].min()) - 1, float(X[:, 1].max()) + 1)
    sizes = [6, 12, 24, 48, 96, 180]

    meta = Meta(
        id="naive-bayes",
        title="朴素贝叶斯决策边界",
        algorithm="Gaussian Naive Bayes",
        category="supervised",
        source="python",
        abstract="朴素贝叶斯假设特征条件独立，为每个类拟合一个高斯分布，用贝叶斯定理算后验。逐帧增加训练样本，观察决策边界如何随类条件分布的估计逐渐稳定。",
        description="在三类高斯数据上逐帧增加训练样本量，展示高斯朴素贝叶斯的决策边界从少样本的粗糙到充分样本的稳定。",
        insight="朴素贝叶斯的『朴素』在于假设特征间条件独立——常不成立但效果意外地好。它本质是为每类估计一个（轴对齐）高斯，决策边界是这些高斯后验相等的二次曲线。训练极快、天然支持在线学习。",
        hyperparams={"distribution": "gaussian", "classes": 3, "sample_sizes": ",".join(map(str, sizes))},
    )
    traj = Trajectory(meta)
    for i, n in enumerate(sizes):
        clf = GaussianNB().fit(X[:n], y[:n])
        acc = float(clf.score(X, y))
        state = scatter_boundary_state(X[:n], y[:n], lambda g: clf.predict(g), xr, yr)
        traj.add_frame(iter=n, metrics={"n_samples": n, "accuracy": round(acc, 4)},
                       family="scatter-boundary", data=state)
        print(f"n={n}: acc={acc:.4f}")
    traj.write(OUTPUT)
    print(f"已写出 {len(traj.frames)} 帧 -> {OUTPUT}")


if __name__ == "__main__":
    main()
