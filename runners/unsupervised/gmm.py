"""ML-Lab：高斯混合模型(GMM) EM 迭代（clusters 家族）。

逐 EM 迭代导出软分配的硬化结果与高斯均值，观察均值移动、归属变化。
运行：poetry run python runners/unsupervised/gmm.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners" / "common"))
from trajectory import Meta, Trajectory  # noqa: E402
from capture_logger import frames_dir  # noqa: E402
from grid import clusters_state  # noqa: E402

OUTPUT = frames_dir() / "gmm.json"


def main() -> None:
    X, _ = make_blobs(n_samples=120, centers=3, cluster_std=1.3, random_state=42)
    k = 3
    gm = GaussianMixture(n_components=k, covariance_type="full", warm_start=True,
                         max_iter=1, init_params="random", random_state=0)

    meta = Meta(
        id="gmm",
        title="高斯混合模型 (GMM / EM)",
        algorithm="GMM (EM)",
        category="unsupervised",
        source="python",
        abstract="GMM 假设数据由若干高斯分布混合生成，用 EM 算法交替估计『每点属于各高斯的概率(E步)』与『各高斯的参数(M步)』。逐帧观察高斯均值移动、点的软归属逐渐稳定。",
        description="在三簇高斯数据上逐 EM 迭代导出均值与硬化归属，展示 EM 的 E/M 两步收敛。",
        insight="EM 是 K-Means 的概率推广：E 步算责任度(soft assignment)，M 步用加权数据更新均值/协方差。相比 K-Means 的硬分配，GMM 能建模椭圆形簇与重叠区域，但同样可能收敛到局部最优。",
        hyperparams={"n_components": k, "covariance": "full", "init": "random", "samples": len(X)},
    )
    traj = Trajectory(meta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for it in range(14):
            gm.fit(X)
            assign = gm.predict(X)
            ll = float(gm.lower_bound_)
            traj.add_frame(iter=it, metrics={"log_likelihood": round(ll, 4)},
                           family="clusters", data=clusters_state(X, assign, gm.means_, k))
            print(f"iter {it}: log_likelihood={ll:.4f}")
    traj.write(OUTPUT)
    print(f"已写出 {len(traj.frames)} 帧 -> {OUTPUT}")


if __name__ == "__main__":
    main()
