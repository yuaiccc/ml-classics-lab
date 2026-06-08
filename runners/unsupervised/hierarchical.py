"""ML-Lab：层次聚类（自底向上合并）动画（clusters 家族）。

用 Ward 连接做凝聚层次聚类，从「每点自成一簇」逐帧合并到少数几簇，
观察点的归属随合并逐渐汇聚。
运行：poetry run python runners/unsupervised/hierarchical.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.datasets import make_blobs

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners" / "common"))
from trajectory import Meta, Trajectory  # noqa: E402
from capture_logger import frames_dir  # noqa: E402
from grid import clusters_state  # noqa: E402

OUTPUT = frames_dir() / "hierarchical.json"


def main() -> None:
    X, _ = make_blobs(n_samples=80, centers=4, cluster_std=1.0, random_state=11)
    Z = linkage(X, method="ward")

    # 从多簇逐帧合并到 1 簇
    k_values = [12, 9, 7, 5, 4, 3, 2, 1]

    meta = Meta(
        id="hierarchical",
        title="层次聚类 (凝聚/Ward)",
        algorithm="Hierarchical Clustering",
        category="unsupervised",
        source="python",
        abstract="凝聚层次聚类从『每个点自成一簇』开始，每步合并最近的两簇，直到全部合并。逐帧降低簇数（相当于在树状图上从底往上切），观察点的归属如何逐级汇聚。",
        description="用 Ward 连接对四簇数据做层次聚类，逐帧从 12 簇合并到 1 簇，展示自底向上的归并过程。",
        insight="层次聚类不需预先指定簇数，产出一棵完整的合并树（树状图），可在任意高度切割得到不同粒度的聚类。Ward 连接每步选择使簇内方差增量最小的合并。缺点是 O(n²) 以上的复杂度，不适合大数据。",
        hyperparams={"linkage": "ward", "k_values": ",".join(map(str, k_values)), "samples": len(X)},
    )
    traj = Trajectory(meta)
    for i, k in enumerate(k_values):
        assign = fcluster(Z, t=k, criterion="maxclust") - 1
        # 质心 = 各簇均值
        cents = np.array([X[assign == c].mean(axis=0) for c in range(assign.max() + 1)])
        traj.add_frame(iter=i, metrics={"clusters": int(assign.max() + 1)},
                       family="clusters", data=clusters_state(X, assign, cents, int(assign.max() + 1)))
        print(f"k={k}: clusters={assign.max() + 1}")
    traj.write(OUTPUT)
    print(f"已写出 {len(traj.frames)} 帧 -> {OUTPUT}")


if __name__ == "__main__":
    main()
