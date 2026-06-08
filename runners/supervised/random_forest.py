"""ML-Lab：随机森林决策边界随树数量增加（scatter-boundary）。

运行：poetry run python runners/supervised/random_forest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_circles
from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners" / "common"))
from trajectory import Meta, Trajectory  # noqa: E402
from capture_logger import frames_dir  # noqa: E402
from grid import scatter_boundary_state  # noqa: E402

OUTPUT = frames_dir() / "random-forest.json"


def main() -> None:
    X, y = make_circles(n_samples=140, noise=0.18, factor=0.45, random_state=42)
    xr = (float(X[:, 0].min()) - 0.4, float(X[:, 0].max()) + 0.4)
    yr = (float(X[:, 1].min()) - 0.4, float(X[:, 1].max()) + 0.4)
    n_trees = [1, 2, 4, 8, 16, 32, 64]

    meta = Meta(
        id="random-forest",
        title="随机森林决策边界",
        algorithm="Random Forest",
        category="supervised",
        source="python",
        abstract="随机森林是多棵决策树的投票集成。逐帧增加树的数量，观察决策边界从单棵树的锯齿状逐渐被『平均』得平滑稳健——这就是 Bagging 降低方差的过程。",
        description="在 make_circles 数据上逐帧增加随机森林的树数量，展示集成如何平滑单棵树的高方差边界。",
        insight="单棵决策树方差高、边界锯齿。随机森林通过 ① 自助采样(bootstrap) ② 随机特征子集 训练多棵去相关的树再投票，集成平均显著降低方差，边界更平滑、泛化更好。",
        hyperparams={"max_depth": 6, "n_trees": ",".join(map(str, n_trees)), "samples": len(X)},
    )
    traj = Trajectory(meta)
    for i, n in enumerate(n_trees):
        clf = RandomForestClassifier(n_estimators=n, max_depth=6, random_state=42).fit(X, y)
        acc = float(clf.score(X, y))
        state = scatter_boundary_state(X, y, lambda g: clf.predict(g), xr, yr)
        traj.add_frame(iter=n, metrics={"n_trees": n, "accuracy": round(acc, 4)},
                       family="scatter-boundary", data=state)
        print(f"n_trees={n}: acc={acc:.4f}")
    traj.write(OUTPUT)
    print(f"已写出 {len(traj.frames)} 帧 -> {OUTPUT}")


if __name__ == "__main__":
    main()
