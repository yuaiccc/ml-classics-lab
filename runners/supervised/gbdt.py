"""ML-Lab：梯度提升树(GBDT)决策边界随提升轮数（scatter-boundary）。

用 staged_decision_function 逐轮取出边界，展示残差逐步拟合。
运行：poetry run python runners/supervised/gbdt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners" / "common"))
from trajectory import Meta, Trajectory  # noqa: E402
from capture_logger import frames_dir  # noqa: E402
from grid import make_grid  # noqa: E402

OUTPUT = frames_dir() / "gbdt.json"


def main() -> None:
    X, y = make_moons(n_samples=120, noise=0.28, random_state=7)
    xr = (float(X[:, 0].min()) - 0.5, float(X[:, 0].max()) + 0.5)
    yr = (float(X[:, 1].min()) - 0.5, float(X[:, 1].max()) + 0.5)
    n_estimators = 60
    gx = gy = 50
    grid = make_grid(xr, yr, gx, gy)

    clf = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=2, learning_rate=0.3, random_state=42)
    clf.fit(X, y)

    points = [{"x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "label": int(l)} for p, l in zip(X, y)]
    # 逐轮决策分数
    grid_stages = list(clf.staged_decision_function(grid))
    train_stages = list(clf.staged_decision_function(X))
    show_rounds = [0, 1, 2, 4, 7, 11, 17, 25, 39, 59]

    meta = Meta(
        id="gbdt",
        title="梯度提升树 (GBDT)",
        algorithm="GBDT",
        category="supervised",
        source="python",
        abstract="梯度提升按顺序训练一系列弱树，每棵新树拟合当前模型的残差（负梯度）。逐帧增加提升轮数，观察决策边界从粗糙逐步精修、贴合数据。",
        description="在 make_moons 上训练 GBDT，用 staged_decision_function 逐轮导出决策边界，展示残差逐步拟合。",
        insight="GBDT 是加法模型：F_m = F_{m-1} + η·h_m，其中 h_m 拟合负梯度（残差）。学习率 η 控制每步幅度。与随机森林(并行平均)不同，GBDT 是串行纠错，通常更强但更易过拟合。",
        hyperparams={"n_estimators": n_estimators, "max_depth": 2, "learning_rate": 0.3, "samples": len(X)},
    )
    traj = Trajectory(meta)
    for r in show_rounds:
        scores = np.asarray(grid_stages[r]).ravel().reshape(gx, gy)
        boundary = [[round(float(v), 4) for v in row] for row in scores]
        train_pred = (np.asarray(train_stages[r]).ravel() > 0).astype(int)
        acc = float((train_pred == y).mean())
        traj.add_frame(
            iter=r + 1,
            metrics={"rounds": r + 1, "accuracy": round(acc, 4)},
            family="scatter-boundary",
            data={"points": points, "boundary": boundary, "gridX": gx, "gridY": gy,
                  "xRange": [xr[0], xr[1]], "yRange": [yr[0], yr[1]]},
        )
        print(f"rounds={r+1}: acc={acc:.4f}")
    traj.write(OUTPUT)
    print(f"已写出 {len(traj.frames)} 帧 -> {OUTPUT}")


if __name__ == "__main__":
    main()
