"""ML-Lab：RBF SVM 决策边界随正则 C 变化（scatter-boundary）。

运行：poetry run python runners/supervised/svm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners" / "common"))
from trajectory import Meta, Trajectory  # noqa: E402
from capture_logger import frames_dir  # noqa: E402
from grid import scatter_boundary_state  # noqa: E402

OUTPUT = frames_dir() / "svm.json"


def main() -> None:
    X, y = make_moons(n_samples=120, noise=0.25, random_state=42)
    xr = (float(X[:, 0].min()) - 0.5, float(X[:, 0].max()) + 0.5)
    yr = (float(X[:, 1].min()) - 0.5, float(X[:, 1].max()) + 0.5)
    c_values = [0.1, 0.3, 1.0, 3.0, 10.0, 50.0]

    meta = Meta(
        id="svm",
        title="SVM (RBF) 决策边界",
        algorithm="SVM",
        category="supervised",
        source="python",
        abstract="支持向量机用核技巧在高维空间找最大间隔分界面。逐帧增大正则系数 C，观察 RBF-SVM 的决策边界从平滑（强正则、欠拟合）逐渐变复杂（弱正则、过拟合）。",
        description="在 make_moons 数据上训练 RBF SVM，逐帧增大 C，展示间隔与边界复杂度的权衡。",
        insight="C 控制对误分类的惩罚：C 小 → 更宽的间隔、更平滑边界但容忍更多错分；C 大 → 更贴合训练点但易过拟合。RBF 核把数据映到高维使非线性可分。",
        hyperparams={"kernel": "rbf", "gamma": "scale", "C_values": ",".join(map(str, c_values)), "samples": len(X)},
    )
    traj = Trajectory(meta)
    for i, C in enumerate(c_values):
        clf = SVC(C=C, kernel="rbf", gamma="scale").fit(X, y)
        acc = float(clf.score(X, y))
        state = scatter_boundary_state(X, y, lambda g: clf.predict(g), xr, yr)
        traj.add_frame(iter=i, metrics={"C": C, "accuracy": round(acc, 4)},
                       family="scatter-boundary", data=state)
        print(f"C={C}: acc={acc:.4f}")
    traj.write(OUTPUT)
    print(f"已写出 {len(traj.frames)} 帧 -> {OUTPUT}")


if __name__ == "__main__":
    main()
