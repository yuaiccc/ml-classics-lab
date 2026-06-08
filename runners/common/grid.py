"""scatter-boundary / clusters 家族的 Python 侧构造辅助。"""

from __future__ import annotations

from typing import Callable

import numpy as np


def make_grid(x_range: tuple[float, float], y_range: tuple[float, float], gx: int, gy: int) -> np.ndarray:
    """返回形状 (gx*gy, 2) 的网格点，顺序与前端 boundary[i][j] 一致（i 沿 x、j 沿 y）。"""
    xs = x_range[0] + (np.arange(gx) / gx) * (x_range[1] - x_range[0])
    ys = y_range[0] + (np.arange(gy) / gy) * (y_range[1] - y_range[0])
    pts = np.array([[x, y] for x in xs for y in ys], dtype=np.float32)
    return pts


def scatter_boundary_state(
    X: np.ndarray,
    y: np.ndarray,
    predict_grid: Callable[[np.ndarray], np.ndarray],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    gx: int = 50,
    gy: int = 50,
) -> dict:
    """构造 scatter-boundary 家族的 state.data。

    predict_grid: 接收 (N,2) 网格点，返回长度 N 的预测值（类别或决策分数）。
    """
    grid = make_grid(x_range, y_range, gx, gy)
    preds = np.asarray(predict_grid(grid)).reshape(gx, gy)
    boundary = [[round(float(v), 4) for v in row] for row in preds]
    points = [{"x": round(float(px), 4), "y": round(float(py), 4), "label": int(ly)}
              for (px, py), ly in zip(X, y)]
    return {
        "points": points,
        "boundary": boundary,
        "gridX": gx,
        "gridY": gy,
        "xRange": [x_range[0], x_range[1]],
        "yRange": [y_range[0], y_range[1]],
    }


def clusters_state(X: np.ndarray, assignments: np.ndarray, centroids: np.ndarray, k: int) -> dict:
    return {
        "points": [{"x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "cluster": int(c)}
                   for p, c in zip(X, assignments)],
        "centroids": [{"x": round(float(c[0]), 4), "y": round(float(c[1]), 4)} for c in centroids],
        "k": k,
    }


def embedding_state(emb: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "embedding": [{"x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "label": int(l)}
                      for p, l in zip(emb, labels)],
    }
