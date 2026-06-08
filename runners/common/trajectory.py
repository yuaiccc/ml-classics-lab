"""统一「帧轨迹」契约的 Python 侧构造与写出。

输出 JSON 逐字段匹配前端 `ml-lab/src/player/types.ts` 的 Trajectory：

    Trajectory = { meta: TrajectoryMeta, frames: Frame[] }
    Frame      = { iter, state, metrics }
    FrameState = { family, data }

RL 走 "curves" 家族，过程动画的数据全在每帧 metrics 里，state.data 留空 {}。
仅依赖标准库。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Meta:
    """对齐 types.ts 的 TrajectoryMeta（键为 camelCase）。"""

    id: str
    title: str
    algorithm: str
    category: str  # "supervised" | "unsupervised" | "deep" | "rl"
    source: str  # "browser" | "python"
    abstract: str
    description: str
    insight: str
    hyperparams: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "algorithm": self.algorithm,
            "category": self.category,
            "source": self.source,
            "abstract": self.abstract,
            "description": self.description,
            "hyperparams": self.hyperparams,
            "insight": self.insight,
        }


@dataclass
class Frame:
    iter: int
    metrics: dict[str, float]
    # FrameState：默认 curves 家族、空 data
    family: str = "curves"
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iter": self.iter,
            "state": {"family": self.family, "data": self.data},
            "metrics": self.metrics,
        }


class Trajectory:
    def __init__(self, meta: Meta) -> None:
        self.meta = meta
        self.frames: list[Frame] = []

    def add_frame(
        self,
        iter: int,
        metrics: dict[str, float],
        *,
        family: str = "curves",
        data: dict[str, Any] | None = None,
    ) -> None:
        self.frames.append(
            Frame(iter=iter, metrics=metrics, family=family, data=data or {})
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "frames": [f.to_dict() for f in self.frames],
        }

    def write(self, path: str | Path) -> Path:
        """序列化为紧凑 JSON（数据文件无需缩进，省体积），自动建父目录。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        return p.resolve()
