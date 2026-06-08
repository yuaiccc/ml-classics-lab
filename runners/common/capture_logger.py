"""共享的训练过程捕获工具：把 Tianshou 每个 epoch 的测试评估记成帧。

供 runners/rl/*.py 复用，避免每个脚本重复实现 logger。
"""

from __future__ import annotations

from pathlib import Path

from tianshou.utils import LazyLogger

from trajectory import Meta, Trajectory


class EpochCaptureLogger(LazyLogger):
    """每次 log_test_data（一个 epoch 的测试评估）记录一帧训练指标。"""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, float]] = []

    def log_test_data(self, log_data: dict, step: int) -> None:
        returns_stat = log_data.get("returns_stat") or {}
        lens_stat = log_data.get("lens_stat") or {}
        self.records.append(
            {
                "env_step": float(step),
                "reward": float(returns_stat.get("mean", 0.0)),
                "reward_std": float(returns_stat.get("std", 0.0)),
                "length": float(lens_stat.get("mean", 0.0)),
            }
        )


def export_trajectory(meta: Meta, logger: EpochCaptureLogger, output: str | Path) -> Path:
    """把捕获到的 per-epoch 记录构造成 Trajectory 并写出。"""
    traj = Trajectory(meta)
    for i, rec in enumerate(logger.records, start=1):
        metrics: dict[str, float] = {
            "reward": rec["reward"],
            "reward_std": rec["reward_std"],
            "length": rec["length"],
            "env_step": rec["env_step"],
        }
        traj.add_frame(iter=i, metrics=metrics)
    written = traj.write(output)
    print(f"已写出 {len(traj.frames)} 帧 -> {written}")
    return written


def frames_dir() -> Path:
    """ml-lab/public/frames 的绝对路径（相对仓库根）。"""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "ml-lab" / "public" / "frames"
