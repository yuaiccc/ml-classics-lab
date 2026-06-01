"""M3 录制管线：把训练好的策略跑一条 episode，导出成前端统一帧契约 (Trajectory) 的 JSON。

帧契约见 rl-lab/src/player/types.ts：
    Trajectory = { meta, frames[] }
    frame      = { iter, state, metrics }
对 RL（env 家族）：state = { observation: number[], action }
"""

import json
import os

import numpy as np

import tianshou as ts
from tianshou.data import Collector, CollectStats, VectorReplayBuffer


def record_episode_frames(algorithm, task: str, max_steps: int = 1200) -> list:
    """用训练好的 algorithm 在 task 上跑一条 episode，返回 frames 列表。

    通过 Collector + ReplayBuffer 提取 (obs, act, rew)，对离散/连续动作通用。
    """
    import gymnasium as gym

    env = ts.env.DummyVectorEnv([lambda: gym.make(task)])
    buf = VectorReplayBuffer(max_steps + 16, 1)
    collector = Collector[CollectStats](algorithm, env, buf)
    collector.reset()
    collector.collect(n_episode=1)

    n = len(buf)
    data = buf[:n]
    obs = np.asarray(data.obs)
    act = np.asarray(data.act)
    rew = np.asarray(data.rew)

    frames = []
    cumulative = 0.0
    for i in range(n):
        cumulative += float(rew[i])
        action = (
            float(act[i])
            if act.ndim == 1
            else [float(x) for x in np.ravel(act[i])]
        )
        frames.append(
            {
                "iter": i,
                "state": {
                    "observation": [float(x) for x in np.ravel(obs[i])],
                    "action": action,
                },
                "metrics": {"reward": float(rew[i]), "return": cumulative},
            }
        )
    env.close()
    return frames


def write_trajectory(file_id: str, meta: dict, frames: list) -> str:
    """写出 Trajectory JSON 到 rl-lab/src/data/frames/<file_id>.json（供前端 import）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "rl-lab", "src", "data", "frames")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{file_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "frames": frames}, f, ensure_ascii=False)
    print(f"[frames] 写出 {len(frames)} 帧 -> {os.path.relpath(path, os.path.join(here, '..'))}")
    return path


def maybe_record(algorithm, task: str, meta: dict) -> None:
    """若设置了环境变量 RECORD_FRAMES=1，则录制并写出。默认不影响脚本行为。"""
    if not os.environ.get("RECORD_FRAMES"):
        return
    print("\n[frames] RECORD_FRAMES=1，录制一条 episode 用于前端回放 ...")
    frames = record_episode_frames(algorithm, task)
    write_trajectory(meta["id"], meta, frames)
