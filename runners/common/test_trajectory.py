"""trajectory.py 的轻量契约校验。可直接运行：python runners/common/test_trajectory.py"""

import json
import tempfile
from pathlib import Path

from trajectory import Meta, Trajectory


def build_sample() -> Trajectory:
    meta = Meta(
        id="sample",
        title="Sample",
        algorithm="PPO",
        category="rl",
        source="python",
        abstract="a",
        description="d",
        insight="i",
        hyperparams={"lr": 2.5e-4},
    )
    traj = Trajectory(meta)
    for e in range(1, 4):
        traj.add_frame(iter=e, metrics={"reward": float(e * 10), "length": float(e)})
    return traj


def test_structure_matches_contract() -> None:
    d = build_sample().to_dict()
    assert set(d.keys()) == {"meta", "frames"}
    meta_keys = {
        "id", "title", "algorithm", "category", "source",
        "abstract", "description", "hyperparams", "insight",
    }
    assert set(d["meta"].keys()) == meta_keys
    assert d["meta"]["source"] == "python"
    assert len(d["frames"]) == 3
    f0 = d["frames"][0]
    assert set(f0.keys()) == {"iter", "state", "metrics"}
    assert set(f0["state"].keys()) == {"family", "data"}
    assert f0["state"]["family"] == "curves"
    assert f0["state"]["data"] == {}
    assert "reward" in f0["metrics"]


def test_iter_increasing_and_reward_present() -> None:
    frames = build_sample().to_dict()["frames"]
    iters = [f["iter"] for f in frames]
    assert iters == sorted(iters)
    assert all("reward" in f["metrics"] for f in frames)


def test_write_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "nested" / "sample.json"
        written = build_sample().write(out)
        assert written.exists()
        loaded = json.loads(written.read_text(encoding="utf-8"))
        assert loaded["meta"]["id"] == "sample"
        assert len(loaded["frames"]) == 3


if __name__ == "__main__":
    test_structure_matches_contract()
    test_iter_increasing_and_reward_present()
    test_write_roundtrip()
    print("all trajectory contract tests passed")
