import { Trajectory } from "./types";

/**
 * 异步加载 Python 预计算的帧轨迹（public/frames/<id>.json）。
 * 用于 source === "python" 的实验，前端纯回放。
 */
export async function loadTrajectory(id: string): Promise<Trajectory> {
  const res = await fetch(`/frames/${id}.json`);
  if (!res.ok) {
    throw new Error(`加载帧轨迹失败 (${id}): ${res.status}`);
  }
  return (await res.json()) as Trajectory;
}
