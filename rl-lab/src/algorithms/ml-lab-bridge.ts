// 把 ml-lab（Tianshou 演进版）的帧格式 {state:{family,data}} 解包成 A 的 {state:data}。
// ml-lab 把 family 标在每帧 state 里；A 按 DEMO 直接绑定 Viz，所以这里只需解包 data。
import { Trajectory } from "@/player/types";

interface RawFrame {
  iter: number;
  state: { family: string; data: unknown };
  metrics: Record<string, number>;
}
interface RawTraj {
  meta: Record<string, unknown>;
  frames: RawFrame[];
}

export function bridgeMlLab(raw: unknown): Trajectory {
  const t = raw as RawTraj;
  return {
    meta: t.meta as never,
    frames: t.frames.map((f) => ({
      iter: f.iter,
      state: f.state.data,
      metrics: f.metrics,
    })),
  } as unknown as Trajectory;
}
