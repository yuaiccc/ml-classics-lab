// env 家族：按 meta.envId 渲染对应 RL 环境的当前观测帧。
// 数据来自 Python 录制管线导出的真实 rollout（solvers/frames_io.py）。
import { EnvState, TrajectoryMeta } from "@/player/types";

const W = 560;
const H = 360;

function CartPole({ obs }: { obs: number[] }) {
  const [x, , theta] = obs; // [位置, 速度, 角度, 角速度]
  const xThreshold = 2.4;
  const cx = W / 2 + (x / xThreshold) * (W / 2 - 60);
  const trackY = H * 0.7;
  const cartW = 70;
  const cartH = 32;
  const poleLen = 130;
  const px = cx + poleLen * Math.sin(theta);
  const py = trackY - poleLen * Math.cos(theta);
  return (
    <>
      <line x1={20} y1={trackY} x2={W - 20} y2={trackY} stroke="#334155" strokeWidth={2} />
      <rect x={cx - cartW / 2} y={trackY - cartH / 2} width={cartW} height={cartH} rx={5} fill="#00e5ff" opacity={0.85} />
      <line x1={cx} y1={trackY} x2={px} y2={py} stroke="#00ff88" strokeWidth={7} strokeLinecap="round" />
      <circle cx={px} cy={py} r={7} fill="#ffab40" />
      <circle cx={cx} cy={trackY} r={4} fill="#0a0e17" stroke="#94a3b8" />
    </>
  );
}

function MountainCar({ obs }: { obs: number[] }) {
  const [pos] = obs; // 位置 ∈ [-1.2, 0.6]
  const x0 = -1.2;
  const x1 = 0.6;
  const hill = (p: number) => Math.sin(3 * p);
  const toPx = (p: number) => 40 + ((p - x0) / (x1 - x0)) * (W - 80);
  const toPy = (p: number) => H * 0.85 - hill(p) * 80;
  const path = Array.from({ length: 80 }, (_, i) => {
    const p = x0 + (i / 79) * (x1 - x0);
    return `${i === 0 ? "M" : "L"}${toPx(p)},${toPy(p)}`;
  }).join(" ");
  const goalX = toPx(0.5);
  return (
    <>
      <path d={path} fill="none" stroke="#334155" strokeWidth={2.5} />
      <line x1={goalX} y1={toPy(0.5)} x2={goalX} y2={toPy(0.5) - 40} stroke="#00ff88" strokeWidth={2} />
      <circle cx={toPx(pos)} cy={toPy(pos) - 8} r={10} fill="#00e5ff" />
    </>
  );
}

function Pendulum({ obs }: { obs: number[] }) {
  const [cosT, sinT] = obs; // [cosθ, sinθ, 角速度]，θ=0 为竖直向上
  const cx = W / 2;
  const cy = H / 2;
  const len = 110;
  // θ 从竖直向上量起：末端 = pivot + len*(sinθ, -cosθ)... gym 约定 θ=0 朝上
  const ex = cx + len * sinT;
  const ey = cy - len * cosT;
  return (
    <>
      <circle cx={cx} cy={cy} r={120} fill="none" stroke="#1e293b" strokeWidth={1} />
      <line x1={cx} y1={cy} x2={ex} y2={ey} stroke="#00ff88" strokeWidth={9} strokeLinecap="round" />
      <circle cx={ex} cy={ey} r={12} fill="#ffab40" />
      <circle cx={cx} cy={cy} r={6} fill="#0a0e17" stroke="#94a3b8" strokeWidth={1.5} />
    </>
  );
}

export default function EnvPlot({ state, meta }: { state: EnvState; meta?: TrajectoryMeta }) {
  const obs = state.observation;
  const envId = meta?.envId ?? "";
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-xl bg-[rgba(10,14,23,0.6)]">
      {envId.startsWith("CartPole") && <CartPole obs={obs} />}
      {envId.startsWith("MountainCar") && <MountainCar obs={obs} />}
      {envId.startsWith("Pendulum") && <Pendulum obs={obs} />}
      <text x={16} y={26} className="font-mono" fill="#64748b" fontSize={12}>
        {envId}
      </text>
    </svg>
  );
}
