import { useRef, useEffect, useState, useCallback } from "react";
import { Play, Pause, RotateCcw, SkipForward } from "lucide-react";
import {
  PendulumState,
  step,
  randomState,
  sacPolicy,
  WEIGHT_SCHEDULES,
  MAX_TORQUE,
} from "@/utils/pendulum";

const CANVAS_W = 600;
const CANVAS_H = 300;
const PIVOT_X = CANVAS_W / 2;
const PIVOT_Y = CANVAS_H * 0.35;
const POLE_LEN = 120;

const EPOCH_CONFIGS = [
  { label: "Epoch 0 (随机)", reward: "~-1480", color: "#b04a3a", noise: 4.0 },
  { label: "Epoch 1 (初学)", reward: "~-195", color: "#c99a4e", noise: 1.5 },
  { label: "Epoch 2 (进步)", reward: "~-152", color: "#5b7b9a", noise: 0.5 },
  { label: "Epoch 3 (稳定)", reward: "~-147", color: "#cc785c", noise: 0.1 },
];

function drawScene(
  ctx: CanvasRenderingContext2D,
  state: PendulumState,
  torque: number,
  epochIdx: number,
  stepNum: number
) {
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  const config = EPOCH_CONFIGS[epochIdx];

  ctx.fillStyle = "rgba(250, 249, 245, 0.95)";
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  ctx.strokeStyle = "rgba(204, 120, 92, 0.08)";
  ctx.lineWidth = 1;
  for (let i = 0; i < CANVAS_W; i += 30) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, CANVAS_H);
    ctx.stroke();
  }
  for (let i = 0; i < CANVAS_H; i += 30) {
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(CANVAS_W, i);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(204, 120, 92, 0.15)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(PIVOT_X, PIVOT_Y - POLE_LEN - 20);
  ctx.lineTo(PIVOT_X, PIVOT_Y - POLE_LEN - 20);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "rgba(204, 120, 92, 0.08)";
  ctx.beginPath();
  ctx.arc(PIVOT_X, PIVOT_Y - POLE_LEN, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(204, 120, 92, 0.3)";
  ctx.lineWidth = 1;
  ctx.stroke();

  const endX = PIVOT_X + Math.sin(state.theta) * POLE_LEN;
  const endY = PIVOT_Y + Math.cos(state.theta) * POLE_LEN;

  const poleGrad = ctx.createLinearGradient(PIVOT_X, PIVOT_Y, endX, endY);
  poleGrad.addColorStop(0, config.color);
  poleGrad.addColorStop(1, config.color + "60");
  ctx.strokeStyle = poleGrad;
  ctx.lineWidth = 8;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(PIVOT_X, PIVOT_Y);
  ctx.lineTo(endX, endY);
  ctx.stroke();

  ctx.fillStyle = config.color;
  ctx.beginPath();
  ctx.arc(endX, endY, 10, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "rgba(204, 120, 92, 0.8)";
  ctx.beginPath();
  ctx.arc(PIVOT_X, PIVOT_Y, 5, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "rgba(214, 210, 196, 0.3)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(PIVOT_X, PIVOT_Y, 18, 0, Math.PI * 2);
  ctx.stroke();

  if (Math.abs(torque) > 0.1) {
    const arcStart = state.theta - Math.PI / 2;
    const arcLen = (torque / MAX_TORQUE) * Math.PI / 3;
    ctx.strokeStyle = "rgba(91, 123, 154, 0.5)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(PIVOT_X, PIVOT_Y, 25, arcStart, arcStart + arcLen, torque < 0);
    ctx.stroke();

    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(91, 123, 154, 0.6)";
    ctx.fillText(
      `τ=${torque.toFixed(2)}`,
      PIVOT_X + 30,
      PIVOT_Y - 5
    );
  }

  const angleDeg = (state.theta * 180 / Math.PI);
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillStyle = "rgba(107, 106, 101, 0.6)";
  ctx.fillText(`Step: ${stepNum}`, 12, 20);
  ctx.fillText(`θ: ${angleDeg.toFixed(1)}°`, 12, 36);
  ctx.fillText(`ω: ${state.thetaDot.toFixed(2)}`, 12, 52);

  ctx.font = "bold 12px 'JetBrains Mono', monospace";
  ctx.fillStyle = config.color;
  ctx.fillText(config.label, CANVAS_W - 150, 20);
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillStyle = "rgba(107, 106, 101, 0.5)";
  ctx.fillText(`奖励: ${config.reward}`, CANVAS_W - 150, 36);

  const upright = Math.abs(angleDeg) < 15;
  if (upright) {
    ctx.font = "bold 11px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#cc785c";
    ctx.fillText("✓ 近似竖直!", CANVAS_W - 150, 52);
  }
}

export default function PendulumViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const stateRef = useRef<PendulumState>(randomState());
  const stepRef = useRef(0);
  const torqueRef = useRef(0);

  const [epochIdx, setEpochIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [stepCount, setStepCount] = useState(0);

  const epochIdxRef = useRef(0);
  epochIdxRef.current = epochIdx;

  const reset = useCallback(() => {
    stateRef.current = randomState();
    stepRef.current = 0;
    torqueRef.current = 0;
    setStepCount(0);
    cancelAnimationFrame(animRef.current);
    setPlaying(false);
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) drawScene(ctx, stateRef.current, 0, epochIdxRef.current, 0);
  }, []);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) drawScene(ctx, stateRef.current, 0, epochIdx, 0);
  }, [epochIdx]);

  useEffect(() => {
    if (!playing) {
      cancelAnimationFrame(animRef.current);
      return;
    }
    let lastTime = 0;
    const FRAME_INTERVAL = 1000 / 50;

    const loop = (time: number) => {
      if (time - lastTime < FRAME_INTERVAL) {
        animRef.current = requestAnimationFrame(loop);
        return;
      }
      lastTime = time;

      const ctx = canvasRef.current?.getContext("2d");
      if (!ctx) return;

      const config = EPOCH_CONFIGS[epochIdxRef.current];
      const weights = WEIGHT_SCHEDULES[epochIdxRef.current];
      const torque = sacPolicy(stateRef.current, weights, config.noise);
      torqueRef.current = torque;

      const newState = step(stateRef.current, torque);
      stateRef.current = newState;
      stepRef.current += 1;
      setStepCount(stepRef.current);

      if (stepRef.current >= 200) {
        setPlaying(false);
      }

      drawScene(ctx, newState, torque, epochIdxRef.current, stepRef.current);
      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [playing]);

  const handleEpochChange = (idx: number) => {
    setEpochIdx(idx);
    stateRef.current = randomState();
    stepRef.current = 0;
    torqueRef.current = 0;
    setStepCount(0);
    cancelAnimationFrame(animRef.current);
    setPlaying(false);
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) drawScene(ctx, stateRef.current, 0, idx, 0);
  };

  return (
    <div className="glass rounded-xl p-6">
      <h2 className="text-sm font-bold text-stone-600 mb-4">
        🔄 Pendulum 策略进化可视化
      </h2>

      <div className="flex justify-center mb-4">
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          className="rounded-lg border border-stone-200/50"
          style={{ maxWidth: "100%", height: "auto" }}
        />
      </div>

      <div className="flex justify-center gap-2 mb-4">
        {EPOCH_CONFIGS.map((cfg, i) => (
          <button
            key={i}
            onClick={() => handleEpochChange(i)}
            className={`px-3 py-1.5 rounded-lg text-[10px] font-mono border transition-all duration-200 ${
              epochIdx === i ? "" : "border-stone-300/50 text-stone-500 hover:border-stone-300"
            }`}
            style={
              epochIdx === i
                ? {
                    borderColor: cfg.color + "80",
                    color: cfg.color,
                    background: cfg.color + "15",
                    boxShadow: `0 0 10px ${cfg.color}20`,
                  }
                : {}
            }
          >
            {cfg.label}
          </button>
        ))}
      </div>

      <div className="flex items-center justify-center gap-3 mb-3">
        <button
          onClick={reset}
          className="w-8 h-8 rounded-lg border border-stone-300 flex items-center justify-center text-stone-500 hover:text-[#cc785c] hover:border-[#cc785c]/30 transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
        <button
          onClick={() => {
            if (stepCount >= 200) reset();
            setPlaying(!playing);
          }}
          className="w-10 h-10 rounded-lg border border-[#cc785c]/30 bg-[#cc785c]/10 flex items-center justify-center text-[#cc785c] hover:bg-[#cc785c]/20 transition-colors"
        >
          {playing ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
        </button>
        <button
          onClick={() => {
            if (!playing) {
              const config = EPOCH_CONFIGS[epochIdx];
              const weights = WEIGHT_SCHEDULES[epochIdx];
              const torque = sacPolicy(stateRef.current, weights, config.noise);
              const newState = step(stateRef.current, torque);
              stateRef.current = newState;
              stepRef.current += 1;
              setStepCount(stepRef.current);
              torqueRef.current = torque;
              const ctx = canvasRef.current?.getContext("2d");
              if (ctx) drawScene(ctx, newState, torque, epochIdx, stepRef.current);
            }
          }}
          className="w-8 h-8 rounded-lg border border-stone-300 flex items-center justify-center text-stone-500 hover:text-[#5b7b9a] hover:border-[#5b7b9a]/30 transition-colors"
        >
          <SkipForward className="w-4 h-4" />
        </button>
      </div>

      <div className="flex items-center justify-center gap-4 text-xs text-stone-500">
        <span>
          步数: <span className="font-mono text-stone-700">{stepCount}</span>/200
        </span>
        <span>
          力矩:{" "}
          <span className="font-mono text-[#5b7b9a]">
            {torqueRef.current.toFixed(2)}
          </span>
        </span>
      </div>

      <p className="text-[10px] text-stone-500 text-center mt-3">
        Epoch 0 杆子疯狂旋转 → Epoch 3 杆子稳定在竖直位置 | 弧线表示施加的力矩方向和大小
      </p>
    </div>
  );
}
