import { useRef, useEffect, useState, useCallback } from "react";
import { Play, Pause, RotateCcw, SkipForward } from "lucide-react";
import {
  CartPoleState,
  step,
  isTerminal,
  randomState,
  linearPolicy,
  OPTIMAL_WEIGHTS,
  X_THRESHOLD,
} from "@/utils/cartpole";

const CANVAS_W = 600;
const CANVAS_H = 300;
const CART_W = 60;
const CART_H = 30;
const POLE_LEN = 120;
const SCALE = CANVAS_W / (X_THRESHOLD * 2.5);
const CART_Y = CANVAS_H * 0.65;

const EPOCH_CONFIGS = [
  { label: "Epoch 0 (随机)", reward: "~22", color: "#b04a3a" },
  { label: "Epoch 1 (初学)", reward: "~173", color: "#c99a4e" },
  { label: "Epoch 2 (进步)", reward: "~304", color: "#5b7b9a" },
  { label: "Epoch 3 (接近)", reward: "~444", color: "#5b7b9a" },
  { label: "Epoch 4 (完美)", reward: "~500", color: "#cc785c" },
];

const WEIGHT_SCHEDULES = [
  [0.1, -0.3, 0.2, -0.1],
  [0.15, 0.3, 1.0, 0.5],
  [0.18, 0.6, 1.8, 1.2],
  [0.19, 0.75, 2.2, 1.6],
  OPTIMAL_WEIGHTS,
];

function drawScene(
  ctx: CanvasRenderingContext2D,
  state: CartPoleState,
  action: number,
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

  const cartX = CANVAS_W / 2 + state.x * SCALE;

  ctx.strokeStyle = "rgba(214, 210, 196, 0.3)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(30, CART_Y + CART_H / 2 + 8);
  ctx.lineTo(CANVAS_W - 30, CART_Y + CART_H / 2 + 8);
  ctx.stroke();

  ctx.fillStyle = "rgba(20, 20, 19, 0.9)";
  ctx.strokeStyle = config.color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(cartX - CART_W / 2, CART_Y - CART_H / 2, CART_W, CART_H, 4);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "rgba(214, 210, 196, 0.5)";
  ctx.beginPath();
  ctx.arc(cartX - CART_W / 4, CART_Y + CART_H / 2 + 4, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(cartX + CART_W / 4, CART_Y + CART_H / 2 + 4, 5, 0, Math.PI * 2);
  ctx.fill();

  const pivotX = cartX;
  const pivotY = CART_Y - CART_H / 2;
  const poleEndX = pivotX + Math.sin(state.theta) * POLE_LEN;
  const poleEndY = pivotY - Math.cos(state.theta) * POLE_LEN;

  const poleGrad = ctx.createLinearGradient(pivotX, pivotY, poleEndX, poleEndY);
  poleGrad.addColorStop(0, config.color);
  poleGrad.addColorStop(1, config.color + "40");
  ctx.strokeStyle = poleGrad;
  ctx.lineWidth = 6;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(pivotX, pivotY);
  ctx.lineTo(poleEndX, poleEndY);
  ctx.stroke();

  ctx.fillStyle = config.color;
  ctx.beginPath();
  ctx.arc(poleEndX, poleEndY, 5, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "rgba(204, 120, 92, 0.8)";
  ctx.beginPath();
  ctx.arc(pivotX, pivotY, 4, 0, Math.PI * 2);
  ctx.fill();

  if (action === 1) {
    ctx.fillStyle = "rgba(91, 123, 154, 0.7)";
    ctx.beginPath();
    ctx.moveTo(cartX + CART_W / 2 + 5, CART_Y);
    ctx.lineTo(cartX + CART_W / 2 + 15, CART_Y - 5);
    ctx.lineTo(cartX + CART_W / 2 + 15, CART_Y + 5);
    ctx.closePath();
    ctx.fill();
  } else {
    ctx.fillStyle = "rgba(91, 123, 154, 0.7)";
    ctx.beginPath();
    ctx.moveTo(cartX - CART_W / 2 - 5, CART_Y);
    ctx.lineTo(cartX - CART_W / 2 - 15, CART_Y - 5);
    ctx.lineTo(cartX - CART_W / 2 - 15, CART_Y + 5);
    ctx.closePath();
    ctx.fill();
  }

  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillStyle = "rgba(107, 106, 101, 0.6)";
  ctx.fillText(`Step: ${stepNum}`, 12, 20);
  ctx.fillText(`θ: ${(state.theta * 180 / Math.PI).toFixed(1)}°`, 12, 36);
  ctx.fillText(`x: ${state.x.toFixed(2)}`, 12, 52);

  ctx.font = "bold 12px 'JetBrains Mono', monospace";
  ctx.fillStyle = config.color;
  ctx.fillText(config.label, CANVAS_W - 140, 20);
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillStyle = "rgba(107, 106, 101, 0.6)";
  ctx.fillText(`奖励: ${config.reward}`, CANVAS_W - 140, 36);
}

export default function CartPoleViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const stateRef = useRef<CartPoleState>(randomState());
  const stepRef = useRef(0);
  const actionRef = useRef(0);

  const [epochIdx, setEpochIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [stepCount, setStepCount] = useState(0);
  const [fallen, setFallen] = useState(false);

  const epochIdxRef = useRef(0);
  epochIdxRef.current = epochIdx;

  const reset = useCallback(() => {
    stateRef.current = randomState();
    stepRef.current = 0;
    actionRef.current = 0;
    setStepCount(0);
    setFallen(false);
    cancelAnimationFrame(animRef.current);
    setPlaying(false);

    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) {
      drawScene(ctx, stateRef.current, 0, epochIdxRef.current, 0);
    }
  }, []);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    drawScene(ctx, stateRef.current, 0, epochIdx, 0);
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

      const weights = WEIGHT_SCHEDULES[epochIdxRef.current];
      const action = linearPolicy(stateRef.current, weights);
      actionRef.current = action;

      const newState = step(stateRef.current, action);
      stateRef.current = newState;
      stepRef.current += 1;
      setStepCount(stepRef.current);

      if (isTerminal(newState) || stepRef.current >= 500) {
        setFallen(isTerminal(newState));
        setPlaying(false);
        drawScene(ctx, newState, action, epochIdxRef.current, stepRef.current);
        return;
      }

      drawScene(ctx, newState, action, epochIdxRef.current, stepRef.current);
      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [playing]);

  const handleEpochChange = (idx: number) => {
    setEpochIdx(idx);
    reset();
  };

  return (
    <div className="glass rounded-xl p-6">
      <h2 className="text-sm font-bold text-stone-600 mb-4">
        🎮 策略进化可视化
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
              epochIdx === i
                ? "border-opacity-60"
                : "border-stone-300/50 text-stone-500 hover:border-stone-300"
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
            if (fallen) reset();
            setPlaying(!playing);
          }}
          className="w-10 h-10 rounded-lg border border-[#cc785c]/30 bg-[#cc785c]/10 flex items-center justify-center text-[#cc785c] hover:bg-[#cc785c]/20 transition-colors"
        >
          {playing ? (
            <Pause className="w-5 h-5" />
          ) : (
            <Play className="w-5 h-5" />
          )}
        </button>
        <button
          onClick={() => {
            if (!playing) {
              const weights = WEIGHT_SCHEDULES[epochIdx];
              const action = linearPolicy(stateRef.current, weights);
              const newState = step(stateRef.current, action);
              if (!isTerminal(newState) && stepRef.current < 500) {
                stateRef.current = newState;
                stepRef.current += 1;
                setStepCount(stepRef.current);
                actionRef.current = action;
                const ctx = canvasRef.current?.getContext("2d");
                if (ctx)
                  drawScene(
                    ctx,
                    newState,
                    action,
                    epochIdx,
                    stepRef.current
                  );
              } else {
                setFallen(true);
              }
            }
          }}
          className="w-8 h-8 rounded-lg border border-stone-300 flex items-center justify-center text-stone-500 hover:text-[#5b7b9a] hover:border-[#5b7b9a]/30 transition-colors"
        >
          <SkipForward className="w-4 h-4" />
        </button>
      </div>

      <div className="flex items-center justify-center gap-4 text-xs text-stone-500">
        <span>
          步数:{" "}
          <span className="font-mono text-stone-700">{stepCount}</span>
        </span>
        <span>
          状态:{" "}
          <span
            className={`font-mono ${
              fallen
                ? "text-[#b04a3a]"
                : stepCount >= 500
                ? "text-[#cc785c]"
                : "text-stone-700"
            }`}
          >
            {fallen ? "杆子倒了!" : stepCount >= 500 ? "完美平衡!" : "进行中..."}
          </span>
        </span>
      </div>

      <p className="text-[10px] text-stone-500 text-center mt-3">
        点击 Epoch 按钮切换训练阶段 → 播放观看策略进化 → Epoch 0 杆子秒倒，Epoch 4 完美平衡
      </p>
    </div>
  );
}
