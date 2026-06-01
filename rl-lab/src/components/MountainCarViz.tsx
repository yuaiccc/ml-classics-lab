import { useRef, useEffect, useState, useCallback } from "react";
import { Play, Pause, RotateCcw, SkipForward } from "lucide-react";
import {
  MountainCarState,
  step,
  isTerminal,
  randomState,
  height,
  linearPolicy,
  randomPolicy,
  MIN_POS,
  MAX_POS,
  GOAL_POS,
  WEIGHT_SCHEDULES,
} from "@/utils/mountaincar";

const CANVAS_W = 600;
const CANVAS_H = 300;
const PAD_X = 40;
const PAD_Y = 40;

const EPOCH_CONFIGS_DQN = [
  { label: "随机探索", sub: "完全随机", color: "#ff5252" },
  { label: "随机探索", sub: "仍无信号", color: "#ff5252" },
  { label: "随机探索", sub: "持续失败", color: "#ff5252" },
];

const EPOCH_CONFIGS_SHAPED = [
  { label: "初期", sub: "开始学", color: "#ffab40" },
  { label: "中期", sub: "有进步", color: "#00e5ff" },
  { label: "后期", sub: "接近目标", color: "#00ff88" },
];

function posToCanvas(pos: number): { cx: number; cy: number } {
  const t = (pos - MIN_POS) / (MAX_POS - MIN_POS);
  const cx = PAD_X + t * (CANVAS_W - 2 * PAD_X);
  const h = height(pos);
  const cy = CANVAS_H - PAD_Y - h * (CANVAS_H - 2 * PAD_Y);
  return { cx, cy };
}

function drawScene(
  ctx: CanvasRenderingContext2D,
  state: MountainCarState,
  action: number,
  epochIdx: number,
  stepNum: number,
  configs: typeof EPOCH_CONFIGS_DQN,
  variant: "dqn" | "shaped"
) {
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  const config = configs[epochIdx];

  ctx.fillStyle = "rgba(10, 14, 23, 0.95)";
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  ctx.strokeStyle = "rgba(0, 255, 136, 0.08)";
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

  ctx.beginPath();
  ctx.strokeStyle = "rgba(100, 116, 139, 0.5)";
  ctx.lineWidth = 2;
  for (let p = MIN_POS; p <= MAX_POS; p += 0.01) {
    const { cx, cy } = posToCanvas(p);
    if (p === MIN_POS) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  }
  ctx.stroke();

  ctx.fillStyle = "rgba(100, 116, 139, 0.08)";
  ctx.beginPath();
  for (let p = MIN_POS; p <= MAX_POS; p += 0.01) {
    const { cx, cy } = posToCanvas(p);
    if (p === MIN_POS) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  }
  ctx.lineTo(CANVAS_W - PAD_X, CANVAS_H - PAD_Y);
  ctx.lineTo(PAD_X, CANVAS_H - PAD_Y);
  ctx.closePath();
  ctx.fill();

  const goalPos = posToCanvas(GOAL_POS);
  ctx.fillStyle = "rgba(0, 255, 136, 0.3)";
  ctx.beginPath();
  ctx.arc(goalPos.cx, goalPos.cy, 12, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(0, 255, 136, 0.6)";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.font = "9px 'JetBrains Mono', monospace";
  ctx.fillStyle = "#00ff88";
  ctx.fillText("🎯", goalPos.cx - 5, goalPos.cy - 16);

  const carPos = posToCanvas(state.position);
  ctx.fillStyle = config.color;
  ctx.beginPath();
  ctx.arc(carPos.cx, carPos.cy - 8, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.3)";
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.fillStyle = "rgba(100, 116, 139, 0.4)";
  ctx.beginPath();
  ctx.arc(carPos.cx - 5, carPos.cy, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(carPos.cx + 5, carPos.cy, 3, 0, Math.PI * 2);
  ctx.fill();

  if (action === 0) {
    ctx.fillStyle = "rgba(0, 229, 255, 0.7)";
    ctx.beginPath();
    ctx.moveTo(carPos.cx - 14, carPos.cy - 8);
    ctx.lineTo(carPos.cx - 22, carPos.cy - 12);
    ctx.lineTo(carPos.cx - 22, carPos.cy - 4);
    ctx.closePath();
    ctx.fill();
  } else if (action === 2) {
    ctx.fillStyle = "rgba(0, 229, 255, 0.7)";
    ctx.beginPath();
    ctx.moveTo(carPos.cx + 14, carPos.cy - 8);
    ctx.lineTo(carPos.cx + 22, carPos.cy - 12);
    ctx.lineTo(carPos.cx + 22, carPos.cy - 4);
    ctx.closePath();
    ctx.fill();
  }

  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
  ctx.fillText(`Step: ${stepNum}`, 12, 20);
  ctx.fillText(`Pos: ${state.position.toFixed(3)}`, 12, 36);
  ctx.fillText(`Vel: ${state.velocity.toFixed(4)}`, 12, 52);

  ctx.font = "bold 12px 'JetBrains Mono', monospace";
  ctx.fillStyle = config.color;
  ctx.fillText(config.label, CANVAS_W - 120, 20);
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
  ctx.fillText(config.sub, CANVAS_W - 120, 36);

  const actionLabels = ["← 左推", "· 不推", "→ 右推"];
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillStyle = "rgba(0, 229, 255, 0.6)";
  ctx.fillText(`动作: ${actionLabels[action]}`, CANVAS_W - 120, 52);
}

interface Props {
  variant: "dqn" | "shaped";
}

export default function MountainCarViz({ variant }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const stateRef = useRef<MountainCarState>(randomState());
  const stepRef = useRef(0);
  const actionRef = useRef(1);

  const configs = variant === "dqn" ? EPOCH_CONFIGS_DQN : EPOCH_CONFIGS_SHAPED;
  const [epochIdx, setEpochIdx] = useState(configs.length - 1);
  const [playing, setPlaying] = useState(false);
  const [stepCount, setStepCount] = useState(0);
  const [reached, setReached] = useState(false);

  const epochIdxRef = useRef(configs.length - 1);
  epochIdxRef.current = epochIdx;

  const reset = useCallback(() => {
    stateRef.current = randomState();
    stepRef.current = 0;
    actionRef.current = 1;
    setStepCount(0);
    setReached(false);
    cancelAnimationFrame(animRef.current);
    setPlaying(false);
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) drawScene(ctx, stateRef.current, 1, epochIdxRef.current, 0, configs, variant);
  }, [configs, variant]);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) drawScene(ctx, stateRef.current, 1, epochIdx, 0, configs, variant);
  }, [epochIdx, configs, variant]);

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

      let action: number;
      if (variant === "dqn") {
        action = randomPolicy();
      } else {
        const weights = WEIGHT_SCHEDULES.shaped[epochIdxRef.current];
        action = linearPolicy(stateRef.current, weights);
      }
      actionRef.current = action;

      const newState = step(stateRef.current, action);
      stateRef.current = newState;
      stepRef.current += 1;
      setStepCount(stepRef.current);

      if (isTerminal(newState)) {
        setReached(true);
        setPlaying(false);
        drawScene(ctx, newState, action, epochIdxRef.current, stepRef.current, configs, variant);
        return;
      }

      if (stepRef.current >= 200) {
        setPlaying(false);
        drawScene(ctx, newState, action, epochIdxRef.current, stepRef.current, configs, variant);
        return;
      }

      drawScene(ctx, newState, action, epochIdxRef.current, stepRef.current, configs, variant);
      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [playing, variant, configs]);

  const handleEpochChange = (idx: number) => {
    setEpochIdx(idx);
    stateRef.current = randomState();
    stepRef.current = 0;
    actionRef.current = 1;
    setStepCount(0);
    setReached(false);
    cancelAnimationFrame(animRef.current);
    setPlaying(false);
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) drawScene(ctx, stateRef.current, 1, idx, 0, configs, variant);
  };

  return (
    <div className="glass rounded-xl p-6">
      <h2 className="text-sm font-bold text-slate-400 mb-4">
        🏔️ MountainCar 策略可视化
        {variant === "dqn" && (
          <span className="text-[#ff5252] ml-2 text-xs font-normal">（纯 DQN — 随机策略）</span>
        )}
        {variant === "shaped" && (
          <span className="text-[#ffab40] ml-2 text-xs font-normal">（奖励塑形 — 逐步改善）</span>
        )}
      </h2>

      <div className="flex justify-center mb-4">
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          className="rounded-lg border border-slate-800/50"
          style={{ maxWidth: "100%", height: "auto" }}
        />
      </div>

      <div className="flex justify-center gap-2 mb-4">
        {configs.map((cfg, i) => (
          <button
            key={i}
            onClick={() => handleEpochChange(i)}
            className={`px-3 py-1.5 rounded-lg text-[10px] font-mono border transition-all duration-200 ${
              epochIdx === i ? "" : "border-slate-700/50 text-slate-600 hover:border-slate-600"
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
          className="w-8 h-8 rounded-lg border border-slate-700 flex items-center justify-center text-slate-500 hover:text-[#00ff88] hover:border-[#00ff88]/30 transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
        <button
          onClick={() => {
            if (stepCount >= 200 || reached) reset();
            setPlaying(!playing);
          }}
          className="w-10 h-10 rounded-lg border border-[#00ff88]/30 bg-[#00ff88]/10 flex items-center justify-center text-[#00ff88] hover:bg-[#00ff88]/20 transition-colors"
        >
          {playing ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
        </button>
        <button
          onClick={() => {
            if (!playing) {
              let action: number;
              if (variant === "dqn") action = randomPolicy();
              else {
                const weights = WEIGHT_SCHEDULES.shaped[epochIdx];
                action = linearPolicy(stateRef.current, weights);
              }
              const newState = step(stateRef.current, action);
              stateRef.current = newState;
              stepRef.current += 1;
              setStepCount(stepRef.current);
              actionRef.current = action;
              if (isTerminal(newState)) setReached(true);
              const ctx = canvasRef.current?.getContext("2d");
              if (ctx) drawScene(ctx, newState, action, epochIdx, stepRef.current, configs, variant);
            }
          }}
          className="w-8 h-8 rounded-lg border border-slate-700 flex items-center justify-center text-slate-500 hover:text-[#00e5ff] hover:border-[#00e5ff]/30 transition-colors"
        >
          <SkipForward className="w-4 h-4" />
        </button>
      </div>

      <div className="flex items-center justify-center gap-4 text-xs text-slate-500">
        <span>
          步数: <span className="font-mono text-slate-300">{stepCount}</span>/200
        </span>
        <span>
          状态:{" "}
          <span className={`font-mono ${reached ? "text-[#00ff88]" : stepCount >= 200 ? "text-[#ff5252]" : "text-slate-300"}`}>
            {reached ? "到达山顶! 🎉" : stepCount >= 200 ? "超时失败" : "进行中..."}
          </span>
        </span>
      </div>

      <p className="text-[10px] text-slate-600 text-center mt-3">
        {variant === "dqn"
          ? "纯 DQN：每步都是 -1，智能体从未体验过到达山顶 → 永远随机探索 → 永远到不了"
          : "奖励塑形：给位置提升和速度额外奖励 → 智能体逐渐学会先退后冲的策略"}
      </p>
    </div>
  );
}
