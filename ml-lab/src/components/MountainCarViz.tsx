import { useRef, useEffect, useState, useCallback } from "react";
import { Play, Pause, RotateCcw, SkipForward } from "lucide-react";
import {
  MountainCarState,
  step,
  isTerminal,
  randomState,
  height,
  MIN_POS,
  MAX_POS,
  GOAL_POS,
  DQN_EPOCH_SCHEDULES,
  SHAPED_EPOCH_SCHEDULES,
  EpochSchedule,
} from "@/utils/mountaincar";

const CANVAS_W = 600;
const CANVAS_H = 300;
const PAD_X = 40;
const PAD_Y = 40;

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
  config: EpochSchedule,
  stepNum: number,
) {
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

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

  ctx.beginPath();
  ctx.strokeStyle = "rgba(214, 210, 196, 0.5)";
  ctx.lineWidth = 2;
  for (let p = MIN_POS; p <= MAX_POS; p += 0.01) {
    const { cx, cy } = posToCanvas(p);
    if (p === MIN_POS) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  }
  ctx.stroke();

  ctx.fillStyle = "rgba(214, 210, 196, 0.08)";
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
  ctx.fillStyle = "rgba(204, 120, 92, 0.3)";
  ctx.beginPath();
  ctx.arc(goalPos.cx, goalPos.cy, 12, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(204, 120, 92, 0.6)";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.font = "9px 'JetBrains Mono', monospace";
  ctx.fillStyle = "#cc785c";
  ctx.fillText("🎯", goalPos.cx - 5, goalPos.cy - 16);

  const carPos = posToCanvas(state.position);
  ctx.fillStyle = config.color;
  ctx.beginPath();
  ctx.arc(carPos.cx, carPos.cy - 8, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.3)";
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.fillStyle = "rgba(214, 210, 196, 0.4)";
  ctx.beginPath();
  ctx.arc(carPos.cx - 5, carPos.cy, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(carPos.cx + 5, carPos.cy, 3, 0, Math.PI * 2);
  ctx.fill();

  if (action === 0) {
    ctx.fillStyle = "rgba(91, 123, 154, 0.7)";
    ctx.beginPath();
    ctx.moveTo(carPos.cx - 14, carPos.cy - 8);
    ctx.lineTo(carPos.cx - 22, carPos.cy - 12);
    ctx.lineTo(carPos.cx - 22, carPos.cy - 4);
    ctx.closePath();
    ctx.fill();
  } else if (action === 2) {
    ctx.fillStyle = "rgba(91, 123, 154, 0.7)";
    ctx.beginPath();
    ctx.moveTo(carPos.cx + 14, carPos.cy - 8);
    ctx.lineTo(carPos.cx + 22, carPos.cy - 12);
    ctx.lineTo(carPos.cx + 22, carPos.cy - 4);
    ctx.closePath();
    ctx.fill();
  }

  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillStyle = "rgba(107, 106, 101, 0.6)";
  ctx.fillText(`Step: ${stepNum}`, 12, 20);
  ctx.fillText(`Pos: ${state.position.toFixed(3)}`, 12, 36);
  ctx.fillText(`Vel: ${state.velocity.toFixed(4)}`, 12, 52);

  ctx.font = "bold 12px 'JetBrains Mono', monospace";
  ctx.fillStyle = config.color;
  ctx.fillText(config.label, CANVAS_W - 120, 20);
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillStyle = "rgba(107, 106, 101, 0.5)";
  ctx.fillText(config.sub, CANVAS_W - 120, 36);

  const actionLabels = ["← 左推", "· 不推", "→ 右推"];
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillStyle = "rgba(91, 123, 154, 0.6)";
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

  const schedules = variant === "dqn" ? DQN_EPOCH_SCHEDULES : SHAPED_EPOCH_SCHEDULES;
  const [epochIdx, setEpochIdx] = useState(schedules.length - 1);
  const [playing, setPlaying] = useState(false);
  const [stepCount, setStepCount] = useState(0);
  const [reached, setReached] = useState(false);

  const epochIdxRef = useRef(schedules.length - 1);
  epochIdxRef.current = epochIdx;

  const currentConfig = schedules[epochIdx];

  const getAction = useCallback(
    (state: MountainCarState, idx: number) => {
      return schedules[idx].policy(state);
    },
    [schedules],
  );

  const reset = useCallback(() => {
    stateRef.current = randomState();
    stepRef.current = 0;
    actionRef.current = 1;
    setStepCount(0);
    setReached(false);
    cancelAnimationFrame(animRef.current);
    setPlaying(false);
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) drawScene(ctx, stateRef.current, 1, schedules[epochIdxRef.current], 0);
  }, [schedules]);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) drawScene(ctx, stateRef.current, 1, currentConfig, 0);
  }, [epochIdx, currentConfig]);

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

      const action = getAction(stateRef.current, epochIdxRef.current);
      actionRef.current = action;

      const newState = step(stateRef.current, action);
      stateRef.current = newState;
      stepRef.current += 1;
      setStepCount(stepRef.current);

      const cfg = schedules[epochIdxRef.current];

      if (isTerminal(newState)) {
        setReached(true);
        setPlaying(false);
        drawScene(ctx, newState, action, cfg, stepRef.current);
        return;
      }

      if (stepRef.current >= 200) {
        setPlaying(false);
        drawScene(ctx, newState, action, cfg, stepRef.current);
        return;
      }

      drawScene(ctx, newState, action, cfg, stepRef.current);
      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [playing, schedules, getAction]);

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
    if (ctx) drawScene(ctx, stateRef.current, 1, schedules[idx], 0);
  };

  return (
    <div className="glass rounded-xl p-6">
      <h2 className="text-sm font-bold text-stone-600 mb-4">
        🏔️ MountainCar 策略可视化
        {variant === "dqn" && (
          <span className="text-[#b04a3a] ml-2 text-xs font-normal">（纯 DQN — 随机策略，永远到不了）</span>
        )}
        {variant === "shaped" && (
          <span className="text-[#c99a4e] ml-2 text-xs font-normal">（奖励塑形 — 动量策略逐步改善）</span>
        )}
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
        {schedules.map((cfg, i) => (
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
            if (stepCount >= 200 || reached) reset();
            setPlaying(!playing);
          }}
          className="w-10 h-10 rounded-lg border border-[#cc785c]/30 bg-[#cc785c]/10 flex items-center justify-center text-[#cc785c] hover:bg-[#cc785c]/20 transition-colors"
        >
          {playing ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
        </button>
        <button
          onClick={() => {
            if (!playing) {
              const action = getAction(stateRef.current, epochIdx);
              const newState = step(stateRef.current, action);
              stateRef.current = newState;
              stepRef.current += 1;
              setStepCount(stepRef.current);
              actionRef.current = action;
              if (isTerminal(newState)) setReached(true);
              const ctx = canvasRef.current?.getContext("2d");
              if (ctx) drawScene(ctx, newState, action, currentConfig, stepRef.current);
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
          状态:{" "}
          <span className={`font-mono ${reached ? "text-[#cc785c]" : stepCount >= 200 ? "text-[#b04a3a]" : "text-stone-700"}`}>
            {reached ? "到达山顶! 🎉" : stepCount >= 200 ? "超时失败" : "进行中..."}
          </span>
        </span>
      </div>

      <p className="text-[10px] text-stone-500 text-center mt-3">
        {variant === "dqn"
          ? "纯 DQN：每步都是 -1，智能体从未体验过到达山顶 → 永远随机探索 → 永远到不了"
          : "动量策略：速度为负→左推（积蓄动量），速度为正→右推（冲向山顶）。后期噪声降低，策略更精准"}
      </p>
    </div>
  );
}
