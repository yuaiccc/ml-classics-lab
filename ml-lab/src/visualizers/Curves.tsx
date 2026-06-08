import { useRef, useEffect } from "react";
import { CurvesState } from "@/player/types";

const CANVAS_W = 500;
const CANVAS_H = 400;
const PAD = 30;

const LABEL_COLORS = ["#cc785c", "#5b7b9a", "#b04a3a", "#c99a4e", "#7a8b5a", "#b86a8a"];

interface Props {
  state: CurvesState;
  metrics: Record<string, number>;
}

export default function Curves({ state }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.fillStyle = "rgba(250, 249, 245, 0.95)";
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    const { embedding } = state;
    if (!embedding || embedding.length === 0) return;

    const xs = embedding.map((p) => p.x);
    const ys = embedding.map((p) => p.y);
    const xMin = Math.min(...xs) - 0.5;
    const xMax = Math.max(...xs) + 0.5;
    const yMin = Math.min(...ys) - 0.5;
    const yMax = Math.max(...ys) + 0.5;

    const plotW = CANVAS_W - 2 * PAD;
    const plotH = CANVAS_H - 2 * PAD;
    const toX = (x: number) => PAD + ((x - xMin) / (xMax - xMin)) * plotW;
    const toY = (y: number) => CANVAS_H - PAD - ((y - yMin) / (yMax - yMin)) * plotH;

    ctx.strokeStyle = "rgba(214, 210, 196, 0.2)";
    ctx.lineWidth = 1;
    ctx.strokeRect(PAD, PAD, plotW, plotH);

    for (const p of embedding) {
      const cx = toX(p.x);
      const cy = toY(p.y);
      const color = LABEL_COLORS[p.label % LABEL_COLORS.length];

      ctx.beginPath();
      ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = color + "90";
      ctx.fill();
    }

    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(107, 106, 101, 0.4)";
    ctx.fillText(xMin.toFixed(1), PAD, CANVAS_H - 8);
    ctx.fillText(xMax.toFixed(1), CANVAS_W - PAD - 25, CANVAS_H - 8);
    ctx.fillText(yMax.toFixed(1), 2, PAD + 10);
    ctx.fillText(yMin.toFixed(1), 2, CANVAS_H - PAD);
  }, [state]);

  return (
    <canvas
      ref={canvasRef}
      width={CANVAS_W}
      height={CANVAS_H}
      className="rounded-lg border border-stone-200/50"
      style={{ maxWidth: "100%", height: "auto" }}
    />
  );
}
