import { useRef, useEffect } from "react";
import { ScatterBoundaryState } from "@/player/types";

const CANVAS_W = 500;
const CANVAS_H = 400;
const PAD = 30;

const CLASS_COLORS = ["#cc785c", "#5b7b9a", "#b04a3a", "#c99a4e", "#7a8b5a"];

interface Props {
  state: ScatterBoundaryState;
  metrics: Record<string, number>;
}

export default function ScatterBoundary({ state }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.fillStyle = "rgba(250, 249, 245, 0.95)";
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    const { points, boundary, gridX, gridY, xRange, yRange, fit } = state;
    const plotW = CANVAS_W - 2 * PAD;
    const plotH = CANVAS_H - 2 * PAD;

    const toCanvasX = (x: number) => PAD + ((x - xRange[0]) / (xRange[1] - xRange[0])) * plotW;
    const toCanvasY = (y: number) => CANVAS_H - PAD - ((y - yRange[0]) / (yRange[1] - yRange[0])) * plotH;

    if (boundary && boundary.length > 0) {
      const cellW = plotW / gridX;
      const cellH = plotH / gridY;
      let minVal = Infinity, maxVal = -Infinity;
      for (let i = 0; i < gridX; i++) {
        for (let j = 0; j < gridY; j++) {
          const v = boundary[i]?.[j] ?? 0;
          if (v < minVal) minVal = v;
          if (v > maxVal) maxVal = v;
        }
      }
      // 三种 boundary 语义：① 有负值 → 决策分数(二分类，阈值0)；
      // ② 全在[0,1] → 概率/0-1类别(阈值0.5)；③ 非负且 max>1 → 多分类标签(直接取整)
      const mode: "signed" | "prob" | "multiclass" =
        minVal < 0 ? "signed" : maxVal <= 1 ? "prob" : "multiclass";
      const classOf = (val: number): number => {
        if (mode === "multiclass") return Math.round(val);
        if (mode === "prob") return val >= 0.5 ? 1 : 0;
        return val >= 0 ? 1 : 0;
      };
      for (let i = 0; i < gridX; i++) {
        for (let j = 0; j < gridY; j++) {
          const val = boundary[i]?.[j] ?? 0;
          const cls = classOf(val);
          const color = CLASS_COLORS[((cls % CLASS_COLORS.length) + CLASS_COLORS.length) % CLASS_COLORS.length];
          ctx.fillStyle = color + "18";
          ctx.fillRect(PAD + i * cellW, PAD + (gridY - 1 - j) * cellH, cellW + 1, cellH + 1);
        }
      }
    }

    if (fit && "slope" in fit) {
      const { slope, intercept } = fit;
      const x1 = xRange[0];
      const x2 = xRange[1];
      const y1 = slope * x1 + intercept;
      const y2 = slope * x2 + intercept;
      ctx.strokeStyle = "#cc785c";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(toCanvasX(x1), toCanvasY(y1));
      ctx.lineTo(toCanvasX(x2), toCanvasY(y2));
      ctx.stroke();
    }

    ctx.strokeStyle = "rgba(214, 210, 196, 0.2)";
    ctx.lineWidth = 1;
    ctx.strokeRect(PAD, PAD, plotW, plotH);

    for (const p of points) {
      const cx = toCanvasX(p.x);
      const cy = toCanvasY(p.y);
      const color = CLASS_COLORS[p.label % CLASS_COLORS.length];

      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = color + "80";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(107, 106, 101, 0.4)";
    ctx.fillText(xRange[0].toFixed(1), PAD, CANVAS_H - 8);
    ctx.fillText(xRange[1].toFixed(1), CANVAS_W - PAD - 20, CANVAS_H - 8);
    ctx.fillText(yRange[1].toFixed(1), 2, PAD + 10);
    ctx.fillText(yRange[0].toFixed(1), 2, CANVAS_H - PAD);
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
