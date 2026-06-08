import { useRef, useEffect } from "react";
import { ClustersState } from "@/player/types";

const CANVAS_W = 500;
const CANVAS_H = 400;
const PAD = 30;

const CLUSTER_COLORS = ["#cc785c", "#5b7b9a", "#b04a3a", "#c99a4e", "#7a8b5a", "#b86a8a", "#8a9a55"];

interface Props {
  state: ClustersState;
  metrics: Record<string, number>;
}

export default function Clusters({ state }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.fillStyle = "rgba(250, 249, 245, 0.95)";
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    const { points, centroids, k } = state;
    if (points.length === 0) return;

    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
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

    for (const p of points) {
      const cx = toX(p.x);
      const cy = toY(p.y);
      // cluster < 0 表示噪声（如 DBSCAN），用灰色小点区分
      const isNoise = p.cluster < 0;
      const color = isNoise ? "#908e85" : CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length];

      ctx.beginPath();
      ctx.arc(cx, cy, isNoise ? 2 : 3.5, 0, Math.PI * 2);
      ctx.fillStyle = color + (isNoise ? "70" : "90");
      ctx.fill();
    }

    for (let i = 0; i < centroids.length; i++) {
      const c = centroids[i];
      const cx = toX(c.x);
      const cy = toY(c.y);
      const color = CLUSTER_COLORS[i % CLUSTER_COLORS.length];

      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.fillStyle = color + "30";
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      ctx.font = "bold 9px 'JetBrains Mono', monospace";
      ctx.fillStyle = color;
      ctx.fillText(`C${i}`, cx + 10, cy + 3);
    }

    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(107, 106, 101, 0.4)";
    ctx.fillText(`K=${k}`, CANVAS_W - PAD - 30, PAD + 15);
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
