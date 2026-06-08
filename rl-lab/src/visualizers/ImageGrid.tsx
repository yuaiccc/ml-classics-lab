import { useRef, useEffect } from "react";
import { ImageGridState, ImageGridGroup } from "./lab2-types";

interface Props {
  state: ImageGridState;
}

// sequential：深底 → 青绿高亮（匹配主题）；diverging：蓝 ← 白 → 红（0.5 为中性）
function colorFor(v: number, colormap: "diverging" | "sequential"): [number, number, number] {
  const t = Math.max(0, Math.min(1, v));
  if (colormap === "diverging") {
    if (t < 0.5) {
      const s = t / 0.5; // 蓝→白
      return [Math.round(40 + s * 215), Math.round(110 + s * 145), 255];
    }
    const s = (t - 0.5) / 0.5; // 白→红
    return [255, Math.round(255 - s * 215), Math.round(255 - s * 215)];
  }
  return [Math.round(10 + t * -10), Math.round(14 + t * 241), Math.round(23 + t * 113)];
}

function drawImage(
  ctx: CanvasRenderingContext2D,
  pixels: number[],
  w: number,
  h: number,
  scale: number,
  colormap: "diverging" | "sequential",
) {
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const [r, g, b] = colorFor(pixels[i] ?? 0, colormap);
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  // 先画到离屏小 canvas，再放大（保持像素块、不插值）
  const off = document.createElement("canvas");
  off.width = w;
  off.height = h;
  off.getContext("2d")!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, w * scale, h * scale);
}

function GroupRow({ group }: { group: ImageGridGroup }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { images, w, h, colormap } = group;
  const scale = w <= 6 ? 14 : 5; // 卷积核放大更多
  const gap = 6;
  const cellW = w * scale;
  const cellH = h * scale;
  const perRow = Math.min(images.length, 12);
  const rows = Math.ceil(images.length / perRow);
  const canvasW = perRow * cellW + (perRow - 1) * gap;
  const canvasH = rows * cellH + (rows - 1) * gap;

  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvasW, canvasH);
    images.forEach((pix, i) => {
      const col = i % perRow;
      const row = Math.floor(i / perRow);
      ctx.save();
      ctx.translate(col * (cellW + gap), row * (cellH + gap));
      drawImage(ctx, pix, w, h, scale, colormap);
      ctx.restore();
    });
  }, [images, w, h, scale, colormap, canvasW, canvasH, cellW, cellH, perRow]);

  return (
    <div className="flex flex-col gap-1">
      <div className="text-[11px] text-stone-500 font-mono">{group.title}</div>
      <canvas ref={ref} width={canvasW} height={canvasH} style={{ maxWidth: "100%", height: "auto" }} />
    </div>
  );
}

function Probe({ probe }: { probe: NonNullable<ImageGridState["probe"]> }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const scale = 3;
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    drawImage(ctx, probe.pixels, probe.w, probe.h, scale, "sequential");
  }, [probe]);
  return (
    <div className="flex flex-col gap-1">
      <div className="text-[11px] text-stone-500 font-mono">
        探针输入{probe.label !== undefined ? ` (数字 ${probe.label})` : ""}
      </div>
      <canvas ref={ref} width={probe.w * scale} height={probe.h * scale} className="rounded border border-stone-200/50" />
    </div>
  );
}

export default function ImageGrid({ state }: Props) {
  return (
    <div className="flex flex-col items-center gap-5 w-full">
      {state.probe && <Probe probe={state.probe} />}
      {state.groups.map((g, i) => (
        <GroupRow key={i} group={g} />
      ))}
    </div>
  );
}
