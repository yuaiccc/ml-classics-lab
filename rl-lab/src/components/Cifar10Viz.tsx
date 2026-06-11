import { useEffect, useRef, useState } from "react";

interface PixelGrid {
  index?: number;
  size?: number;
  pixels: number[];
}

interface Cifar10Result {
  finalAccuracy: number;
  finalLoss: number;
  visuals: {
    classNames: string[];
    sample: PixelGrid & {
      label: number;
      prediction: number;
      confidence: number;
    };
    conv1Kernels: PixelGrid[];
    conv1FeatureMaps: PixelGrid[];
    mistakes: Array<PixelGrid & { label: number; prediction: number; confidence: number }>;
  };
}

function CanvasGrid({ grid, size, rgb = false, label }: { grid: PixelGrid; size: number; rgb?: boolean; label: string }) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(size, size);
    for (let i = 0; i < size * size; i++) {
      const offset = i * 4;
      if (rgb) {
        imageData.data[offset] = grid.pixels[i * 3] ?? 0;
        imageData.data[offset + 1] = grid.pixels[i * 3 + 1] ?? 0;
        imageData.data[offset + 2] = grid.pixels[i * 3 + 2] ?? 0;
      } else {
        const value = grid.pixels[i] ?? 0;
        imageData.data[offset] = value;
        imageData.data[offset + 1] = value;
        imageData.data[offset + 2] = value;
      }
      imageData.data[offset + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }, [grid, rgb, size]);

  return (
    <div className="rounded-lg border border-stone-200 bg-stone-100/40 p-2">
      <canvas
        ref={ref}
        width={size}
        height={size}
        className="w-full aspect-square rounded-md bg-black image-rendering-pixelated"
        aria-label={label}
      />
      <div className="mt-1 text-center text-[10px] font-mono text-stone-500">{label}</div>
    </div>
  );
}

export default function Cifar10Viz() {
  const [result, setResult] = useState<Cifar10Result | null>(null);

  useEffect(() => {
    let mounted = true;
    fetch("/results/cifar10-cnn.json")
      .then((res) => res.json())
      .then((data) => {
        if (mounted) setResult(data);
      })
      .catch(() => {
        if (mounted) setResult(null);
      });
    return () => {
      mounted = false;
    };
  }, []);

  if (!result) {
    return (
      <div className="glass rounded-xl p-6 mb-8">
        <div className="text-sm text-stone-500">CIFAR-10 可视化数据尚未加载。</div>
      </div>
    );
  }

  const sample = result.visuals.sample;
  const names = result.visuals.classNames;

  return (
    <div className="glass rounded-xl p-6 mb-8">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-5">
        <h2 className="text-sm font-bold text-stone-600">CIFAR-10 卷积响应</h2>
        <div className="text-[10px] font-mono text-stone-500">
          accuracy {(result.finalAccuracy * 100).toFixed(2)}% · loss {result.finalLoss.toFixed(4)}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[180px_1fr] gap-5 mb-6">
        <div>
          <div className="text-[10px] text-stone-500 font-mono mb-2">RGB INPUT</div>
          <CanvasGrid
            grid={sample}
            size={32}
            rgb
            label={`${names[sample.prediction]} / true ${names[sample.label]}`}
          />
          <div className="mt-2 text-[10px] font-mono text-stone-500">
            confidence {(sample.confidence * 100).toFixed(1)}%
          </div>
        </div>

        <div>
          <div className="text-[10px] text-stone-500 font-mono mb-2">CONV1 RGB KERNELS 3×3</div>
          <div className="grid grid-cols-6 md:grid-cols-12 gap-2">
            {result.visuals.conv1Kernels.map((kernel) => (
              <CanvasGrid key={`kernel-${kernel.index}`} grid={kernel} size={kernel.size ?? 3} rgb label={`k${kernel.index}`} />
            ))}
          </div>
        </div>
      </div>

      <div>
        <div className="text-[10px] text-stone-500 font-mono mb-2">CONV1 FEATURE MAPS</div>
        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-12 gap-2">
          {result.visuals.conv1FeatureMaps.map((feature) => (
            <CanvasGrid key={`feature-${feature.index}`} grid={feature} size={feature.size ?? 32} label={`f${feature.index}`} />
          ))}
        </div>
      </div>
    </div>
  );
}
