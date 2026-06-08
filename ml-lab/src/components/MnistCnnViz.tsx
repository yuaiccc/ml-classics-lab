import { useEffect, useRef, useState } from "react";

interface PixelGrid {
  index?: number;
  size?: number;
  pixels: number[];
}

interface MnistCnnResult {
  finalAccuracy: number;
  finalLoss: number;
  visuals: {
    sample: PixelGrid & {
      label: number;
      prediction: number;
      confidence: number;
    };
    conv1Kernels: PixelGrid[];
    conv1FeatureMaps: PixelGrid[];
  };
}

function PixelCanvas({ grid, size, label }: { grid: PixelGrid; size: number; label: string }) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(size, size);
    for (let i = 0; i < grid.pixels.length; i++) {
      const value = grid.pixels[i];
      const offset = i * 4;
      imageData.data[offset] = value;
      imageData.data[offset + 1] = value;
      imageData.data[offset + 2] = value;
      imageData.data[offset + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }, [grid, size]);

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

export default function MnistCnnViz() {
  const [result, setResult] = useState<MnistCnnResult | null>(null);

  useEffect(() => {
    let mounted = true;
    fetch("/results/mnist-cnn.json")
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
        <div className="text-sm text-stone-500">MNIST CNN 卷积可视化数据尚未加载。</div>
      </div>
    );
  }

  const sample = result.visuals.sample;

  return (
    <div className="glass rounded-xl p-6 mb-8">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-5">
        <h2 className="text-sm font-bold text-stone-600">CNN 卷积核与特征图</h2>
        <div className="text-[10px] font-mono text-stone-500">
          accuracy {(result.finalAccuracy * 100).toFixed(2)}% · loss {result.finalLoss.toFixed(4)}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[160px_1fr] gap-5 mb-6">
        <div>
          <div className="text-[10px] text-stone-500 font-mono mb-2">INPUT</div>
          <PixelCanvas grid={sample} size={28} label={`pred ${sample.prediction} / true ${sample.label}`} />
          <div className="mt-2 text-[10px] font-mono text-stone-500">
            confidence {(sample.confidence * 100).toFixed(1)}%
          </div>
        </div>

        <div>
          <div className="text-[10px] text-stone-500 font-mono mb-2">CONV1 KERNELS 5×5</div>
          <div className="grid grid-cols-6 md:grid-cols-12 gap-2">
            {result.visuals.conv1Kernels.map((kernel) => (
              <PixelCanvas
                key={`kernel-${kernel.index}`}
                grid={kernel}
                size={kernel.size ?? 5}
                label={`k${kernel.index}`}
              />
            ))}
          </div>
        </div>
      </div>

      <div>
        <div className="text-[10px] text-stone-500 font-mono mb-2">CONV1 FEATURE MAPS</div>
        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-12 gap-2">
          {result.visuals.conv1FeatureMaps.map((feature) => (
            <PixelCanvas
              key={`feature-${feature.index}`}
              grid={feature}
              size={feature.size ?? 28}
              label={`f${feature.index}`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
