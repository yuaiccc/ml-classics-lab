import { useEffect, useRef, useState } from "react";

interface MnistSample {
  index: number;
  label: number;
  prediction: number;
  confidence: number;
  pixels: number[];
}

interface MnistResult {
  finalAccuracy: number;
  finalLoss: number;
  samples: {
    correct: MnistSample[];
    mistakes: MnistSample[];
  };
}

function DigitCanvas({ sample }: { sample: MnistSample }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const correct = sample.label === sample.prediction;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(28, 28);
    for (let i = 0; i < sample.pixels.length; i++) {
      const value = sample.pixels[i];
      const offset = i * 4;
      imageData.data[offset] = value;
      imageData.data[offset + 1] = value;
      imageData.data[offset + 2] = value;
      imageData.data[offset + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }, [sample]);

  return (
    <div className="rounded-lg border border-stone-200 bg-stone-100/40 p-2">
      <canvas
        ref={ref}
        width={28}
        height={28}
        className="w-full aspect-square rounded-md bg-black image-rendering-pixelated"
        aria-label={`MNIST sample ${sample.index}`}
      />
      <div className="mt-2 flex items-center justify-between gap-2 text-[10px] font-mono">
        <span className={correct ? "text-[#cc785c]" : "text-[#b04a3a]"}>
          {sample.prediction}
        </span>
        <span className="text-stone-500">true {sample.label}</span>
      </div>
      <div className="mt-1 h-1 rounded-full bg-stone-200 overflow-hidden">
        <div
          className={correct ? "h-full bg-[#cc785c]" : "h-full bg-[#b04a3a]"}
          style={{ width: `${Math.round(sample.confidence * 100)}%` }}
        />
      </div>
    </div>
  );
}

export default function MnistViz() {
  const [result, setResult] = useState<MnistResult | null>(null);

  useEffect(() => {
    let mounted = true;
    fetch("/results/mnist-mlp.json")
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
        <div className="text-sm text-stone-500">MNIST 可视化数据尚未加载。</div>
      </div>
    );
  }

  return (
    <div className="glass rounded-xl p-6 mb-8">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-5">
        <h2 className="text-sm font-bold text-stone-600">手写数字预测样本</h2>
        <div className="text-[10px] font-mono text-stone-500">
          accuracy {(result.finalAccuracy * 100).toFixed(2)}% · loss {result.finalLoss.toFixed(4)}
        </div>
      </div>

      <div className="mb-5">
        <div className="text-[10px] text-stone-500 font-mono mb-2">CORRECT SAMPLES</div>
        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-12 gap-2">
          {result.samples.correct.map((sample) => (
            <DigitCanvas key={`correct-${sample.index}`} sample={sample} />
          ))}
        </div>
      </div>

      {result.samples.mistakes.length > 0 && (
        <div>
          <div className="text-[10px] text-stone-500 font-mono mb-2">MISTAKES</div>
          <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-12 gap-2">
            {result.samples.mistakes.map((sample) => (
              <DigitCanvas key={`mistake-${sample.index}`} sample={sample} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
