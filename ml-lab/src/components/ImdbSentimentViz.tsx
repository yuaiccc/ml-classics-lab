import { useEffect, useMemo, useState } from "react";

interface TokenWeight {
  token: string;
  weight?: number;
  attention?: number;
}

interface SentimentSample {
  text: string;
  label: string;
  prediction: string;
  confidence: number;
  tokens: TokenWeight[];
}

interface ImdbResult {
  algorithm: string;
  accuracy: number;
  f1: number;
  topPositive?: TokenWeight[];
  topNegative?: TokenWeight[];
  samples: SentimentSample[];
}

export default function ImdbSentimentViz({ resultPath = "/results/imdb-tfidf-logreg.json" }: { resultPath?: string }) {
  const [result, setResult] = useState<ImdbResult | null>(null);
  const [step, setStep] = useState(0);

  useEffect(() => {
    fetch(resultPath)
      .then((res) => res.json())
      .then(setResult)
      .catch(() => setResult(null));
  }, [resultPath]);

  useEffect(() => {
    if (!result?.samples.length) return;
    const timer = window.setInterval(() => {
      setStep((value) => (value + 1) % result.samples.length);
    }, 1800);
    return () => window.clearInterval(timer);
  }, [result]);

  const sample = result?.samples[step % result.samples.length];
  const tokenStrip = useMemo(() => {
    if (!sample) return [];
    return sample.tokens.slice(0, 34);
  }, [sample]);

  if (!result || !sample) {
    return (
      <div className="glass rounded-xl p-6 mb-8">
        <div className="text-xs font-mono text-stone-500">Loading IMDb sentiment visualization...</div>
      </div>
    );
  }

  return (
    <div className="glass rounded-xl p-6 mb-8 overflow-hidden">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-5">
        <div>
          <h2 className="text-sm font-bold text-stone-700">IMDb 情感分类动态解释</h2>
          <p className="text-[11px] text-stone-500 mt-1">
            {result.algorithm} 的 token 信号按时间流动，高亮模型判断情感时最敏感的位置。
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
          <div className="rounded-md border border-[#cc785c]/20 bg-[#cc785c]/5 px-2 py-1">
            <div className="text-stone-500">ACC</div>
            <div className="text-[#cc785c]">{(result.accuracy * 100).toFixed(2)}%</div>
          </div>
          <div className="rounded-md border border-[#5b7b9a]/20 bg-[#5b7b9a]/5 px-2 py-1">
            <div className="text-stone-500">F1</div>
            <div className="text-[#5b7b9a]">{(result.f1 * 100).toFixed(2)}%</div>
          </div>
        </div>
      </div>

      <div className="relative rounded-lg border border-stone-200 bg-stone-100/50 p-4 mb-4">
        <div className="motion-scanline absolute inset-0 opacity-40" />
        <div className="relative flex flex-wrap gap-2 min-h-[112px]">
          {tokenStrip.map((token, index) => {
            const rawScore = token.weight ?? token.attention ?? 0;
            const magnitude = Math.min(1, token.attention ?? Math.abs(rawScore) / 8);
            const isPositive = rawScore >= 0 || token.attention !== undefined;
            return (
              <span
                key={`${token.token}-${index}-${step}`}
                className="rounded-md border px-2 py-1 text-[11px] font-mono transition-all duration-500"
                style={{
                  borderColor: isPositive ? `rgba(204, 120, 92,${0.16 + magnitude * 0.55})` : `rgba(248,113,113,${0.16 + magnitude * 0.55})`,
                  background: isPositive ? `rgba(204, 120, 92,${0.04 + magnitude * 0.22})` : `rgba(248,113,113,${0.04 + magnitude * 0.22})`,
                  color: isPositive ? "#7a8b5a" : "#c08a7a",
                  transform: `translateY(${Math.sin((index + step) * 0.75) * 2}px)`,
                  opacity: 0.58 + magnitude * 0.42,
                }}
              >
                {token.token}
              </span>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1.1fr_0.9fr] gap-4">
        <div className="rounded-lg border border-stone-200 bg-stone-100/40 p-4">
          <div className="text-[10px] font-mono text-stone-500 mb-2">SAMPLE #{step + 1}</div>
          <p className="text-xs text-stone-600 leading-relaxed mb-3">{sample.text}...</p>
          <div className="flex flex-wrap gap-2 text-[10px] font-mono">
            <span className="rounded-full border border-stone-300 px-2 py-1 text-stone-600">label: {sample.label}</span>
            <span className="rounded-full border border-[#5b7b9a]/30 px-2 py-1 text-[#5b7b9a]">pred: {sample.prediction}</span>
            <span className="rounded-full border border-[#c99a4e]/30 px-2 py-1 text-[#c99a4e]">conf: {(sample.confidence * 100).toFixed(1)}%</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {result.topPositive && result.topNegative ? (
            <>
              <TokenColumn title="positive" tokens={result.topPositive.slice(0, 7)} color="#cc785c" />
              <TokenColumn title="negative" tokens={result.topNegative.slice(0, 7)} color="#b04a3a" />
            </>
          ) : (
            <div className="col-span-2 rounded-lg border border-stone-200 bg-stone-100/40 p-3">
              <div className="text-[10px] font-mono text-stone-500 mb-2">ATTENTION TRACE</div>
              <div className="space-y-2">
                {tokenStrip.slice(0, 10).map((token, index) => (
                  <div key={`${token.token}-${index}`}>
                    <div className="flex justify-between gap-2 text-[10px] font-mono mb-1">
                      <span className="truncate text-stone-600">{token.token}</span>
                      <span className="text-[#5b7b9a]">{((token.attention ?? 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-1 rounded-full bg-stone-100 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-[#5b7b9a] transition-all duration-700"
                        style={{ width: `${Math.min(100, (token.attention ?? 0) * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function TokenColumn({ title, tokens, color }: { title: string; tokens: TokenWeight[]; color: string }) {
  return (
    <div className="rounded-lg border border-stone-200 bg-stone-100/40 p-3">
      <div className="text-[10px] font-mono text-stone-500 mb-2">{title}</div>
      <div className="space-y-2">
        {tokens.map((token) => (
          <div key={token.token}>
            <div className="flex justify-between gap-2 text-[10px] font-mono mb-1">
              <span className="truncate text-stone-600">{token.token}</span>
              <span style={{ color }}>{(token.weight ?? 0).toFixed(1)}</span>
            </div>
            <div className="h-1 rounded-full bg-stone-100 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{ width: `${Math.min(100, (Math.abs(token.weight ?? 0) / 12) * 100)}%`, background: color }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
