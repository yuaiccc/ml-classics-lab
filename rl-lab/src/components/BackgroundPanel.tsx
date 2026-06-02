// 「现实意义」面板：解决什么实际问题 + 真实应用场景 + 可选配图。
import { Background } from "@/data/backgrounds";
import { Globe } from "lucide-react";

export default function BackgroundPanel({ bg }: { bg: Background }) {
  return (
    <div className="glass rounded-xl p-5">
      <div className="flex items-center gap-1.5 text-xs text-[#00e5ff] mb-2 font-semibold">
        <Globe className="w-3.5 h-3.5" /> 现实意义 · 解决什么实际问题
      </div>
      <p className="text-sm text-slate-300 leading-relaxed">{bg.realWorld}</p>

      <div className="flex flex-wrap gap-2 mt-3">
        {bg.uses.map((u) => (
          <span
            key={u}
            className="text-xs px-2 py-1 rounded-md bg-[rgba(0,255,136,0.08)] text-[#00ff88]/90 border border-[#00ff88]/20"
          >
            {u}
          </span>
        ))}
      </div>

      {bg.images && (
        <div className="flex flex-wrap gap-3 mt-4">
          {bg.images.map((img) => (
            <figure key={img.src} className="flex flex-col items-center gap-1">
              <img
                src={img.src}
                alt={img.caption}
                loading="lazy"
                className="w-32 h-32 object-cover rounded-lg border border-slate-700"
              />
              <figcaption className="text-[11px] text-slate-500 font-mono">{img.caption}</figcaption>
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}
