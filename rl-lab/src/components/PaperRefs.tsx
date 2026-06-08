import { PaperRef } from "@/data/papers";

export default function PaperRefs({ refs }: { refs: PaperRef[] }) {
  return (
    <div className="mt-4 rounded-lg border border-slate-700/40 bg-slate-800/30 p-3">
      <div className="text-xs font-semibold text-slate-300 mb-2">📄 参考文献</div>
      <ul className="flex flex-col gap-1.5">
        {refs.map((r) => (
          <li key={r.url} className="text-[12px] leading-snug">
            <a
              href={r.url}
              target="_blank"
              rel="noreferrer"
              className="text-sky-400 hover:underline font-mono"
            >
              {r.name}
            </a>
            {r.note && <span className="ml-2 text-amber-400/80">· {r.note}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}
