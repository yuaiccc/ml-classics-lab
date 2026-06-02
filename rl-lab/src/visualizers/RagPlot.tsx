// rag 家族：问题 + 按相似度排序逐条揭示的知识库（Top-K 高亮）+ 生成答案。
import { RagState } from "@/player/types";
import { Search, FileText, Sparkles } from "lucide-react";

export default function RagPlot({ state }: { state: RagState }) {
  const { query, docs, shown, topK, answer } = state;
  const maxScore = Math.max(0.01, ...docs.map((d) => d.score));

  return (
    <div className="rounded-xl bg-[rgba(10,14,23,0.6)] p-5 flex flex-col gap-3">
      {/* 问题 */}
      <div className="rounded-lg p-3 flex gap-2 items-start bg-[rgba(0,229,255,0.08)] border border-[#00e5ff]/30">
        <Search className="w-4 h-4 mt-0.5 text-[#00e5ff] shrink-0" />
        <div>
          <div className="text-[11px] text-[#00e5ff] font-semibold mb-0.5">问题</div>
          <div className="text-sm text-slate-200">{query}</div>
        </div>
      </div>

      {/* 检索：按相似度排序逐条揭示 */}
      <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
        <FileText className="w-3.5 h-3.5" /> 知识库检索（按相似度排序，绿色=选中的 Top-{topK}）
      </div>
      <div className="flex flex-col gap-1.5">
        {docs.slice(0, shown).map((d, i) => {
          const top = i < topK;
          return (
            <div
              key={i}
              className={`rounded-md px-2.5 py-1.5 flex items-center gap-3 border ${
                top ? "bg-[rgba(0,255,136,0.08)] border-[#00ff88]/30" : "bg-white/[0.02] border-slate-800"
              }`}
            >
              <span className={`font-mono text-xs w-12 shrink-0 ${top ? "text-[#00ff88]" : "text-slate-500"}`}>
                {d.score.toFixed(3)}
              </span>
              <div className="flex-1 min-w-0">
                <div className="h-1.5 rounded-full mb-1" style={{ width: `${(d.score / maxScore) * 100}%`, background: top ? "#00ff88" : "#475569" }} />
                <div className="text-xs text-slate-300 truncate">{d.text}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* 生成答案 */}
      {answer && (
        <div className="rounded-lg p-3 flex gap-2 items-start bg-[rgba(179,136,255,0.1)] border border-[#b388ff]/40 mt-1 animate-fade-in-up">
          <Sparkles className="w-4 h-4 mt-0.5 text-[#b388ff] shrink-0" />
          <div>
            <div className="text-[11px] text-[#b388ff] font-semibold mb-0.5">大模型基于检索资料生成的回答</div>
            <div className="text-sm text-slate-200 leading-relaxed">{answer}</div>
          </div>
        </div>
      )}
    </div>
  );
}
