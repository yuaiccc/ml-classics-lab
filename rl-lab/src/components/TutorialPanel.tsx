// 小白教程面板：把一个算法渲染成“解决什么问题 / 直觉 / 看点 / 概念 / 动手试试”五段式小课。
import { Tutorial } from "@/player/types";
import { Target, Lightbulb, Eye, BookOpen, MousePointerClick } from "lucide-react";

export default function TutorialPanel({ tutorial }: { tutorial: Tutorial }) {
  return (
    <div className="glass rounded-xl p-5 flex flex-col gap-5">
      {/* 解决什么问题 */}
      <section>
        <div className="flex items-center gap-1.5 text-xs text-[#00e5ff] mb-1.5 font-semibold">
          <Target className="w-3.5 h-3.5" /> 解决什么问题
        </div>
        <p className="text-sm text-slate-300 leading-relaxed">{tutorial.problem}</p>
      </section>

      {/* 直觉理解 */}
      <section>
        <div className="flex items-center gap-1.5 text-xs text-[#ffab40] mb-1.5 font-semibold">
          <Lightbulb className="w-3.5 h-3.5" /> 直觉理解
        </div>
        <p className="text-sm text-slate-400 leading-relaxed">{tutorial.intuition}</p>
      </section>

      {/* 看动画注意什么 */}
      <section>
        <div className="flex items-center gap-1.5 text-xs text-[#00ff88] mb-2 font-semibold">
          <Eye className="w-3.5 h-3.5" /> 看动画注意什么
        </div>
        <ul className="flex flex-col gap-1.5">
          {tutorial.watch.map((w, i) => (
            <li key={i} className="text-sm text-slate-400 leading-relaxed flex gap-2">
              <span className="text-[#00ff88]/60 font-mono shrink-0">{i + 1}.</span>
              <span>{w}</span>
            </li>
          ))}
        </ul>
      </section>

      {/* 关键概念 */}
      <section>
        <div className="flex items-center gap-1.5 text-xs text-[#b388ff] mb-2 font-semibold">
          <BookOpen className="w-3.5 h-3.5" /> 关键概念
        </div>
        <dl className="flex flex-col gap-2">
          {tutorial.concepts.map((c) => (
            <div key={c.term} className="text-sm">
              <dt className="text-slate-200 font-medium inline">{c.term}</dt>
              <dd className="text-slate-500 inline"> —— {c.explain}</dd>
            </div>
          ))}
        </dl>
      </section>

      {/* 动手试试 */}
      {tutorial.tryThis && (
        <section className="rounded-lg bg-[rgba(0,229,255,0.06)] border border-[rgba(0,229,255,0.15)] p-3">
          <div className="flex items-center gap-1.5 text-xs text-[#00e5ff] mb-1 font-semibold">
            <MousePointerClick className="w-3.5 h-3.5" /> 动手试试
          </div>
          <p className="text-sm text-slate-400 leading-relaxed">{tutorial.tryThis}</p>
        </section>
      )}
    </div>
  );
}
