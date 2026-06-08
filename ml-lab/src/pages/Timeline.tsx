import { Link } from "react-router-dom";
import { experiments, ExperimentLegacy } from "@/data/experiments";
import { ArrowLeft, BookOpen, CalendarDays, Sparkles, TimerReset } from "lucide-react";

const ERA_LABELS: Record<ExperimentLegacy["history"]["era"], string> = {
  classical: "经典数学与控制",
  statistical: "统计学习成形",
  neural: "早期神经网络",
  "deep-rl": "深度强化学习",
};

const ERA_COLORS: Record<ExperimentLegacy["history"]["era"], string> = {
  classical: "#cc785c",
  statistical: "#5b7b9a",
  neural: "#b86a8a",
  "deep-rl": "#c99a4e",
};

const CATEGORY_LABELS: Record<ExperimentLegacy["category"], string> = {
  supervised: "监督学习",
  unsupervised: "无监督学习",
  deep: "深度学习",
  rl: "强化学习",
};

const sortedExperiments = [...experiments].sort((a, b) => {
  if (a.history.problemYear !== b.history.problemYear) {
    return a.history.problemYear - b.history.problemYear;
  }
  return a.history.breakthroughYear - b.history.breakthroughYear;
});

const groupedByEra = sortedExperiments.reduce<Record<string, ExperimentLegacy[]>>((groups, exp) => {
  const era = exp.history.era;
  if (!groups[era]) groups[era] = [];
  groups[era].push(exp);
  return groups;
}, {});

const eraOrder: ExperimentLegacy["history"]["era"][] = ["classical", "statistical", "neural", "deep-rl"];

function yearsBetween(exp: ExperimentLegacy) {
  return Math.max(0, exp.history.breakthroughYear - exp.history.problemYear);
}

export default function Timeline() {
  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Link to="/" className="inline-flex items-center gap-1 text-sm text-stone-500 hover:text-[#cc785c] transition-colors mb-6">
        <ArrowLeft className="w-4 h-4" /> 返回总览
      </Link>

      <header className="mb-8">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-lg bg-[rgba(201, 154, 78,0.14)] border border-[rgba(201, 154, 78,0.28)] flex items-center justify-center">
            <BookOpen className="w-5 h-5 text-[#c99a4e]" />
          </div>
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-stone-800">算法科学史时间轴</h1>
            <p className="text-sm text-stone-500 mt-1">
              按“问题提出 → 代表性突破”归档实验，观察一个想法从数学、统计到深度强化学习的迁移。
            </p>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        <div className="glass rounded-xl p-5">
          <div className="flex items-center gap-3">
            <CalendarDays className="w-5 h-5 text-[#5b7b9a]" />
            <div>
              <div className="text-2xl font-bold font-mono text-[#5b7b9a]">{sortedExperiments[0].history.problemYear}</div>
              <div className="text-xs text-stone-500">最早问题节点</div>
            </div>
          </div>
        </div>
        <div className="glass rounded-xl p-5">
          <div className="flex items-center gap-3">
            <Sparkles className="w-5 h-5 text-[#cc785c]" />
            <div>
              <div className="text-2xl font-bold font-mono text-[#cc785c]">{experiments.length}</div>
              <div className="text-xs text-stone-500">已归档实验</div>
            </div>
          </div>
        </div>
        <div className="glass rounded-xl p-5">
          <div className="flex items-center gap-3">
            <TimerReset className="w-5 h-5 text-[#c99a4e]" />
            <div>
              <div className="text-2xl font-bold font-mono text-[#c99a4e]">
                {Math.max(...experiments.map(yearsBetween))}y
              </div>
              <div className="text-xs text-stone-500">最长问题到突破间隔</div>
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-8">
        {eraOrder.map((era) => {
          const group = groupedByEra[era];
          if (!group?.length) return null;
          const color = ERA_COLORS[era];

          return (
            <section key={era}>
              <div className="flex items-center gap-3 mb-4">
                <span className="text-sm font-bold font-mono" style={{ color }}>
                  {ERA_LABELS[era]}
                </span>
                <span className="text-[10px] font-mono text-stone-500">{group.length} 个实验</span>
                <div className="flex-1 h-px bg-stone-200/60" />
              </div>

              <div className="relative pl-5 sm:pl-8">
                <div className="absolute left-1.5 sm:left-3 top-0 bottom-0 w-px bg-stone-200" />
                <div className="space-y-4">
                  {group.map((exp) => {
                    const gap = yearsBetween(exp);
                    return (
                      <Link
                        key={exp.id}
                        to={`/experiment/${exp.id}`}
                        className="block group"
                      >
                        <article className="relative glass glass-hover rounded-xl p-5 transition-all duration-300">
                          <div
                            className="absolute -left-[22px] sm:-left-[30px] top-6 w-3 h-3 rounded-full border-2 border-[#f0eee6]"
                            style={{ background: color, boxShadow: `0 0 14px ${color}` }}
                          />
                          <div className="grid grid-cols-1 lg:grid-cols-[170px_1fr] gap-4">
                            <div>
                              <div className="text-2xl font-bold font-mono" style={{ color }}>
                                {exp.history.problemYear}
                              </div>
                              <div className="text-[10px] font-mono text-stone-500 mt-1">
                                → {exp.history.breakthroughYear}
                                {gap > 0 ? ` · ${gap}y` : ""}
                              </div>
                            </div>

                            <div>
                              <div className="flex flex-wrap items-center gap-2 mb-2">
                                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full border border-stone-300 text-stone-500">
                                  {CATEGORY_LABELS[exp.category]}
                                </span>
                                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full border border-[#5b7b9a]/25 text-[#5b7b9a] bg-[#5b7b9a]/5">
                                  {exp.algorithm}
                                </span>
                              </div>
                              <h2 className="text-base font-bold text-stone-800 mb-2">
                                {exp.env}
                              </h2>
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
                                <div className="rounded-lg border border-stone-200 bg-stone-100/30 p-3">
                                  <div className="text-[10px] text-stone-500 font-mono mb-1">PROBLEM</div>
                                  <div className="text-xs text-stone-700">{exp.history.problemLabel}</div>
                                </div>
                                <div className="rounded-lg border border-stone-200 bg-stone-100/30 p-3">
                                  <div className="text-[10px] text-stone-500 font-mono mb-1">BREAKTHROUGH</div>
                                  <div className="text-xs text-stone-700">{exp.history.breakthroughLabel}</div>
                                </div>
                              </div>
                              <p className="text-xs text-stone-500 leading-relaxed">
                                {exp.history.note}
                              </p>
                            </div>
                          </div>
                        </article>
                      </Link>
                    );
                  })}
                </div>
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}
