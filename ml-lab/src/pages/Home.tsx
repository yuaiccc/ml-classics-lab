import { useState, useMemo } from "react";
import { Link } from "react-router-dom";
import { experiments, ExperimentLegacy } from "@/data/experiments";
import { classicProblems, ClassicProblem } from "@/data/classicProblems";
import { FlaskConical, TrendingUp, Trophy, ArrowRight, ChevronDown, ChevronRight, BookOpen, Clock, Layers, Zap, ClipboardList } from "lucide-react";

const CATEGORIES = ["all", "supervised", "unsupervised", "deep", "rl"] as const;

const CATEGORY_LABELS: Record<string, string> = {
  all: "全部",
  supervised: "监督学习",
  unsupervised: "无监督学习",
  deep: "深度学习",
  rl: "强化学习",
};

const CATEGORY_ICONS: Record<string, string> = {
  supervised: "📊",
  unsupervised: "🔍",
  deep: "🧠",
  rl: "🎮",
};

const CATEGORY_COLORS: Record<string, string> = {
  supervised: "#cc785c",
  unsupervised: "#5b7b9a",
  deep: "#7a8b5a",
  rl: "#c99a4e",
};

type CategoryKey = (typeof CATEGORIES)[number];

type ProblemGroup = {
  key: string;
  env: string;
  category: ExperimentLegacy["category"];
  experiments: ExperimentLegacy[];
  lead: ExperimentLegacy;
};

const STATUS_MAP: Record<string, { label: string; cls: string }> = {
  success: { label: "成功", cls: "status-success" },
  failed: { label: "失败", cls: "status-failed" },
  partial: { label: "部分", cls: "status-partial" },
};

const ALG_BADGE_MAP: Record<string, string> = {
  PPO: "badge-ppo",
  DQN: "badge-dqn",
  SAC: "badge-sac",
  "DQN + Reward Shaping": "badge-shaped",
  MLP: "badge-sac",
  "LeNet-style CNN": "badge-shaped",
  "Small CNN": "badge-ppo",
  "Q-Learning": "badge-ppo",
  "TF-IDF + Logistic Regression": "badge-shaped",
  "DistilBERT fine-tune": "badge-sac",
  "Random Policy": "badge-dqn",
};

function createProblemGroups(items: ExperimentLegacy[]): ProblemGroup[] {
  const groups = new Map<string, ProblemGroup>();

  for (const exp of items) {
    const key = `${exp.category}:${exp.env}`;
    const current = groups.get(key);
    if (!current) {
      groups.set(key, {
        key,
        env: exp.env,
        category: exp.category,
        experiments: [exp],
        lead: exp,
      });
      continue;
    }

    current.experiments.push(exp);
    if (exp.status === "success" && current.lead.status !== "success") {
      current.lead = exp;
    }
  }

  return [...groups.values()].map((group) => ({
    ...group,
    experiments: [...group.experiments].sort((a, b) => {
      const statusRank = { success: 0, partial: 1, failed: 2 };
      return statusRank[a.status] - statusRank[b.status];
    }),
  }));
}

function ProblemCard({ group }: { group: ProblemGroup }) {
  const color = CATEGORY_COLORS[group.category] || "#cc785c";
  const successCount = group.experiments.filter((exp) => exp.status === "success").length;
  const totalSteps = group.experiments.reduce((sum, exp) => sum + exp.totalSteps, 0);

  return (
    <article className="glass glass-hover rounded-xl p-5 transition-all duration-300 h-full flex flex-col">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full border border-stone-300 text-stone-500">
              {CATEGORY_LABELS[group.category]}
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full border border-[#cc785c]/25 text-[#cc785c] bg-[#cc785c]/5">
              {successCount}/{group.experiments.length} 成功
            </span>
          </div>
          <h3 className="font-serif text-lg text-stone-900 mb-1">
            {group.env}
          </h3>
        </div>
        <div className="text-right shrink-0">
          <div className="text-2xl font-bold font-mono" style={{ color }}>
            {group.experiments.length}
          </div>
          <div className="text-[10px] text-stone-500 font-mono">methods</div>
        </div>
      </div>

      <p className="text-xs text-stone-500 mb-4 line-clamp-2 flex-grow">
        {group.lead.abstract || group.lead.description}
      </p>

      <div className="space-y-2 mb-4">
        {group.experiments.map((exp) => {
          const status = STATUS_MAP[exp.status];
          const badge = ALG_BADGE_MAP[exp.algorithm] || "badge-ppo";
          return (
            <Link
              key={exp.id}
              to={`/experiment/${exp.id}`}
              className="block rounded-lg border border-stone-200 bg-stone-100/30 px-3 py-2 hover:border-[#5b7b9a]/30 transition-colors group/method"
            >
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2 mb-1">
                    <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${badge}`}>
                      {exp.algorithm}
                    </span>
                    <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${status.cls}`}>
                      {status.label}
                    </span>
                  </div>
                  <div className="text-[11px] text-stone-500 line-clamp-1">
                    {exp.description}
                  </div>
                </div>
                <div className="text-right shrink-0">
                  <div className="flex items-center justify-end gap-1 text-xs font-mono text-stone-700">
                    <Zap className="w-3 h-3 text-stone-500" />
                    {exp.finalReward}
                  </div>
                  <ArrowRight className="w-3 h-3 text-stone-400 group-hover/method:text-[#cc785c] transition-colors ml-auto mt-1" />
                </div>
              </div>
            </Link>
          );
        })}
      </div>

      <div className="flex items-center justify-between text-xs text-stone-500 border-t border-stone-200/50 pt-3">
        <span className="flex items-center gap-1">
          <Layers className="w-3 h-3" />
          <span className="font-mono">{totalSteps.toLocaleString()} steps</span>
        </span>
        <span className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          <span className="font-mono">{group.experiments.length === 1 ? `${group.lead.trainingTime}s` : "多次实验"}</span>
        </span>
      </div>
    </article>
  );
}

function ClassicProblemCard({ problem }: { problem: ClassicProblem }) {
  const color = CATEGORY_COLORS[problem.category] || "#cc785c";

  return (
    <article className="glass rounded-xl p-5 h-full border-dashed border-stone-300/70">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full border border-stone-300 text-stone-500">
              {CATEGORY_LABELS[problem.category]}
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full border border-[#c99a4e]/30 text-[#c99a4e] bg-[#c99a4e]/10">
              待复现
            </span>
          </div>
          <h3 className="font-serif text-lg text-stone-900 mb-1">
            {problem.title}
          </h3>
          <div className="text-[10px] font-mono text-stone-500">
            {problem.benchmark} · {problem.domain}
          </div>
        </div>
        <ClipboardList className="w-5 h-5 shrink-0" style={{ color }} />
      </div>

      <p className="text-xs text-stone-500 leading-relaxed mb-4">
        {problem.whyClassic}
      </p>

      <div className="grid grid-cols-1 gap-3 mb-4">
        <div className="rounded-lg border border-stone-200 bg-stone-100/30 p-3">
          <div className="text-[10px] text-stone-500 font-mono mb-1">COMMON METHODS</div>
          <div className="flex flex-wrap gap-1.5">
            {problem.commonMethods.map((method) => (
              <span key={method} className="rounded-md bg-stone-100 px-2 py-1 text-[10px] text-stone-600 font-mono">
                {method}
              </span>
            ))}
          </div>
        </div>
        <div className="rounded-lg border border-stone-200 bg-stone-100/30 p-3">
          <div className="text-[10px] text-stone-500 font-mono mb-1">PLANNED RUNS</div>
          <div className="text-xs text-stone-600 leading-relaxed">
            {problem.plannedRuns.join(" / ")}
          </div>
        </div>
      </div>

      <div className="border-t border-stone-200/50 pt-3 text-[11px] text-stone-500 leading-relaxed">
        <span className="font-mono text-[#5b7b9a]">{problem.history.problemYear}</span>
        {" "}
        {problem.history.problemLabel}
        {" → "}
        <span className="font-mono text-[#c99a4e]">{problem.history.breakthroughYear}</span>
        {" "}
        {problem.history.breakthroughLabel}
      </div>
    </article>
  );
}

export default function Home() {
  const [activeCategory, setActiveCategory] = useState<CategoryKey>("all");
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const successCount = experiments.filter((e) => e.status === "success").length;
  const bestExp = experiments.reduce((a, b) => (a.finalReward > b.finalReward ? a : b));

  const filteredExperiments = useMemo(() => {
    if (activeCategory === "all") return experiments;
    return experiments.filter((e) => e.category === activeCategory);
  }, [activeCategory]);

  const problemGroups = useMemo(() => createProblemGroups(filteredExperiments), [filteredExperiments]);
  const filteredClassicProblems = useMemo(() => {
    if (activeCategory === "all") return classicProblems;
    return classicProblems.filter((problem) => problem.category === activeCategory);
  }, [activeCategory]);

  const groupedProblems = useMemo(() => {
    const groups: Record<string, ProblemGroup[]> = {};
    for (const group of problemGroups) {
      if (!groups[group.category]) groups[group.category] = [];
      groups[group.category].push(group);
    }
    return groups;
  }, [problemGroups]);

  const categoryOrder = ["supervised", "unsupervised", "deep", "rl"];

  const toggleCollapse = (cat: string) => {
    setCollapsed((prev) => ({ ...prev, [cat]: !prev[cat] }));
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <header className="pt-10 pb-14 sm:pt-16 sm:pb-20 border-b border-stone-200/70 mb-12">
        <div className="flex items-center gap-2 mb-6">
          <FlaskConical className="w-4 h-4 text-[#cc785c]" />
          <span className="text-xs font-mono tracking-widest uppercase text-stone-500">ML Lab</span>
        </div>
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-8">
          <div className="max-w-3xl">
            <h1 className="font-serif text-[2.75rem] sm:text-6xl leading-[1.05] tracking-tight text-stone-900">
              看见算法<br className="hidden sm:block" />学习的过程
            </h1>
            <p className="mt-6 text-lg text-stone-600 leading-relaxed max-w-xl">
              一个机器学习算法可视化实验室。每个经典算法都配一段逐步收敛的过程动画，
              覆盖监督、无监督、深度学习与强化学习四大家族。
            </p>
          </div>
          <Link
            to="/timeline"
            className="inline-flex items-center gap-2 rounded-full border border-stone-300 px-4 py-2 text-sm text-stone-700 hover:border-[#cc785c] hover:text-[#cc785c] transition-colors self-start shrink-0"
          >
            <BookOpen className="w-4 h-4" />
            科学史时间轴
          </Link>
        </div>
      </header>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        <div className="glass rounded-xl p-5 animate-fade-in-up">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-[rgba(91, 123, 154,0.15)] flex items-center justify-center">
              <FlaskConical className="w-4 h-4 text-[#5b7b9a]" />
            </div>
            <div>
              <div className="text-2xl font-bold font-mono text-[#5b7b9a] neon-text-cyan">{createProblemGroups(experiments).length}</div>
              <div className="text-xs text-stone-500">问题总数</div>
            </div>
          </div>
        </div>
        <div className="glass rounded-xl p-5 animate-fade-in-up" style={{ animationDelay: "0.1s" }}>
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-[rgba(204, 120, 92,0.15)] flex items-center justify-center">
              <TrendingUp className="w-4 h-4 text-[#cc785c]" />
            </div>
            <div>
              <div className="text-2xl font-bold font-mono text-[#cc785c] neon-text">
                {((successCount / experiments.length) * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-stone-500">成功率</div>
            </div>
          </div>
        </div>
        <div className="glass rounded-xl p-5 animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-[rgba(201, 154, 78,0.15)] flex items-center justify-center">
              <Trophy className="w-4 h-4 text-[#c99a4e]" />
            </div>
            <div>
              <div className="text-sm font-bold text-[#c99a4e]">{bestExp.algorithm}</div>
              <div className="text-xs text-stone-500">最佳: {bestExp.env}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 mb-6">
        <span className="text-xs text-stone-500 mr-2">分类</span>
        {CATEGORIES.map((cat) => (
          <button
            key={cat}
            onClick={() => setActiveCategory(cat)}
            className={`px-3 py-1 rounded-full text-xs font-mono border transition-all ${
              activeCategory === cat
                ? "border-[#cc785c]/40 text-[#cc785c] bg-[#cc785c]/10"
                : "border-stone-300 text-stone-500 hover:border-slate-500"
            }`}
          >
            {CATEGORY_LABELS[cat]}
          </button>
        ))}
      </div>

      {activeCategory === "all" ? (
        categoryOrder.map((cat) => {
          const group = groupedProblems[cat];
          if (!group || group.length === 0) return null;
          const isCollapsed = collapsed[cat];
          const color = CATEGORY_COLORS[cat] || "#cc785c";
          const experimentCount = group.reduce((sum, item) => sum + item.experiments.length, 0);
          const successInGroup = group.reduce((sum, item) => sum + item.experiments.filter((exp) => exp.status === "success").length, 0);

          return (
            <div key={cat} className="mb-8">
              <button
                onClick={() => toggleCollapse(cat)}
                className="w-full flex items-center gap-3 mb-4 group"
              >
                {isCollapsed ? (
                  <ChevronRight className="w-4 h-4 text-stone-500 group-hover:text-stone-700 transition-colors" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-stone-500 group-hover:text-stone-700 transition-colors" />
                )}
                <span className="text-sm">{CATEGORY_ICONS[cat]}</span>
                <span className="font-serif text-xl" style={{ color }}>
                  {CATEGORY_LABELS[cat]}
                </span>
                <span className="text-[10px] font-mono text-stone-500">
                  {group.length} 个问题 · {experimentCount} 个方法 · {successInGroup}/{experimentCount} 成功
                </span>
                <div className="flex-1 h-px bg-stone-200/50" />
              </button>

              {!isCollapsed && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                  {group.map((exp, i) => (
                    <div
                      key={exp.key}
                      className="block animate-fade-in-up"
                      style={{ animationDelay: `${i * 0.06}s` }}
                    >
                      <ProblemCard group={exp} />
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {problemGroups.map((group, i) => (
            <div
              key={group.key}
              className="block animate-fade-in-up"
              style={{ animationDelay: `${i * 0.06}s` }}
            >
              <ProblemCard group={group} />
            </div>
          ))}
        </div>
      )}

      {filteredClassicProblems.length > 0 && (
        <section className="mt-12">
          <div className="flex items-center gap-3 mb-4">
            <ClipboardList className="w-4 h-4 text-[#c99a4e]" />
            <span className="text-sm font-bold font-mono text-[#c99a4e]">
              待复现经典问题
            </span>
            <span className="text-[10px] font-mono text-stone-500">
              {filteredClassicProblems.length} 个候选 · 不计入成功率
            </span>
            <div className="flex-1 h-px bg-stone-200/50" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {filteredClassicProblems.map((problem) => (
              <ClassicProblemCard key={problem.id} problem={problem} />
            ))}
          </div>
        </section>
      )}

      <footer className="mt-16 pt-6 border-t border-stone-200/50 text-center">
        <p className="text-xs text-stone-500">
          ML Lab · 追加实验只需在{" "}
          <code className="font-mono text-[#cc785c]/60">src/data/experiments.ts</code>{" "}
          数组末尾 push 新对象
          <ArrowRight className="w-3 h-3 inline mx-1" />自动渲染
        </p>
      </footer>
    </div>
  );
}
