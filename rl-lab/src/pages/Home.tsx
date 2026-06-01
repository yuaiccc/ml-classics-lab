import { Link } from "react-router-dom";
import { useLabStore } from "@/hooks/useLabStore";
import { FlaskConical, TrendingUp, Trophy, ArrowRight, Sparkles } from "lucide-react";
import ExperimentCard from "@/components/ExperimentCard";

const ALGORITHMS = ["all", "PPO", "DQN", "SAC"];

export default function Home() {
  const { filter, setFilter, filteredExperiments, experiments } = useLabStore();
  const filtered = filteredExperiments();
  const successCount = experiments.filter((e) => e.status === "success").length;
  const bestExp = experiments.reduce((a, b) =>
    a.finalReward > b.finalReward ? a : b
  );

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <header className="mb-10">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-lg bg-[rgba(0,255,136,0.15)] border border-[rgba(0,255,136,0.3)] flex items-center justify-center">
            <FlaskConical className="w-5 h-5 text-[#00ff88]" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight">
            <span className="neon-text text-[#00ff88]">Tianshou</span>{" "}
            <span className="text-slate-300">RL Lab</span>
          </h1>
        </div>
        <p className="text-slate-500 text-sm ml-[52px]">
          强化学习实验可视化看板 · 基于 Tianshou 框架
        </p>
        <Link
          to="/lab"
          className="inline-flex items-center gap-1.5 mt-4 ml-[52px] px-3 py-1.5 rounded-lg text-xs border border-[#b388ff]/40 text-[#b388ff] hover:bg-[rgba(179,136,255,0.1)] transition-all"
        >
          <Sparkles className="w-3.5 h-3.5" /> 算法过程动画 · 实验台（回归 / 分类 / 聚类 / 降维 / 深度学习 / 强化学习）
          <ArrowRight className="w-3 h-3" />
        </Link>
      </header>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        <div className="glass rounded-xl p-5 animate-fade-in-up">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-[rgba(0,229,255,0.15)] flex items-center justify-center">
              <FlaskConical className="w-4 h-4 text-[#00e5ff]" />
            </div>
            <div>
              <div className="text-2xl font-bold font-mono text-[#00e5ff] neon-text-cyan">
                {experiments.length}
              </div>
              <div className="text-xs text-slate-500">实验总数</div>
            </div>
          </div>
        </div>
        <div
          className="glass rounded-xl p-5 animate-fade-in-up"
          style={{ animationDelay: "0.1s" }}
        >
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-[rgba(0,255,136,0.15)] flex items-center justify-center">
              <TrendingUp className="w-4 h-4 text-[#00ff88]" />
            </div>
            <div>
              <div className="text-2xl font-bold font-mono text-[#00ff88] neon-text">
                {((successCount / experiments.length) * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-slate-500">成功率</div>
            </div>
          </div>
        </div>
        <div
          className="glass rounded-xl p-5 animate-fade-in-up"
          style={{ animationDelay: "0.2s" }}
        >
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-[rgba(255,171,64,0.15)] flex items-center justify-center">
              <Trophy className="w-4 h-4 text-[#ffab40]" />
            </div>
            <div>
              <div className="text-sm font-bold text-[#ffab40]">
                {bestExp.algorithm}
              </div>
              <div className="text-xs text-slate-500">
                最佳: {bestExp.env} ({bestExp.finalReward})
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 mb-6">
        <span className="text-xs text-slate-500 mr-2">筛选算法</span>
        {ALGORITHMS.map((alg) => (
          <button
            key={alg}
            onClick={() => setFilter(alg)}
            className={`px-3 py-1 rounded-full text-xs font-mono border transition-all duration-200 ${
              filter === alg
                ? "filter-btn-active"
                : "border-slate-700 text-slate-500 hover:border-slate-500"
            }`}
          >
            {alg === "all" ? "全部" : alg}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {filtered.map((exp, i) => (
          <Link
            key={exp.id}
            to={`/experiment/${exp.id}`}
            className="block animate-fade-in-up"
            style={{ animationDelay: `${i * 0.08}s` }}
          >
            <ExperimentCard experiment={exp} />
          </Link>
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-20 text-slate-600">
          <FlaskConical className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p>暂无匹配的实验</p>
        </div>
      )}

      <footer className="mt-16 pt-6 border-t border-slate-800/50 text-center">
        <p className="text-xs text-slate-600">
          Tianshou RL Lab · 追加实验只需在{" "}
          <code className="font-mono text-[#00ff88]/60">
            src/data/experiments.ts
          </code>{" "}
          数组末尾 push 新对象
          <ArrowRight className="w-3 h-3 inline mx-1" />
          自动渲染
        </p>
      </footer>
    </div>
  );
}
