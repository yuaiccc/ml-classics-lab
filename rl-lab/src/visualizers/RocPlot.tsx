// roc 家族：分数分布 + 阈值线 + ROC 曲线 + 混淆矩阵 + 精确率/召回率。
import { RocState } from "@/player/types";

const C_POS = "#00ff88";
const C_NEG = "#ffab40";

export default function RocPlot({ state }: { state: RocState }) {
  const { threshold, pos, neg, roc, current, confusion, precision, recall, f1, auc } = state;
  const all = [...pos, ...neg];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const SW = 540;
  const sx = (v: number) => 30 + ((v - lo) / (hi - lo)) * (SW - 60);

  return (
    <div className="rounded-xl bg-[rgba(10,14,23,0.6)] p-4 flex flex-col gap-4">
      {/* 分数分布 + 阈值线 */}
      <div>
        <div className="text-xs text-slate-500 mb-1">样本分数（绿=正类 / 橙=负类）· 竖线=判定阈值</div>
        <svg viewBox={`0 0 ${SW} 110`} className="w-full">
          <line x1={30} y1={55} x2={SW - 30} y2={55} stroke="#334155" strokeWidth={1} />
          {pos.map((v, i) => (
            <circle key={`p${i}`} cx={sx(v)} cy={20 + (i % 6) * 5} r={2.5} fill={C_POS} opacity={0.7} />
          ))}
          {neg.map((v, i) => (
            <circle key={`n${i}`} cx={sx(v)} cy={70 + (i % 6) * 5} r={2.5} fill={C_NEG} opacity={0.7} />
          ))}
          <line x1={sx(threshold)} y1={6} x2={sx(threshold)} y2={104} stroke="#e2e8f0" strokeWidth={2} strokeDasharray="4 3" />
          <text x={sx(threshold)} y={4} textAnchor="middle" fill="#e2e8f0" fontSize={10} className="font-mono">阈值</text>
          <text x={SW - 30} y={50} textAnchor="end" fill="#475569" fontSize={9}>→ 判为正类</text>
        </svg>
      </div>

      <div className="flex gap-6 flex-wrap">
        {/* ROC 曲线 */}
        <div>
          <div className="text-xs text-slate-500 mb-1">ROC 曲线（AUC = {auc.toFixed(3)}）</div>
          <svg width={200} height={200} className="bg-[#0a0e17] rounded-md">
            {(() => {
              const P = (fpr: number, tpr: number) => `${20 + fpr * 160},${180 - tpr * 160}`;
              return (
                <>
                  <line x1={20} y1={180} x2={180} y2={20} stroke="#334155" strokeDasharray="4 3" />
                  <polyline points={roc.map((p) => P(p.fpr, p.tpr)).join(" ")} fill="none" stroke="#00e5ff" strokeWidth={2} />
                  <circle cx={20 + current.fpr * 160} cy={180 - current.tpr * 160} r={5} fill="#00ff88" stroke="#0a0e17" strokeWidth={1.5} />
                  <line x1={20} y1={180} x2={180} y2={180} stroke="#475569" />
                  <line x1={20} y1={20} x2={20} y2={180} stroke="#475569" />
                  <text x={100} y={196} textAnchor="middle" fill="#64748b" fontSize={10}>假阳率 FPR</text>
                  <text x={8} y={100} textAnchor="middle" fill="#64748b" fontSize={10} transform="rotate(-90 8 100)">真阳率 TPR</text>
                </>
              );
            })()}
          </svg>
        </div>

        {/* 混淆矩阵 + 指标 */}
        <div className="flex flex-col gap-2 text-xs">
          <div className="text-slate-500">混淆矩阵</div>
          <div className="grid grid-cols-[auto_1fr_1fr] gap-1 font-mono">
            <div />
            <div className="text-center text-slate-500 text-[10px]">预测正</div>
            <div className="text-center text-slate-500 text-[10px]">预测负</div>
            <div className="text-slate-500 text-[10px] self-center">实际正</div>
            <div className="text-center py-2 rounded bg-[rgba(0,255,136,0.15)] text-[#00ff88]">{confusion.tp}<div className="text-[9px] text-slate-500">TP</div></div>
            <div className="text-center py-2 rounded bg-[rgba(255,82,82,0.12)] text-[#ff5252]">{confusion.fn}<div className="text-[9px] text-slate-500">FN</div></div>
            <div className="text-slate-500 text-[10px] self-center">实际负</div>
            <div className="text-center py-2 rounded bg-[rgba(255,82,82,0.12)] text-[#ff5252]">{confusion.fp}<div className="text-[9px] text-slate-500">FP</div></div>
            <div className="text-center py-2 rounded bg-[rgba(0,255,136,0.15)] text-[#00ff88]">{confusion.tn}<div className="text-[9px] text-slate-500">TN</div></div>
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-1 font-mono">
            <Row k="精确率" v={precision} />
            <Row k="召回率" v={recall} />
            <Row k="F1" v={f1} />
            <Row k="阈值" v={threshold} raw />
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ k, v, raw }: { k: string; v: number; raw?: boolean }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-500">{k}</span>
      <span className="text-slate-200">{raw ? v.toFixed(2) : v.toFixed(3)}</span>
    </div>
  );
}
