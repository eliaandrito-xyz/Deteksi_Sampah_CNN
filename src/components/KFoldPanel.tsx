import { KFoldResult } from '../types';

interface Props {
  results: KFoldResult[];
}

function avg(arr: number[]) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function std(arr: number[]) {
  const m = avg(arr);
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
}

export default function KFoldPanel({ results }: Props) {
  if (results.length === 0) {
    return (
      <div className="bg-slate-800 rounded-xl p-12 border border-dashed border-slate-600 text-center">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-700 flex items-center justify-center">
          <svg className="w-8 h-8 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <p className="text-slate-400 text-sm">Belum ada hasil K-Fold</p>
        <p className="text-slate-500 text-xs mt-1">Jalankan training terlebih dahulu</p>
      </div>
    );
  }

  const accs  = results.map(r => r.accuracy);
  const precs = results.map(r => r.precision);
  const recs  = results.map(r => r.recall);
  const f1s   = results.map(r => r.f1Score);

  const summary = [
    { label: 'Accuracy',  values: accs,  color: '#22d3ee' },
    { label: 'Precision', values: precs, color: '#34d399' },
    { label: 'Recall',    values: recs,  color: '#fb923c' },
    { label: 'F1-Score',  values: f1s,   color: '#a78bfa' },
  ];

  const maxAcc = Math.max(...accs);
  const minAcc = Math.min(...accs);

  const BarChart = ({ metric, color }: { metric: number[]; color: string }) => {
    const W = 280;
    const H = 80;
    const barW = (W - 16) / metric.length - 4;
    const minV = Math.max(0, Math.min(...metric) - 0.05);
    const maxV = Math.min(1, Math.max(...metric) + 0.02);
    const range = maxV - minV;

    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {[0, 0.5, 1].map(t => {
          const y = H - 10 - t * (H - 20);
          const v = minV + t * range;
          return (
            <g key={t}>
              <line x1={8} y1={y} x2={W - 8} y2={y} stroke="#334155" strokeWidth="0.5" />
              <text x={2} y={y + 3} fill="#64748b" fontSize="7">{(v * 100).toFixed(0)}</text>
            </g>
          );
        })}
        {metric.map((v, i) => {
          const barH = ((v - minV) / range) * (H - 20);
          const x = 8 + i * (barW + 4);
          const y = H - 10 - barH;
          return (
            <g key={i}>
              <rect x={x} y={y} width={barW} height={barH} fill={color} opacity={0.8} rx={2} />
              <text x={x + barW / 2} y={H - 1} fill="#94a3b8" fontSize="7" textAnchor="middle">F{i + 1}</text>
            </g>
          );
        })}
      </svg>
    );
  };

  return (
    <div className="space-y-5">
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-4 border border-slate-600">
        <h2 className="text-base font-bold text-slate-100 mb-1">K-Fold Cross Validation</h2>
        <p className="text-xs text-slate-400">{results.length}-Fold — evaluasi robustness model</p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-3">
        {summary.map(({ label, values, color }) => (
          <div key={label} className="bg-slate-800 rounded-xl p-3 border border-slate-700">
            <p className="text-xs text-slate-400 mb-1">{label}</p>
            <p className="text-lg font-bold font-mono" style={{ color }}>
              {(avg(values) * 100).toFixed(2)}%
            </p>
            <p className="text-xs text-slate-500">± {(std(values) * 100).toFixed(2)}%</p>
            <div className="mt-2">
              <BarChart metric={values} color={color} />
            </div>
          </div>
        ))}
      </div>

      {/* Per-fold table */}
      <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Hasil per Fold</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700">
                <th className="text-left py-2 pr-3">Fold</th>
                <th className="text-right py-2 px-2">Accuracy</th>
                <th className="text-right py-2 px-2">Precision</th>
                <th className="text-right py-2 px-2">Recall</th>
                <th className="text-right py-2 pl-2">F1-Score</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.fold} className="border-b border-slate-700/50 text-slate-200">
                  <td className="py-2 pr-3">
                    <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${
                      r.accuracy === maxAcc ? 'bg-cyan-500 text-white' :
                      r.accuracy === minAcc ? 'bg-red-900 text-red-200' :
                      'bg-slate-700 text-slate-300'
                    }`}>
                      {r.fold}
                    </span>
                  </td>
                  <td className="text-right py-2 px-2 font-mono">{(r.accuracy * 100).toFixed(2)}%</td>
                  <td className="text-right py-2 px-2 font-mono">{(r.precision * 100).toFixed(2)}%</td>
                  <td className="text-right py-2 px-2 font-mono">{(r.recall * 100).toFixed(2)}%</td>
                  <td className="text-right py-2 pl-2 font-mono">{(r.f1Score * 100).toFixed(2)}%</td>
                </tr>
              ))}
              {/* Average row */}
              <tr className="text-slate-100 font-semibold bg-slate-700/40">
                <td className="py-2 pr-3 text-slate-300">Rata-rata</td>
                <td className="text-right py-2 px-2 font-mono text-cyan-400">{(avg(accs) * 100).toFixed(2)}%</td>
                <td className="text-right py-2 px-2 font-mono text-emerald-400">{(avg(precs) * 100).toFixed(2)}%</td>
                <td className="text-right py-2 px-2 font-mono text-orange-400">{(avg(recs) * 100).toFixed(2)}%</td>
                <td className="text-right py-2 pl-2 font-mono text-violet-400">{(avg(f1s) * 100).toFixed(2)}%</td>
              </tr>
              <tr className="text-slate-400 text-xs">
                <td className="py-1 pr-3">Std Dev</td>
                <td className="text-right py-1 px-2 font-mono">±{(std(accs) * 100).toFixed(2)}%</td>
                <td className="text-right py-1 px-2 font-mono">±{(std(precs) * 100).toFixed(2)}%</td>
                <td className="text-right py-1 px-2 font-mono">±{(std(recs) * 100).toFixed(2)}%</td>
                <td className="text-right py-1 pl-2 font-mono">±{(std(f1s) * 100).toFixed(2)}%</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="flex gap-4 mt-3 text-xs text-slate-500">
          <span><span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-cyan-500 text-white text-xs mr-1">n</span> Fold terbaik</span>
          <span><span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-red-900 text-red-200 text-xs mr-1">n</span> Fold terburuk</span>
        </div>
      </div>

      {/* Stability gauge */}
      <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Stabilitas Model</h3>
        {(() => {
          const stdAcc = std(accs);
          const stability = stdAcc < 0.02 ? 'Sangat Stabil' : stdAcc < 0.04 ? 'Stabil' : stdAcc < 0.07 ? 'Cukup Stabil' : 'Tidak Stabil';
          const stabColor = stdAcc < 0.02 ? 'text-emerald-400' : stdAcc < 0.04 ? 'text-cyan-400' : stdAcc < 0.07 ? 'text-orange-400' : 'text-red-400';
          const stabPct = Math.max(0, 100 - stdAcc * 1000);
          return (
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm text-slate-400">Variasi Akurasi antar Fold</span>
                <span className={`text-sm font-bold ${stabColor}`}>{stability}</span>
              </div>
              <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${stabPct}%`,
                    background: stdAcc < 0.02 ? '#34d399' : stdAcc < 0.04 ? '#22d3ee' : stdAcc < 0.07 ? '#fb923c' : '#f87171',
                  }}
                />
              </div>
              <p className="text-xs text-slate-500 mt-2">
                Std Dev akurasi: {(stdAcc * 100).toFixed(2)}% — {
                  stdAcc < 0.02 ? 'Model sangat konsisten, siap untuk produksi.'
                  : stdAcc < 0.04 ? 'Model cukup konsisten.'
                  : stdAcc < 0.07 ? 'Pertimbangkan lebih banyak data atau regularisasi.'
                  : 'Model tidak stabil, perlu penyesuaian arsitektur.'
                }
              </p>
            </div>
          );
        })()}
      </div>
    </div>
  );
}
