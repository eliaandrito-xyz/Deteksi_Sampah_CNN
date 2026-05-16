import { TrainingMetrics } from '../types';

interface Props {
  metrics: TrainingMetrics[];
  isRunning: boolean;
  currentEpoch: number;
  totalEpochs: number;
}


export default function TrainingChart({ metrics, isRunning, currentEpoch, totalEpochs }: Props) {
  const latest = metrics[metrics.length - 1];
  const progress = totalEpochs > 0 ? (currentEpoch / totalEpochs) * 100 : 0;

  const renderChart = (
    trainKey: 'trainAcc' | 'trainLoss',
    valKey: 'valAcc' | 'valLoss',
    label: string,
    trainColor: string,
    valColor: string,
    isPercent: boolean
  ) => {
    if (metrics.length < 2) return null;
    const trainVals = metrics.map(d => d[trainKey]);
    const valVals = metrics.map(d => d[valKey]);
    const allVals = [...trainVals, ...valVals];
    const min = Math.min(...allVals);
    const max = Math.max(...allVals);
    const range = max - min || 0.01;
    const W = 400;
    const H = 120;
    const pad = 8;

    const toPath = (vals: number[]) =>
      vals.map((v, i) => {
        const x = pad + (i / (vals.length - 1)) * (W - pad * 2);
        const y = pad + ((max - v) / range) * (H - pad * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');

    const gridLines = [0, 0.25, 0.5, 0.75, 1].map(t => {
      const v = min + t * range;
      const y = pad + (1 - t) * (H - pad * 2);
      return { y, label: isPercent ? `${(v * 100).toFixed(0)}%` : v.toFixed(3) };
    });

    return (
      <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
        <div className="flex justify-between items-center mb-3">
          <h3 className="text-sm font-semibold text-slate-200">{label}</h3>
          <div className="flex gap-4 text-xs">
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 inline-block" style={{ backgroundColor: trainColor }} />
              <span className="text-slate-400">Train</span>
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 inline-block border-t-2 border-dashed" style={{ borderColor: valColor }} />
              <span className="text-slate-400">Validasi</span>
            </span>
          </div>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-32">
          {gridLines.map(({ y, label: gl }) => (
            <g key={y}>
              <line x1={pad} y1={y} x2={W - pad} y2={y} stroke="#334155" strokeWidth="0.5" />
              <text x={2} y={y + 3} fill="#64748b" fontSize="8">{gl}</text>
            </g>
          ))}
          <polyline fill="none" stroke={trainColor} strokeWidth="2" points={toPath(trainVals)} strokeLinecap="round" />
          <polyline fill="none" stroke={valColor} strokeWidth="2" strokeDasharray="4,3" points={toPath(valVals)} strokeLinecap="round" />
        </svg>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-4 border border-slate-600">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-base font-bold text-slate-100">Monitoring Training</h2>
          <span className={`text-xs px-2 py-1 rounded-full font-medium ${
            isRunning ? 'bg-emerald-900 text-emerald-300 animate-pulse' : 'bg-slate-700 text-slate-400'
          }`}>
            {isRunning ? 'Berjalan' : metrics.length > 0 ? 'Selesai' : 'Menunggu'}
          </span>
        </div>
        {totalEpochs > 0 && (
          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>Epoch {currentEpoch} / {totalEpochs}</span>
              <span>{progress.toFixed(0)}%</span>
            </div>
            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-teal-500 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {latest ? (
        <div className="grid grid-cols-2 gap-3">
          {[
            { label: 'Train Accuracy', value: latest.trainAcc, color: '#22d3ee', fmt: (v: number) => `${(v * 100).toFixed(2)}%` },
            { label: 'Val Accuracy',   value: latest.valAcc,   color: '#34d399', fmt: (v: number) => `${(v * 100).toFixed(2)}%` },
            { label: 'Train Loss',     value: latest.trainLoss, color: '#fb923c', fmt: (v: number) => v.toFixed(4) },
            { label: 'Val Loss',       value: latest.valLoss,   color: '#f87171', fmt: (v: number) => v.toFixed(4) },
          ].map(({ label, value, color, fmt }) => (
            <div key={label} className="bg-slate-800 rounded-xl p-3 border border-slate-700">
              <p className="text-xs text-slate-400 mb-1">{label}</p>
              <p className="text-xl font-bold font-mono" style={{ color }}>{fmt(value)}</p>
            </div>
          ))}
        </div>
      ) : isRunning ? (
        <div className="bg-slate-800 rounded-xl p-8 border border-slate-700 text-center">
          <div className="flex justify-center mb-3">
            <div className="w-6 h-6 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
          </div>
          <p className="text-slate-400 text-xs">Menunggu data epoch pertama dari Python...</p>
        </div>
      ) : null}

      {metrics.length >= 2 ? (
        <>
          {renderChart('trainAcc', 'valAcc', 'Kurva Akurasi', '#22d3ee', '#34d399', true)}
          {renderChart('trainLoss', 'valLoss', 'Kurva Loss', '#fb923c', '#f87171', false)}
        </>
      ) : isRunning && metrics.length > 0 ? (
        <div className="bg-slate-800 rounded-xl p-12 border border-dashed border-slate-700 text-center">
          <p className="text-slate-500 text-xs italic">Grafik akan muncul setelah 2 epoch selesai...</p>
        </div>
      ) : metrics.length === 0 && !isRunning && (
        <div className="bg-slate-800 rounded-xl p-12 border border-dashed border-slate-600 text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-700 flex items-center justify-center">
            <svg className="w-8 h-8 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2m0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <p className="text-slate-400 text-sm">Belum ada data training</p>
          <p className="text-slate-500 text-xs mt-1">Jalankan script Python untuk melihat progress di sini.</p>
        </div>
      )}
    </div>
  );
}
