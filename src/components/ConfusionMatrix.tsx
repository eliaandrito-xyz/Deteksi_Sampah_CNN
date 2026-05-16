import { ConfusionMatrixData } from '../types';

interface Props {
  data: ConfusionMatrixData | null;
  title?: string;
}

export default function ConfusionMatrix({ data, title = "Confusion Matrix" }: Props) {
  if (!data) {
    return (
      <div className="bg-slate-800 rounded-xl p-12 border border-dashed border-slate-600 text-center">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-700 flex items-center justify-center">
          <svg className="w-8 h-8 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
          </svg>
        </div>
        <p className="text-slate-400 text-sm">Belum ada data evaluasi</p>
        <p className="text-slate-500 text-xs mt-1">Jalankan training terlebih dahulu</p>
      </div>
    );
  }

  const { matrix, labels } = data;
  const total = matrix.flat().reduce((a, b) => a + b, 0);

  const tp = matrix[0][0];
  const fn = matrix[0][1];
  const fp = matrix[1][0];
  const tn = matrix[1][1];

  const accuracy  = (tp + tn) / total;
  const precision = tp / (tp + fp) || 0;
  const recall    = tp / (tp + fn) || 0;
  const f1        = (2 * precision * recall) / (precision + recall) || 0;
  const specificity = tn / (tn + fp) || 0;

  const maxVal = Math.max(...matrix.flat());
  const cellColor = (val: number, isCorrect: boolean) => {
    const intensity = val / maxVal;
    if (isCorrect) {
      const b = Math.round(255 - intensity * 180);
      return `rgb(34, ${Math.round(150 + intensity * 60)}, ${b})`;
    }
    return `rgba(248, 113, 113, ${0.1 + intensity * 0.7})`;
  };

  return (
    <div className="space-y-5">
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-4 border border-slate-600">
        <h2 className="text-base font-bold text-slate-100 mb-1">{title}</h2>
        <p className="text-xs text-slate-400">Visualisasi performa klasifikasi model</p>
      </div>

      {/* Matrix */}
      <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
        <div className="flex flex-col items-center">
          {/* Predicted header */}
          <div className="flex ml-20 mb-2 w-full max-w-xs">
            <div className="flex-1 text-center text-xs font-semibold text-slate-400 uppercase tracking-wider" style={{ marginLeft: '0px' }}>
              ← Prediksi →
            </div>
          </div>

          <div className="flex items-center">
            {/* Actual label (rotated) */}
            <div className="flex items-center mr-3" style={{ height: 160 }}>
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider"
                style={{ transform: 'rotate(-90deg)', whiteSpace: 'nowrap' }}>
                ← Aktual →
              </div>
            </div>

            {/* Matrix grid */}
            <div>
              {/* Col headers */}
              <div className="flex mb-1">
                <div className="w-20" />
                {labels.map(l => (
                  <div key={l} className="w-20 text-center text-xs font-semibold text-slate-300">{l}</div>
                ))}
              </div>

              {matrix.map((row, i) => (
                <div key={i} className="flex items-center mb-1">
                  {/* Row header */}
                  <div className="w-20 text-right pr-3 text-xs font-semibold text-slate-300">{labels[i]}</div>
                  {row.map((val, j) => {
                    const isCorrect = i === j;
                    const pct = ((val / total) * 100).toFixed(1);
                    return (
                      <div
                        key={j}
                        className="w-20 h-20 flex flex-col items-center justify-center rounded-lg mx-0.5 border"
                        style={{
                          backgroundColor: cellColor(val, isCorrect),
                          borderColor: isCorrect ? '#22d3ee44' : '#f8717144',
                        }}
                      >
                        <span className="text-xl font-bold text-white">{val}</span>
                        <span className="text-xs text-white/70">{pct}%</span>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div className="flex gap-4 mt-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#22c55e' }} />
              <span className="text-slate-400">Benar (TP/TN)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-4 h-4 rounded bg-red-500/60" />
              <span className="text-slate-400">Salah (FP/FN)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-3">
        {[
          { label: 'Accuracy',    value: accuracy,    color: '#22d3ee', desc: 'Prediksi benar keseluruhan' },
          { label: 'Precision',   value: precision,   color: '#34d399', desc: 'Ketepatan prediksi positif' },
          { label: 'Recall',      value: recall,      color: '#fb923c', desc: 'Sensitivitas deteksi positif' },
          { label: 'F1-Score',    value: f1,          color: '#a78bfa', desc: 'Harmonic mean precision & recall' },
          { label: 'Specificity', value: specificity, color: '#f472b6', desc: 'Ketepatan prediksi negatif' },
          { label: 'Total Data',  value: total,       color: '#94a3b8', desc: 'Jumlah sampel uji', raw: true },
        ].map(({ label, value, color, desc, raw }) => (
          <div key={label} className="bg-slate-800 rounded-xl p-3 border border-slate-700">
            <p className="text-xs text-slate-400">{label}</p>
            <p className="text-xl font-bold font-mono" style={{ color }}>
              {raw ? value : `${((value as number) * 100).toFixed(2)}%`}
            </p>
            <p className="text-xs text-slate-500 mt-0.5">{desc}</p>
          </div>
        ))}
      </div>

      {/* Classification Report */}
      <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Classification Report</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700">
                <th className="text-left py-1.5 pr-4">Kelas</th>
                <th className="text-right py-1.5 px-3">Precision</th>
                <th className="text-right py-1.5 px-3">Recall</th>
                <th className="text-right py-1.5 px-3">F1-Score</th>
                <th className="text-right py-1.5 pl-3">Support</th>
              </tr>
            </thead>
            <tbody>
              <tr className="text-slate-200 border-b border-slate-700/50">
                <td className="py-1.5 pr-4 text-cyan-400">{labels[0]}</td>
                <td className="text-right py-1.5 px-3">{precision.toFixed(4)}</td>
                <td className="text-right py-1.5 px-3">{recall.toFixed(4)}</td>
                <td className="text-right py-1.5 px-3">{f1.toFixed(4)}</td>
                <td className="text-right py-1.5 pl-3">{tp + fn}</td>
              </tr>
              <tr className="text-slate-200 border-b border-slate-700/50">
                <td className="py-1.5 pr-4 text-teal-400">{labels[1]}</td>
                <td className="text-right py-1.5 px-3">{(tn / (tn + fn)).toFixed(4)}</td>
                <td className="text-right py-1.5 px-3">{specificity.toFixed(4)}</td>
                <td className="text-right py-1.5 px-3">{(2 * (tn / (tn + fn)) * specificity / ((tn / (tn + fn)) + specificity)).toFixed(4)}</td>
                <td className="text-right py-1.5 pl-3">{fp + tn}</td>
              </tr>
              <tr className="text-slate-300 font-semibold">
                <td className="py-1.5 pr-4">avg / total</td>
                <td className="text-right py-1.5 px-3">{((precision + tn / (tn + fn)) / 2).toFixed(4)}</td>
                <td className="text-right py-1.5 px-3">{((recall + specificity) / 2).toFixed(4)}</td>
                <td className="text-right py-1.5 px-3">{f1.toFixed(4)}</td>
                <td className="text-right py-1.5 pl-3">{total}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
