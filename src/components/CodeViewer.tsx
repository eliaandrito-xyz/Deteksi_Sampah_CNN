import { useState } from 'react';
import { ModelConfig } from '../types';
import { generatePythonCode } from '../utils/codeGenerator';
import { Copy, Check, Download } from 'lucide-react';

interface Props {
  config: ModelConfig;
}

export default function CodeViewer({ config }: Props) {
  const [copied, setCopied] = useState(false);
  const code = generatePythonCode(config);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cnn_klasifikasi_sampah.py';
    a.click();
    URL.revokeObjectURL(url);
  };

  const lines = code.split('\n');

  return (
    <div className="space-y-4">
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-4 border border-slate-600">
        <div className="flex justify-between items-start">
          <div>
            <h2 className="text-base font-bold text-slate-100 mb-1">Kode Python</h2>
            <p className="text-xs text-slate-400">
              Kode siap pakai sesuai konfigurasi — {lines.length} baris
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleCopy}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                copied
                  ? 'bg-emerald-600 text-white'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              {copied ? <Check size={12} /> : <Copy size={12} />}
              {copied ? 'Disalin!' : 'Salin'}
            </button>
            <button
              onClick={handleDownload}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-cyan-600 text-white hover:bg-cyan-500 transition-all"
            >
              <Download size={12} />
              Unduh .py
            </button>
          </div>
        </div>
      </div>

      {/* Dependencies info */}
      <div className="bg-slate-800 rounded-xl p-3 border border-amber-800/50">
        <p className="text-xs font-semibold text-amber-400 mb-1.5">Dependensi yang Diperlukan</p>
        <code className="text-xs text-slate-300 font-mono">
          pip install tensorflow numpy matplotlib seaborn scikit-learn
        </code>
      </div>

      {/* Code block */}
      <div className="bg-slate-900 rounded-xl border border-slate-700 overflow-hidden">
        {/* Editor header */}
        <div className="flex items-center gap-2 px-4 py-2 bg-slate-800 border-b border-slate-700">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <div className="w-3 h-3 rounded-full bg-yellow-500" />
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span className="ml-2 text-xs text-slate-400 font-mono">cnn_klasifikasi_sampah.py</span>
        </div>

        <div className="overflow-auto max-h-[60vh]">
          <table className="w-full text-xs font-mono border-collapse">
            <tbody>
              {lines.map((line, i) => {
                const n = i + 1;
                const hl = line.startsWith('#') || line.startsWith('# ═') || line.startsWith('# ─');
                const isSection = line.startsWith('# ─') || line.startsWith('# =') || line.startsWith('# ════');
                return (
                  <tr key={n} className={`group ${isSection ? 'bg-slate-800/40' : 'hover:bg-slate-800/30'}`}>
                    <td className="w-10 pl-3 pr-2 py-0.5 text-slate-600 select-none text-right border-r border-slate-800">
                      {n}
                    </td>
                    <td className="pl-4 py-0.5 pr-4 whitespace-pre">
                      <ColorizedLine line={line} isComment={hl} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Struktur Dataset */}
      <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Struktur Dataset yang Dibutuhkan</h3>
        <pre className="text-xs font-mono text-slate-400 leading-relaxed">
{`dataset/
├── organik/
│   ├── sampah_organik_001.jpg
│   ├── sampah_organik_002.jpg
│   └── ...
└── non_organik/
    ├── sampah_non_001.jpg
    ├── sampah_non_002.jpg
    └── ...`}
        </pre>
        <p className="text-xs text-slate-500 mt-3">
          Minimal 100–200 gambar per kelas. Semakin banyak data, semakin baik performa model.
        </p>
      </div>
    </div>
  );
}

function ColorizedLine({ line, isComment }: { line: string; isComment: boolean }) {
  if (isComment) return <span className="text-slate-500">{line}</span>;

  const keywords = ['import', 'from', 'def', 'class', 'if', 'else', 'elif', 'for', 'while',
    'return', 'in', 'not', 'and', 'or', 'True', 'False', 'None', 'as', 'with', 'lambda'];
  const builtins = ['print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted',
    'min', 'max', 'sum', 'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool'];

  const parts: { text: string; color: string }[] = [];
  let remaining = line;

  while (remaining.length > 0) {
    if (remaining.startsWith("'") || remaining.startsWith('"')) {
      const q = remaining[0];
      const end = remaining.indexOf(q, 1);
      if (end !== -1) {
        parts.push({ text: remaining.slice(0, end + 1), color: '#86efac' });
        remaining = remaining.slice(end + 1);
        continue;
      }
    }

    const match = remaining.match(/^[a-zA-Z_][a-zA-Z0-9_]*/);
    if (match) {
      const word = match[0];
      if (keywords.includes(word)) {
        parts.push({ text: word, color: '#f472b6' });
      } else if (builtins.includes(word)) {
        parts.push({ text: word, color: '#fb923c' });
      } else if (/^[A-Z]/.test(word)) {
        parts.push({ text: word, color: '#67e8f9' });
      } else {
        parts.push({ text: word, color: '#e2e8f0' });
      }
      remaining = remaining.slice(word.length);
      continue;
    }

    const numMatch = remaining.match(/^[0-9]+\.?[0-9]*(e[+-]?[0-9]+)?/);
    if (numMatch) {
      parts.push({ text: numMatch[0], color: '#a78bfa' });
      remaining = remaining.slice(numMatch[0].length);
      continue;
    }

    parts.push({ text: remaining[0], color: '#94a3b8' });
    remaining = remaining.slice(1);
  }

  return (
    <>
      {parts.map((p, i) => (
        <span key={i} style={{ color: p.color }}>{p.text}</span>
      ))}
    </>
  );
}
