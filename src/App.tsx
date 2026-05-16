import { useState, useCallback, useRef, useEffect } from 'react';
import { Settings, BarChart2, Grid2x2 as Grid, Code, Cpu, Leaf, Trash2, Camera, Check } from 'lucide-react';
import { ModelConfig, TrainingMetrics, ConfusionMatrixData, KFoldResult, TabType } from './types';
import ConfigPanel from './components/ConfigPanel';
import TrainingChart from './components/TrainingChart';
import ConfusionMatrix from './components/ConfusionMatrix';
import ImageDetection from './components/ImageDetection';
import KFoldPanel from './components/KFoldPanel';
import CodeViewer from './components/CodeViewer';
import { simulateTrainingMetrics, simulateConfusionMatrix, simulateKFoldResults } from './utils/simulation';

const DEFAULT_CONFIG: ModelConfig = {
  epochs: 30,
  batchSize: 32,
  learningRate: 0.001,
  dropoutRate: 0.3,
  kFolds: 5,
  optimizer: 'adam',
  architecture: 'custom',
  imageSize: 224,
  augmentation: true,
  splitRatio: 0.8, // 80% Training
};

const TABS: { id: TabType; label: string; icon: React.ReactNode }[] = [
  { id: 'config', label: 'Konfigurasi', icon: <Settings size={15} /> },
  { id: 'training', label: 'Training', icon: <BarChart2 size={15} /> },
  { id: 'evaluation', label: 'Evaluasi', icon: <Grid size={15} /> },
  { id: 'detect', label: 'Deteksi', icon: <Camera size={15} /> },
  { id: 'code', label: 'Kode Python', icon: <Code size={15} /> },
];

export default function App() {
  const [config, setConfig] = useState<ModelConfig>(DEFAULT_CONFIG);
  const [activeTab, setActiveTab] = useState<TabType>('config');
  const [isRunning, setIsRunning] = useState(false);
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [cmDataTrain, setCmDataTrain] = useState<ConfusionMatrixData | null>(null);
  const [cmDataVal, setCmDataVal] = useState<ConfusionMatrixData | null>(null);
  const [kfoldResults, setKfoldResults] = useState<KFoldResult[]>([]);
  const [isModelTrained, setIsModelTrained] = useState(false);
  const [totalEpochsFromBackend, setTotalEpochsFromBackend] = useState(0);
  const [isServerOnline, setIsServerOnline] = useState(false);
  const cancelRef = useRef(false);

  // Load from localStorage and check backend status
  useEffect(() => {
    const savedMetrics = localStorage.getItem('training_metrics');
    const savedCmTrain = localStorage.getItem('cm_data_train');
    const savedCmVal = localStorage.getItem('cm_data_val');
    const savedKfold = localStorage.getItem('kfold_results');

    if (savedMetrics) {
      const parsedMetrics = JSON.parse(savedMetrics);
      setMetrics(parsedMetrics);
      setCurrentEpoch(parsedMetrics.length);
    }
    if (savedCmTrain) setCmDataTrain(JSON.parse(savedCmTrain));
    if (savedCmVal) setCmDataVal(JSON.parse(savedCmVal));
    if (savedKfold) setKfoldResults(JSON.parse(savedKfold));

    // Check backend status immediately and then every 2 seconds for live training
    const checkStatus = () => {
      fetch('http://localhost:5000/get_metrics')
        .then(res => res.json())
        .then(data => {
          setIsServerOnline(true);
          if (data.is_training) {
            setMetrics(data.metrics);
            setCurrentEpoch(data.current_epoch);
            setTotalEpochsFromBackend(data.total_epochs);
            setIsRunning(true);
            setActiveTab('training');
            // Reset evaluation data when training starts
            setCmDataTrain(null);
            setCmDataVal(null);
            setKfoldResults([]);
          } else if (isRunning && !data.is_training) {
            // Training baru saja selesai
            setIsRunning(false);
            setIsModelTrained(true);
            setActiveTab('evaluation');
          }

          // Selalu update data evaluasi & metrik jika ada di server (agar tidak hilang saat refresh)
          if (data.metrics && data.metrics.length > 0) {
            setMetrics(data.metrics);
            setCurrentEpoch(data.current_epoch || data.metrics.length);
          }
          if (data.cm_train) setCmDataTrain(data.cm_train);
          if (data.cm_val) setCmDataVal(data.cm_val);
          if (data.kfold_results) setKfoldResults(data.kfold_results);
        })
        .catch(() => {
          setIsServerOnline(false);
          console.log('Backend server not running');
        });

      fetch('http://localhost:5000/status')
        .then(res => res.json())
        .then(data => {
          setIsModelTrained(data.model_trained);
        })
        .catch(() => setIsModelTrained(false));
    };

    checkStatus();
    const interval = setInterval(checkStatus, 2000);
    return () => clearInterval(interval);
  }, [isRunning]);

  useEffect(() => {
    localStorage.setItem('training_metrics', JSON.stringify(metrics));
    localStorage.setItem('cm_data_train', JSON.stringify(cmDataTrain));
    localStorage.setItem('cm_data_val', JSON.stringify(cmDataVal));
    localStorage.setItem('kfold_results', JSON.stringify(kfoldResults));
  }, [metrics, cmDataTrain, cmDataVal, kfoldResults]);

  const startSimulation = useCallback(async () => {
    setIsRunning(true);
    setMetrics([]);
    setCurrentEpoch(0);
    setCmData(null);
    setKfoldResults([]);
    cancelRef.current = false;
    setActiveTab('training');

    const allMetrics = simulateTrainingMetrics(config);
    const delay = Math.max(40, Math.min(300, 6000 / config.epochs));

    for (let i = 0; i < allMetrics.length; i++) {
      if (cancelRef.current) break;
      await new Promise(r => setTimeout(r, delay));
      setMetrics(prev => [...prev, allMetrics[i]]);
      setCurrentEpoch(i + 1);
    }

    if (!cancelRef.current) {
      setCmDataTrain(simulateConfusionMatrix());
      setCmDataVal(simulateConfusionMatrix());
      setKfoldResults(simulateKFoldResults(config));
    }
    setIsRunning(false);
  }, [config]);

  const getBestMetric = () => {
    if (metrics.length === 0) return null;
    return metrics.reduce((best, m) => m.valAcc > best.valAcc ? m : best, metrics[0]);
  };

  const best = getBestMetric();

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-teal-600 flex items-center justify-center shadow-lg shadow-cyan-900/50">
              <Cpu size={16} className="text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-slate-100 leading-none">CNN Deteksi Sampah</h1>
              <p className="text-xs text-slate-500 mt-0.5">Klasifikasi Sampah Organik & Non-Organik</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 mr-2">
              <div className={`w-2 h-2 rounded-full ${isServerOnline ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]' : 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]'}`} />
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                Server: {isServerOnline ? 'Online' : 'Offline'}
              </span>
            </div>
            {best && (
              <div className="hidden sm:flex items-center gap-3 bg-slate-800 rounded-lg px-3 py-1.5 border border-slate-700">
                <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-xs text-slate-400">Best Val Acc</span>
                  <span className="text-xs font-bold text-emerald-400 font-mono">
                    {(best.valAcc * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="w-px h-4 bg-slate-700" />
                <span className="text-xs text-slate-500">Epoch {best.epoch}</span>
              </div>
            )}
            <div className="flex items-center gap-2">
              {isModelTrained && (
                <div className="flex items-center gap-1 bg-cyan-900/30 rounded-full px-2 py-0.5 border border-cyan-800/50">
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)]" />
                  <span className="text-[10px] text-cyan-400 font-bold uppercase tracking-wider">Model Ready</span>
                </div>
              )}
              <div className="flex items-center gap-1 bg-cyan-900/30 rounded-full px-2 py-0.5 border border-cyan-800/50">
                <BarChart2 size={10} className="text-cyan-400" />
                <span className="text-[10px] text-cyan-400 font-bold">Rasio 80:20</span>
              </div>
              <div className="flex items-center gap-1 bg-green-900/30 rounded-full px-2 py-0.5 border border-green-800/50">
                <Leaf size={10} className="text-green-400" />
                <span className="text-xs text-green-400">Organik</span>
              </div>
              <div className="flex items-center gap-1 bg-orange-900/30 rounded-full px-2 py-0.5 border border-orange-800/50">
                <Trash2 size={10} className="text-orange-400" />
                <span className="text-xs text-orange-400">Non-Organik</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Tabs */}
      <div className="border-b border-slate-800 bg-slate-900/60">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1">
            {TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-4 py-3 text-xs font-semibold border-b-2 transition-colors ${activeTab === tab.id
                  ? 'border-cyan-500 text-cyan-400'
                  : 'border-transparent text-slate-400 hover:text-slate-300'
                  }`}
              >
                {tab.icon}
                {tab.label}
                {tab.id === 'training' && isRunning && (
                  <span className="ml-1 w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
                )}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-6">
        {activeTab === 'config' && (
          <div className="max-w-xl mx-auto space-y-4">
            {isModelTrained && (
              <div className="bg-emerald-900/20 border border-emerald-500/50 rounded-xl p-4 flex items-center gap-4">
                <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400">
                  <Check size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-emerald-400">Model Siap Digunakan!</h3>
                  <p className="text-xs text-slate-400">Sistem mendeteksi model sudah dilatih di Python. Anda bisa langsung ke tab Deteksi.</p>
                </div>
                <button
                  onClick={() => setActiveTab('detect')}
                  className="ml-auto px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold rounded-lg transition-colors"
                >
                  Ke Tab Deteksi
                </button>
              </div>
            )}
            <ConfigPanel
              config={config}
              onChange={setConfig}
              onStart={startSimulation}
              isRunning={isRunning}
            />
          </div>
        )}

        {activeTab === 'training' && (
          <div className="max-w-2xl mx-auto">
            <TrainingChart
              metrics={metrics}
              isRunning={isRunning}
              currentEpoch={currentEpoch}
              totalEpochs={totalEpochsFromBackend || config.epochs}
            />
          </div>
        )}


        {activeTab === 'evaluation' && (
          <div className="space-y-8 max-w-6xl mx-auto">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <ConfusionMatrix
                data={cmDataTrain}
                title="Confusion Matrix (Training Set)"
              />
              <ConfusionMatrix
                data={cmDataVal}
                title="Confusion Matrix (Validation Set)"
              />
            </div>
            <KFoldPanel results={kfoldResults} />
          </div>
        )}

        {activeTab === 'detect' && (
          <div className="max-w-4xl mx-auto">
            <ImageDetection isServerReady={isModelTrained} />
          </div>
        )}

        {activeTab === 'code' && (
          <div className="max-w-4xl mx-auto">
            <CodeViewer config={config} />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-3 text-center">
        <p className="text-xs text-slate-600">
          CNN Waste Classification System — 404 — NTT, Andrito Elia
        </p>
      </footer>
    </div>
  );
}
