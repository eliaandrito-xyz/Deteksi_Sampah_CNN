import { ModelConfig } from '../types';

interface Props {
  config: ModelConfig;
  onChange: (config: ModelConfig) => void;
  onStart: () => void;
  isRunning: boolean;
}

function SliderField({
  label, name, value, min, max, step, format, description, onChange,
}: {
  label: string; name: keyof ModelConfig; value: number; min: number; max: number;
  step: number; format: (v: number) => string; description: string;
  onChange: (name: keyof ModelConfig, value: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <label className="text-sm font-semibold text-slate-200">{label}</label>
        <span className="text-sm font-mono font-bold text-cyan-400 bg-slate-800 px-2 py-0.5 rounded">
          {format(value)}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(name, parseFloat(e.target.value))}
        className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
      />
      <p className="text-xs text-slate-500">{description}</p>
    </div>
  );
}

export default function ConfigPanel({ config, onChange, onStart, isRunning }: Props) {
  const set = (name: keyof ModelConfig, value: number | string | boolean) =>
    onChange({ ...config, [name]: value });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-4 border border-slate-600">
        <h2 className="text-base font-bold text-slate-100 mb-1">Konfigurasi Model</h2>
        <p className="text-xs text-slate-400">Atur hyperparameter dan arsitektur CNN</p>
      </div>

      {/* Architecture & Image Size */}
      <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 space-y-4">
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Arsitektur</h3>

        <div className="space-y-2">
          <label className="text-sm font-semibold text-slate-200">Model Arsitektur</label>
          <select
            value={config.architecture}
            onChange={e => set('architecture', e.target.value)}
            className="w-full bg-slate-700 border border-slate-600 text-slate-200 text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            <option value="custom">Custom CNN</option>
            <option value="mobilenet">MobileNetV2 (Transfer Learning)</option>
            <option value="resnet50">ResNet50 (Transfer Learning)</option>
            <option value="vgg16">VGG16 (Transfer Learning)</option>
          </select>
          <p className="text-xs text-slate-500">MobileNetV2 & ResNet50 direkomendasikan untuk dataset kecil</p>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-semibold text-slate-200">Ukuran Input Gambar</label>
          <select
            value={config.imageSize}
            onChange={e => set('imageSize', parseInt(e.target.value))}
            className="w-full bg-slate-700 border border-slate-600 text-slate-200 text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            <option value={64}>64 × 64</option>
            <option value={128}>128 × 128</option>
            <option value={224}>224 × 224 (Standar ImageNet)</option>
            <option value={256}>256 × 256</option>
          </select>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-semibold text-slate-200">Optimizer</label>
          <select
            value={config.optimizer}
            onChange={e => set('optimizer', e.target.value)}
            className="w-full bg-slate-700 border border-slate-600 text-slate-200 text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            <option value="adam">Adam (Direkomendasikan)</option>
            <option value="sgd">SGD + Momentum</option>
            <option value="rmsprop">RMSprop</option>
          </select>
        </div>

        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold text-slate-200">Data Augmentation</p>
            <p className="text-xs text-slate-500">Rotasi, flip, zoom untuk memperbanyak data</p>
          </div>
          <button
            onClick={() => set('augmentation', !config.augmentation)}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              config.augmentation ? 'bg-cyan-500' : 'bg-slate-600'
            }`}
          >
            <span className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform shadow ${
              config.augmentation ? 'translate-x-7' : 'translate-x-1'
            }`} />
          </button>
        </div>

        <div className="pt-2 border-t border-slate-700/50 mt-4">
          <SliderField
            label="Rasio Pembagian Data (Train:Val)"
            name="splitRatio"
            value={config.splitRatio}
            min={0.5}
            max={0.9}
            step={0.05}
            format={v => `${Math.round(v * 100)}:${Math.round((1 - v) * 100)}`}
            description="Perbandingan data untuk latihan vs data untuk evaluasi (Standar 80:20)"
            onChange={set}
          />
        </div>
      </div>

      {/* Hyperparameters */}
      <div className="bg-slate-800 rounded-xl p-4 border border-slate-700 space-y-5">
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Hyperparameter</h3>

        <SliderField
          label="Jumlah Epoch" name="epochs" value={config.epochs}
          min={5} max={100} step={5}
          format={v => `${v}`}
          description="Jumlah iterasi penuh melalui dataset"
          onChange={set}
        />
        <SliderField
          label="Batch Size" name="batchSize" value={config.batchSize}
          min={8} max={128} step={8}
          format={v => `${v}`}
          description="Jumlah sampel per update gradient"
          onChange={set}
        />
        <SliderField
          label="Learning Rate" name="learningRate" value={config.learningRate}
          min={0.00001} max={0.01} step={0.00001}
          format={v => v.toExponential(1)}
          description="Kecepatan adaptasi bobot model"
          onChange={set}
        />
        <SliderField
          label="Dropout Rate" name="dropoutRate" value={config.dropoutRate}
          min={0.1} max={0.7} step={0.05}
          format={v => `${(v * 100).toFixed(0)}%`}
          description="Regulasi untuk mencegah overfitting"
          onChange={set}
        />
        <SliderField
          label="K-Fold (K)" name="kFolds" value={config.kFolds}
          min={2} max={10} step={1}
          format={v => `${v} Fold`}
          description="Jumlah partisi untuk cross-validation"
          onChange={set}
        />
      </div>

      {/* Summary */}
      <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">Ringkasan Konfigurasi</h3>
        <div className="grid grid-cols-2 gap-2">
          {[
            ['Arsitektur', config.architecture === 'custom' ? 'Custom CNN' : config.architecture.toUpperCase()],
            ['Epoch', config.epochs],
            ['Batch', config.batchSize],
            ['LR', config.learningRate.toExponential(1)],
            ['Dropout', `${(config.dropoutRate * 100).toFixed(0)}%`],
            ['K-Fold', `${config.kFolds}`],
            ['Input', `${config.imageSize}×${config.imageSize}`],
            ['Augment', config.augmentation ? 'Ya' : 'Tidak'],
          ].map(([k, v]) => (
            <div key={k} className="flex justify-between text-xs bg-slate-700 rounded px-2 py-1.5">
              <span className="text-slate-400">{k}</span>
              <span className="text-slate-100 font-mono font-semibold">{v}</span>
            </div>
          ))}
        </div>
      </div>

      <button
        onClick={onStart}
        disabled={isRunning}
        className={`w-full py-3 rounded-xl text-sm font-bold tracking-wide transition-all ${
          isRunning
            ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
            : 'bg-gradient-to-r from-cyan-500 to-teal-500 text-white hover:from-cyan-400 hover:to-teal-400 shadow-lg shadow-cyan-900/40 hover:shadow-cyan-800/50 active:scale-95'
        }`}
      >
        {isRunning ? 'Simulasi Berjalan...' : 'Mulai Simulasi Training'}
      </button>
    </div>
  );
}
