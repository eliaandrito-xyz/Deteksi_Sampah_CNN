import { TrainingMetrics, ConfusionMatrixData, KFoldResult, ModelConfig } from '../types';

export function simulateTrainingMetrics(config: ModelConfig): TrainingMetrics[] {
  const metrics: TrainingMetrics[] = [];
  let trainAcc = 0.5 + Math.random() * 0.1;
  let valAcc = 0.45 + Math.random() * 0.1;
  let trainLoss = 0.9 + Math.random() * 0.2;
  let valLoss = 1.0 + Math.random() * 0.2;

  const targetAcc = 0.88 + Math.random() * 0.08;
  const archBonus = { custom: 0, mobilenet: 0.04, resnet50: 0.05, vgg16: 0.03 }[config.architecture];

  for (let epoch = 1; epoch <= config.epochs; epoch++) {
    const progress = epoch / config.epochs;
    const noise = () => (Math.random() - 0.5) * 0.015;

    trainAcc = Math.min(0.99, trainAcc + ((targetAcc + archBonus - trainAcc) * 0.15) + noise());
    valAcc = Math.min(0.97, valAcc + ((targetAcc + archBonus - 0.03 - valAcc) * 0.12) + noise());
    trainLoss = Math.max(0.01, trainLoss - (trainLoss * 0.12) + Math.abs(noise()));
    valLoss = Math.max(0.03, valLoss - (valLoss * 0.1) + Math.abs(noise()) * 1.2);

    if (progress > 0.7 && config.dropoutRate > 0.4) {
      valAcc += 0.005;
      valLoss -= 0.005;
    }

    metrics.push({
      epoch,
      trainAcc: Math.min(0.99, trainAcc),
      valAcc: Math.min(0.97, valAcc),
      trainLoss: Math.max(0.01, trainLoss),
      valLoss: Math.max(0.03, valLoss),
    });
  }
  return metrics;
}

export function simulateConfusionMatrix(): ConfusionMatrixData {
  const tp = 85 + Math.floor(Math.random() * 12);
  const tn = 82 + Math.floor(Math.random() * 12);
  const fp = 8 + Math.floor(Math.random() * 8);
  const fn = 6 + Math.floor(Math.random() * 8);

  return {
    matrix: [
      [tp, fn],
      [fp, tn],
    ],
    labels: ['Organik', 'Non-Organik'],
  };
}

export function simulateKFoldResults(config: ModelConfig): KFoldResult[] {
  const results: KFoldResult[] = [];
  const baseAcc = 0.86 + Math.random() * 0.06;

  for (let fold = 1; fold <= config.kFolds; fold++) {
    const noise = () => (Math.random() - 0.5) * 0.04;
    const acc = Math.min(0.98, Math.max(0.75, baseAcc + noise()));
    const prec = Math.min(0.98, Math.max(0.75, acc + noise() * 0.5));
    const rec = Math.min(0.98, Math.max(0.75, acc + noise() * 0.5));
    const f1 = (2 * prec * rec) / (prec + rec);

    results.push({ fold, accuracy: acc, precision: prec, recall: rec, f1Score: f1 });
  }
  return results;
}
