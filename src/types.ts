export interface ModelConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  dropoutRate: number;
  kFolds: number;
  optimizer: 'adam' | 'sgd' | 'rmsprop';
  architecture: 'custom' | 'mobilenet' | 'resnet50' | 'vgg16';
  imageSize: number;
  augmentation: boolean;
  splitRatio: number;
}

export interface TrainingMetrics {
  epoch: number;
  trainLoss: number;
  trainAcc: number;
  valLoss: number;
  valAcc: number;
}

export interface ConfusionMatrixData {
  matrix: number[][];
  labels: string[];
}

export interface KFoldResult {
  fold: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
}

export type TabType = 'config' | 'training' | 'evaluation' | 'detect' | 'code';
