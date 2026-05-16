import { ModelConfig } from '../types';

export function generatePythonCode(config: ModelConfig): string {
  const optimizerCode = {
    adam: `optimizer = tf.keras.optimizers.Adam(learning_rate=${config.learningRate})`,
    sgd: `optimizer = tf.keras.optimizers.SGD(learning_rate=${config.learningRate}, momentum=0.9)`,
    rmsprop: `optimizer = tf.keras.optimizers.RMSprop(learning_rate=${config.learningRate})`,
  }[config.optimizer];

  const architectureCode = getArchitectureCode(config);

  return `# ============================================================
# CNN Klasifikasi Sampah Organik dan Non-Organik
# Konfigurasi: Epoch=${config.epochs}, Batch=${config.batchSize},
#              LR=${config.learningRate}, Dropout=${config.dropoutRate}
#              K-Fold=${config.kFolds}, Arsitektur=${config.architecture}
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, classification_report,
                              precision_score, recall_score, f1_score, accuracy_score)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────
# Konfigurasi Parameter
# ────────────────────────────────────────────────
EPOCHS        = ${config.epochs}
BATCH_SIZE    = ${config.batchSize}
LEARNING_RATE = ${config.learningRate}
DROPOUT_RATE  = ${config.dropoutRate}
K_FOLDS       = ${config.kFolds}
IMAGE_SIZE    = (${config.imageSize}, ${config.imageSize})
INPUT_SHAPE   = (${config.imageSize}, ${config.imageSize}, 3)
NUM_CLASSES   = 2
CLASS_NAMES   = ['Organik', 'Non-Organik']

# ────────────────────────────────────────────────
# Persiapan Dataset
# ────────────────────────────────────────────────
def load_dataset(data_dir):
    """
    Struktur direktori yang diharapkan:
    data_dir/
        organik/
            gambar1.jpg
            gambar2.jpg
        non_organik/
            gambar1.jpg
            gambar2.jpg
    """
    images, labels = [], []
    class_map = {'organik': 0, 'non_organik': 1}

    for class_name, label in class_map.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"[PERINGATAN] Direktori tidak ditemukan: {class_dir}")
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(class_dir, fname)
                img = tf.keras.preprocessing.image.load_img(
                    img_path, target_size=IMAGE_SIZE
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)

    return np.array(images), np.array(labels)

${config.augmentation ? `
# ────────────────────────────────────────────────
# Augmentasi Data
# ────────────────────────────────────────────────
def get_augmentation_generator():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
` : '# Augmentasi: Dinonaktifkan'}

# ────────────────────────────────────────────────
# Arsitektur Model CNN
# ────────────────────────────────────────────────
${architectureCode}

# ────────────────────────────────────────────────
# Kompilasi Model
# ────────────────────────────────────────────────
def compile_model(model):
    ${optimizerCode}
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ────────────────────────────────────────────────
# Visualisasi Confusion Matrix
# ────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, fold=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, linecolor='gray'
    )
    title = f'Confusion Matrix' + (f' – Fold {fold}' if fold else '')
    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Prediksi', fontsize=12)
    plt.ylabel('Aktual', fontsize=12)
    plt.tight_layout()
    fname = f'confusion_matrix_fold{fold}.png' if fold else 'confusion_matrix_final.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"[INFO] Confusion matrix disimpan: {fname}")

# ────────────────────────────────────────────────
# Visualisasi Riwayat Training
# ────────────────────────────────────────────────
def plot_training_history(history, fold=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    suffix = f' – Fold {fold}' if fold else ''

    # Akurasi
    axes[0].plot(history.history['accuracy'],     label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validasi', linewidth=2, linestyle='--')
    axes[0].set_title(f'Akurasi Model{suffix}', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Akurasi')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validasi', linewidth=2, linestyle='--')
    axes[1].set_title(f'Loss Model{suffix}', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fname = f'training_history_fold{fold}.png' if fold else 'training_history.png'
    plt.savefig(fname, dpi=150)
    plt.show()

# ────────────────────────────────────────────────
# K-Fold Cross Validation
# ────────────────────────────────────────────────
def run_kfold_cross_validation(X, y):
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    print(f"\\n{'='*60}")
    print(f"  Memulai {K_FOLDS}-Fold Cross Validation")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\\n[Fold {fold}/{K_FOLDS}] ─────────────────────────────")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model()
        model = compile_model(model)
        ${config.augmentation ? `
        aug_gen = get_augmentation_generator()
        train_gen = aug_gen.flow(X_train, y_train, batch_size=BATCH_SIZE)
        history = model.fit(
            train_gen,
            steps_per_epoch=len(X_train) // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            verbose=1
        )` : `
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            verbose=1
        )`}

        y_pred_proba = model.predict(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)

        acc  = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        fold_results.append({
            'fold': fold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        })

        print(f"  Akurasi   : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")

        plot_confusion_matrix(y_val, y_pred, fold=fold)
        plot_training_history(history, fold=fold)

        model.save(f'model_fold_{fold}.h5')

    # Ringkasan K-Fold
    print(f"\\n{'='*60}")
    print("  Ringkasan K-Fold Cross Validation")
    print(f"{'='*60}")
    accs = [r['accuracy']  for r in fold_results]
    prcs = [r['precision'] for r in fold_results]
    recs = [r['recall']    for r in fold_results]
    f1s  = [r['f1_score']  for r in fold_results]

    print(f"  Akurasi   : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Precision : {np.mean(prcs):.4f} ± {np.std(prcs):.4f}")
    print(f"  Recall    : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"  F1-Score  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    plot_kfold_summary(fold_results)
    return fold_results

# ────────────────────────────────────────────────
# Visualisasi Ringkasan K-Fold
# ────────────────────────────────────────────────
def plot_kfold_summary(fold_results):
    folds     = [r['fold']      for r in fold_results]
    accs      = [r['accuracy']  for r in fold_results]
    precs     = [r['precision'] for r in fold_results]
    recs      = [r['recall']    for r in fold_results]
    f1s       = [r['f1_score']  for r in fold_results]

    x = np.arange(len(folds))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, accs,  width, label='Akurasi',   color='#2196F3', alpha=0.85)
    ax.bar(x - 0.5*width, precs, width, label='Precision', color='#4CAF50', alpha=0.85)
    ax.bar(x + 0.5*width, recs,  width, label='Recall',    color='#FF9800', alpha=0.85)
    ax.bar(x + 1.5*width, f1s,   width, label='F1-Score',  color='#F44336', alpha=0.85)

    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Skor', fontsize=12)
    ax.set_title(f'Perbandingan Metrik per Fold ({K_FOLDS}-Fold CV)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('kfold_summary.png', dpi=150)
    plt.show()
    print("[INFO] Ringkasan K-Fold disimpan: kfold_summary.png")

# ────────────────────────────────────────────────
# Prediksi Gambar Tunggal
# ────────────────────────────────────────────────
def predict_image(model, img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    print(f"\\n[PREDIKSI] Gambar : {img_path}")
    print(f"           Kelas  : {predicted_class}")
    print(f"           Confidence : {confidence:.2f}%")
    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, predictions)):
        print(f"           {cls}: {prob*100:.2f}%")
    return predicted_class, confidence

# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
if __name__ == '__main__':
    DATA_DIR = './dataset'   # ganti dengan path dataset Anda

    print("\\n[INFO] Memuat dataset...")
    X, y = load_dataset(DATA_DIR)

    if len(X) == 0:
        print("[ERROR] Dataset kosong. Pastikan struktur direktori sudah benar.")
        exit(1)

    print(f"[INFO] Total gambar : {len(X)}")
    print(f"[INFO] Distribusi kelas:")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(y == i)
        print(f"       {name}: {count} gambar ({count/len(y)*100:.1f}%)")

    print("\\n[INFO] Memulai K-Fold Cross Validation...")
    results = run_kfold_cross_validation(X, y)

    print("\\n[INFO] Melatih model final pada seluruh data...")
    final_model = build_model()
    final_model = compile_model(final_model)
    final_model.summary()
    ${config.augmentation ? `
    aug_gen = get_augmentation_generator()
    train_gen = aug_gen.flow(X, y, batch_size=BATCH_SIZE)
    history = final_model.fit(
        train_gen,
        steps_per_epoch=len(X) // BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1
    )` : `
    history = final_model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )`}

    final_model.save('model_klasifikasi_sampah_final.h5')
    print("\\n[INFO] Model final disimpan: model_klasifikasi_sampah_final.h5")
    print("[INFO] Selesai!")
`;
}

function getArchitectureCode(config: ModelConfig): string {
  if (config.architecture === 'custom') {
    return `def build_model():
    model = keras.Sequential([
        # Blok Konvolusi 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=INPUT_SHAPE),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(${config.dropoutRate * 0.5}),

        # Blok Konvolusi 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(${config.dropoutRate * 0.5}),

        # Blok Konvolusi 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(${config.dropoutRate * 0.5}),

        # Fully Connected
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(${config.dropoutRate}),
        layers.Dense(128, activation='relu'),
        layers.Dropout(${config.dropoutRate * 0.5}),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model`;
  }

  if (config.architecture === 'mobilenet') {
    return `def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Fine-tuning: set True untuk unfreeze layer

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(${config.dropoutRate}),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model`;
  }

  if (config.architecture === 'resnet50') {
    return `def build_model():
    base_model = tf.keras.applications.ResNet50(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Fine-tuning: set True untuk unfreeze layer

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(${config.dropoutRate})(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model`;
  }

  // VGG16
  return `def build_model():
    base_model = tf.keras.applications.VGG16(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Fine-tuning: set True untuk unfreeze layer

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(${config.dropoutRate})(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(${config.dropoutRate * 0.5})(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model`;
}
