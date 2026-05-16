import os
import sys
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Memaksa terminal Windows menggunakan UTF-8 agar tidak error 'charmap'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model_klasifikasi_sampah_final.h5"
CLASS_NAMES = ['Organik', 'Non-organik']
model = None

# Variable global untuk menyimpan progress training real-time
training_stats = {
    "metrics": [],
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "cm_train": None,
    "cm_val": None,
    "kfold_results": []
}

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_stats
    training_stats = {
        "metrics": [],
        "is_training": True,
        "current_epoch": 0,
        "total_epochs": request.json.get("total_epochs", 0),
        "cm_train": None,
        "cm_val": None,
        "kfold_results": []
    }
    return jsonify({"status": "training_started"})

@app.route('/update_metrics', methods=['POST'])
def update_metrics():
    global training_stats
    data = request.json
    training_stats["metrics"].append(data["metric"])
    training_stats["current_epoch"] = data["epoch"]
    training_stats["total_epochs"] = data.get("total_epochs", training_stats["total_epochs"])
    training_stats["is_training"] = True
    return jsonify({"status": "success"})

@app.route('/finish_training', methods=['POST'])
def finish_training():
    global training_stats
    data = request.json
    training_stats["is_training"] = False
    if data:
        training_stats["cm_train"] = data.get("cm_train")
        training_stats["cm_val"] = data.get("cm_val")
        training_stats["kfold_results"] = data.get("kfold_results", [])
    
    # Muat ulang model setelah training selesai
    load_trained_model()
    return jsonify({"status": "model_reloaded"})

@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    return jsonify(training_stats)

def load_trained_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[INFO] Model berhasil dimuat dari {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Gagal memuat model: {e}")
    else:
        print(f"[WARN] File model {MODEL_PATH} tidak ditemukan. Silakan jalankan training terlebih dahulu.")

# Muat model saat server dinyalakan
load_trained_model()

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'model_trained': os.path.exists(MODEL_PATH),
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model belum siap atau belum dilatih'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang dikirim'}), 400

    file = request.files['image']
    try:
        # Baca gambar dan preprocess
        img_bytes = file.read()
        
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        # Gunakan preprocess_input yang sama dengan saat training
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        predictions = model.predict(img_array)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Siapkan hasil detail (semua kelas)
        results_detail = {}
        for i, name in enumerate(CLASS_NAMES):
            results_detail[name] = float(predictions[0][i])

        return jsonify({
            'label': CLASS_NAMES[predicted_class_idx],
            'confidence': confidence,
            'detail': results_detail
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)