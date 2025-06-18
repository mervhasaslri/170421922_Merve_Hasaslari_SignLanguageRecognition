from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import threading
from real_time_sign_language import RealTimeSignLanguageRecognizer

app = Flask(__name__)

# Model yolları
MODEL_PATH = "models/sign_language/best_model.h5"
CLASS_MAP_PATH = "archive/SignList_ClassId_TR_EN.csv"

# Global tanıyıcı
recognizer = None
current_prediction = None

def initialize_recognizer():
    global recognizer
    recognizer = RealTimeSignLanguageRecognizer(MODEL_PATH, CLASS_MAP_PATH)
    print("İşaret dili tanıyıcı başlatıldı")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global current_prediction
    
    if recognizer is None:
        return jsonify({'error': 'Tanıyıcı başlatılmadı'}), 500
    
    try:
        # Gelen görüntüyü al
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # data:image/jpeg;base64, kısmını atla
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Görüntüyü işle
        landmarks_data = recognizer.extract_landmarks(frame)
        features = recognizer.prepare_sequence(landmarks_data)
        
        # Buffer'a ekle
        recognizer.landmark_buffer.append(features)
        if len(recognizer.landmark_buffer) > recognizer.sequence_length:
            recognizer.landmark_buffer.pop(0)
        
        # Tahmin yap
        if len(recognizer.landmark_buffer) == recognizer.sequence_length:
            prediction = recognizer.predict()
            if prediction:
                current_prediction = prediction
        
        return jsonify({'status': 'success'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    if current_prediction:
        return jsonify(current_prediction)
    return jsonify({'error': 'Henüz tahmin yapılmadı'}), 404

if __name__ == '__main__':
    # Tanıyıcıyı ayrı bir thread'de başlat
    threading.Thread(target=initialize_recognizer, daemon=True).start()
    app.run(host='0.0.0.0', port=5001, threaded=True)