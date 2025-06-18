import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from pathlib import Path
import json
import pandas as pd
from models.sign_language_recognizer import SignLanguageRecognizer
from datetime import datetime

class RealTimeSignLanguageRecognizer:
    def __init__(self, model_path, class_map_path, sequence_length=30, confidence_threshold=0.5):
        """
        Gerçek zamanlı işaret dili tanıma sınıfı
        
        Args:
            model_path: Eğitilmiş model dosyası
            class_map_path: Sınıf eşleştirmelerinin bulunduğu CSV dosyası
            sequence_length: İşlenecek frame sayısı
            confidence_threshold: Tahmin için minimum güven eşiği
        """
        # MediaPipe kurulumu
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Model ve sınıf haritası yükleme
        self.recognizer = SignLanguageRecognizer({
            'train_landmarks_dir': 'archive/processed_train/landmarks',
            'val_landmarks_dir': 'archive/processed_val/landmarks',
            'test_landmarks_dir': 'archive/processed_test/landmarks',
            'train_labels_path': 'archive/train_labels.csv',
            'val_labels_path': 'archive/val_labels.csv',
            'test_labels_path': 'archive/test_labels.csv',
            'class_map_path': 'archive/SignList_ClassId_TR_EN.csv',
        })
        self.recognizer.load_model(model_path)
        # Model özetini konsola yazdır
        print("\n--- Model Summary ---")
        self.recognizer.model.summary()
        print("---------------------\n")
        self.class_map = pd.read_csv(class_map_path)
        
        # Parametreler
        self.sequence_length = sequence_length
        self.landmark_buffer = []
        self.prediction_interval = 5  # Her 5 frame'de bir tahmin yap
        self.frame_count = 0
        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_cooldown = 1.0  # Tahminler arası minimum süre (saniye)
        self.confidence_threshold = confidence_threshold  # <-- Eşik eklendi
        
    def extract_landmarks(self, frame):
        """Frame'den el ve vücut landmarklarını çıkar"""
        # BGR'dan RGB'ye dönüştür
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Landmark çıkarma
        hands_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        
        # Landmark verilerini topla
        landmarks_data = {
            'hands': [],
            'pose': None
        }
        
        # El landmarklarını ekle
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                landmarks_data['hands'].append(hand_data)
        
        # Vücut landmarklarını ekle
        if pose_results.pose_landmarks:
            pose_data = []
            for landmark in pose_results.pose_landmarks.landmark:
                pose_data.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            landmarks_data['pose'] = pose_data
        
        return landmarks_data
    
    def prepare_sequence(self, landmarks_data):
        """Landmark verilerini model için uygun formata dönüştür"""
        features = []
        
        # El landmarklarını ekle
        if landmarks_data['hands']:
            hand = landmarks_data['hands'][0]
            for landmark in hand:
                features.extend([landmark['x'], landmark['y'], landmark['z']])
        else:
            # El landmarkları yoksa sıfırlarla doldur
            features.extend([0.0] * (21 * 3))  # 21 el noktası * 3 koordinat
        
        # Poz landmarklarını ekle
        if landmarks_data['pose']:
            for landmark in landmarks_data['pose']:
                features.extend([landmark['x'], landmark['y'], landmark['z'], landmark['visibility']])
        else:
            # Poz landmarkları yoksa sıfırlarla doldur
            features.extend([0.0] * (33 * 4))  # 33 poz noktası * 4 özellik
        
        return features
    
    def predict(self):
        """Landmark buffer'dan tahmin yap"""
        if len(self.landmark_buffer) < self.sequence_length:
            return None
        
        # Buffer'ı numpy dizisine dönüştür
        sequence = np.array(self.landmark_buffer)
        
        # Tahmin yap
        predictions = self.recognizer.model.predict(np.expand_dims(sequence, axis=0))
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        class_info = self.class_map.iloc[predicted_class]
        # Güven eşiği kontrolü
        if confidence < self.confidence_threshold:
            return None
        # Konsola tahmin edilen işareti, güven değeri ve saatini yazdır
        now = datetime.now().strftime('%H:%M:%S')
        print(f"Tahmin edilen işaret: {class_info['TR']} ({class_info['EN']}) - Güven: {confidence:.2f} - Saat: {now}")
        return {
            'class_id': predicted_class,
            'confidence': float(confidence),
            'tr_label': class_info['TR'],
            'en_label': class_info['EN']
        }
    
    def draw_landmarks(self, frame, landmarks_data):
        """Frame üzerine landmarkları çiz"""
        # El landmarklarını çiz
        if landmarks_data['hands']:
            for hand_landmarks in landmarks_data['hands']:
                for landmark in hand_landmarks:
                    x = int(landmark['x'] * frame.shape[1])
                    y = int(landmark['y'] * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Vücut landmarklarını çiz
        if landmarks_data['pose']:
            for landmark in landmarks_data['pose']:
                x = int(landmark['x'] * frame.shape[1])
                y = int(landmark['y'] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        
        return frame
    
    def draw_prediction(self, frame, prediction):
        """Frame üzerine tahmin sonucunu çiz"""
        if prediction:
            text = f"{prediction['tr_label']} ({prediction['en_label']}) - {prediction['confidence']:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame
    
    def run(self):
        """Gerçek zamanlı işaret dili tanıma döngüsü"""
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Kamera görüntüsü alınamadı")
                break
            
            # Frame'i yatay olarak çevir (ayna görüntüsü)
            frame = cv2.flip(frame, 1)
            
            # Landmark çıkarma
            landmarks_data = self.extract_landmarks(frame)
            
            # Landmark buffer'a ekle
            features = self.prepare_sequence(landmarks_data)
            self.landmark_buffer.append(features)
            
            # Buffer boyutunu kontrol et
            if len(self.landmark_buffer) > self.sequence_length:
                self.landmark_buffer.pop(0)
            
            # Belirli aralıklarla tahmin yap
            current_time = time.time()
            if (self.frame_count % self.prediction_interval == 0 and 
                len(self.landmark_buffer) == self.sequence_length and
                current_time - self.last_prediction_time > self.prediction_cooldown):
                
                prediction = self.predict()
                if prediction and prediction['confidence'] > 0.7:  # Güven eşiği
                    self.last_prediction = prediction
                    self.last_prediction_time = current_time
            
            # Landmarkları ve tahmin sonucunu çiz
            frame = self.draw_landmarks(frame, landmarks_data)
            frame = self.draw_prediction(frame, self.last_prediction)
            
            # Frame'i göster
            cv2.imshow('İşaret Dili Tanıma', frame)
            
            # Frame sayacını artır
            self.frame_count += 1
            
            # Çıkış için 'q' tuşuna basılmasını bekle
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Kaynakları serbest bırak
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.pose.close()

if __name__ == "__main__":
    # Model ve sınıf haritası yolları
    model_path = "models/sign_language/best_model.h5"
    class_map_path = "archive/SignList_ClassId_TR_EN.csv"
    
    # Gerçek zamanlı tanıyıcıyı başlat
    recognizer = RealTimeSignLanguageRecognizer(model_path, class_map_path, confidence_threshold=0.5)
    recognizer.run() 