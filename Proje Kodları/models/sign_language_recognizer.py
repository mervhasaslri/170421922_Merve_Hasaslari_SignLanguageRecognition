import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import logging
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, Flatten
from tensorflow.keras import regularizers
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tf.keras.utils.register_keras_serializable()
class FixedLSTM(tf.keras.layers.LSTM):
    @classmethod
    def from_config(cls, config):
        if 'time_major' in config:
            del config['time_major']
        return super().from_config(config)

@tf.keras.utils.register_keras_serializable()
class FixedCategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):
    @classmethod
    def from_config(cls, config):
        if 'reduction' in config and config['reduction'] == 'auto':
            config['reduction'] = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        return super().from_config(config)

class SignLanguageRecognizer:
    """
    İşaret dili tanıma için hibrit CNN-LSTM modeli.
    """
    
    def __init__(self, config):
        # Only require these paths if we're doing training
        if 'train_landmarks_dir' in config:
            self.train_landmarks_dir = Path(config['train_landmarks_dir'])
            self.val_landmarks_dir = Path(config['val_landmarks_dir'])
            self.test_landmarks_dir = Path(config['test_landmarks_dir'])
            self.train_labels = pd.read_csv(config['train_labels_path'])
            self.val_labels = pd.read_csv(config['val_labels_path'])
            self.test_labels = pd.read_csv(config['test_labels_path'])
        else:
            # Real-time prediction mode
            self.train_landmarks_dir = None
            self.val_landmarks_dir = None
            self.test_landmarks_dir = None
            self.train_labels = None
            self.val_labels = None
            self.test_labels = None
        
        self.class_map = pd.read_csv(config['class_map_path'])
        self.num_classes = len(self.class_map)
        
        # Model parametreleri
        self.sequence_length = config.get('sequence_length', 30)
        self.frame_features = config.get('frame_features', 195)  # (21 el noktası * 3 + 33 poz noktası * 4)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        self.learning_rate = config.get('learning_rate', 0.0003)
        
        # Model
        self.model = None
        
    def load_landmarks_data(self, video_name, data_type='train'):
        """JSON dosyasından landmark verilerini yükle"""
        try:
            if data_type == 'train':
                landmarks_dir = self.train_landmarks_dir
            elif data_type == 'val':
                landmarks_dir = self.val_landmarks_dir
            else:  # test
                landmarks_dir = self.test_landmarks_dir
                
            json_path = landmarks_dir / f"{video_name}_color_landmarks.json"
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading landmarks for {video_name}: {str(e)}")
            return None

    def prepare_sequence_data(self, landmarks_data, video_name=None, class_id=None, augment=True, low_sample_classes=None):
        """Landmark verilerini model için uygun formata dönüştür ve veri augmentasyonu uygula"""
        if not landmarks_data:
            return None
            
        frames = list(landmarks_data['frames'].values())
        frame_keys = list(landmarks_data['frames'].keys())

        # Eksik frame sayısı %30'dan fazla ise videoyu atla
        if augment:
            valid_frames = [f for f in frames if f['hands'] and f['pose']]
            if len(valid_frames) < self.sequence_length * 0.7:
                return None  # Yetersiz bilgi, atla

        # Sabit uzunlukta sekans oluştur
        if len(frames) > self.sequence_length:
            indices = np.linspace(0, len(frames)-1, self.sequence_length, dtype=int)
            frames = [frames[i] for i in indices]
            frame_keys = [frame_keys[i] for i in indices]
        else:
            last_frame = frames[-1] if frames else {'hands': [], 'pose': None}
            last_key = frame_keys[-1] if frame_keys else '00000'
            frames.extend([last_frame] * (self.sequence_length - len(frames)))
            frame_keys.extend([last_key] * (self.sequence_length - len(frame_keys)))

        is_low_sample = (low_sample_classes is not None and class_id in low_sample_classes)
        noise_level = 0.02 * (2 if is_low_sample else 1)
        temporal_shift_prob = 0.8 if is_low_sample else 0.5
        frame_zero_prob = 0.6 if is_low_sample else 0.3

        # Temporal shift augmentasyonu
        if augment and np.random.random() < temporal_shift_prob:
            shift = np.random.randint(-2, 3)
            if shift > 0:
                frames = frames[shift:] + frames[:shift]
                frame_keys = frame_keys[shift:] + frame_keys[:shift]
            elif shift < 0:
                frames = frames[shift:] + frames[:shift]
                frame_keys = frame_keys[shift:] + frame_keys[:shift]

        sequence = []
        for i, frame in enumerate(frames):
            features = []
            # El landmarklarını ekle
            if frame['hands']:
                hand = frame['hands'][0]
                for landmark in hand:
                    coords = [landmark['x'], landmark['y'], landmark['z']]
                    if augment and np.random.random() < 0.5:
                        coords = [c + np.random.normal(0, noise_level) for c in coords]
                    features.extend(coords)
            else:
                features.extend([0.0] * (21 * 3))
            # Poz landmarklarını ekle
            if frame['pose']:
                for landmark in frame['pose']:
                    coords = [landmark['x'], landmark['y'], landmark['z'], landmark['visibility']]
                    if augment and np.random.random() < 0.5:
                        coords = [c + np.random.normal(0, noise_level) if i < 3 else c for i, c in enumerate(coords)]
                    features.extend(coords)
            else:
                features.extend([0.0] * (33 * 4))
            sequence.append(features)

        # Eksik frame simülasyonu (augmentation aktifse)
        if augment and np.random.random() < frame_zero_prob:
            num_frames_to_zero = np.random.randint(1, 4)
            frames_to_zero = np.random.choice(len(sequence), num_frames_to_zero, replace=False)
            for idx in frames_to_zero:
                sequence[idx] = [0.0] * len(sequence[idx])

        return np.array(sequence)

    def prepare_dataset(self):
        """Eğitim verilerini hazırla (sadece landmark)"""
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []
        low_sample_classes = self.get_low_sample_classes(threshold=0.5)
        # Eğitim verileri
        for _, row in self.train_labels.iterrows():
            video_name = row[0]
            class_id = row[1]
            landmarks_data = self.load_landmarks_data(video_name, 'train')
            if landmarks_data is None:
                continue
            sequence = self.prepare_sequence_data(landmarks_data, video_name=video_name, class_id=class_id, augment=True, low_sample_classes=low_sample_classes)
            if sequence is None:
                continue
            X_train.append(sequence)
            y_train.append(class_id)
        # Validation verileri
        for _, row in self.val_labels.iterrows():
            video_name = row[0]
            class_id = row[1]
            landmarks_data = self.load_landmarks_data(video_name, 'val')
            if landmarks_data is None:
                continue
            sequence = self.prepare_sequence_data(landmarks_data, video_name=video_name, class_id=class_id, augment=False)
            if sequence is None:
                continue
            X_val.append(sequence)
            y_val.append(class_id)
        # Test verileri
        for _, row in self.test_labels.iterrows():
            video_name = row[0]
            class_id = row[1]
            landmarks_data = self.load_landmarks_data(video_name, 'test')
            if landmarks_data is None:
                continue
            sequence = self.prepare_sequence_data(landmarks_data, video_name=video_name, class_id=class_id, augment=False)
            if sequence is None:
                continue
            X_test.append(sequence)
            y_test.append(class_id)
        X_train = np.array(X_train)
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        X_val = np.array(X_val)
        y_val = to_categorical(y_val, num_classes=self.num_classes)
        X_test = np.array(X_test)
        y_test = to_categorical(y_test, num_classes=self.num_classes)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_model(self):
        """Model: Landmark + RGB (Conv2D) fusion, sequence modeling, classification"""
        input_shape_landmark = (self.sequence_length, self.frame_features)
        input_shape_rgb = (self.sequence_length, 64, 64, 3)

        # Landmark input
        landmark_input = tf.keras.Input(shape=input_shape_landmark, name="landmark_input")
        # RGB input
        rgb_input = tf.keras.Input(shape=input_shape_rgb, name="rgb_input")

        # Landmark feature projection
        x_lm = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu'))(landmark_input)
        x_lm = tf.keras.layers.BatchNormalization()(x_lm)

        # RGB feature extraction (her frame için Conv2D)
        def rgb_cnn_block():
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(2),
                tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(2),
                tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(64, activation='relu')
            ])
            return model
        x_rgb = tf.keras.layers.TimeDistributed(rgb_cnn_block())(rgb_input)
        x_rgb = tf.keras.layers.BatchNormalization()(x_rgb)

        # Landmark ve RGB feature'larını birleştir
        x = tf.keras.layers.Concatenate(axis=-1)([x_lm, x_rgb])  # (batch, seq, 128)
        x = tf.keras.layers.Dropout(0.3)(x)

        # Sequence modeling (ör: BiLSTM)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # Temporal pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # Dense + output
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=[landmark_input, rgb_input], outputs=outputs)
        self.model.summary()
        return self.model

    def train(self, save_dir="models"):
        """Modeli eğit ve kaydet"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Veriyi hazırla
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_dataset()
        
        # Modeli oluştur
        self.model = self.create_model()
        
        # Modeli derle
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        
        # Learning rate scheduler fonksiyonu
        def exp_decay(epoch):
            initial_lr = self.learning_rate
            decay_rate = 0.85  # Daha agresif decay
            decay_steps = 4    # Daha sık decay
            
            return initial_lr * (decay_rate ** (epoch // decay_steps))
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                min_delta=0.001
            ),
            tf.keras.callbacks.LearningRateScheduler(
                exp_decay,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=5,  # Daha erken müdahale
                min_lr=0.000001,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                save_dir / 'best_model.h5',
                monitor='val_loss',  # val_accuracy yerine val_loss'u izle
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                save_dir / 'training_history.csv'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=save_dir / 'logs',
                histogram_freq=1,
                update_freq='epoch'
            )
        ]
        
        # Sınıf ağırlıklarını hesapla
        class_weights = {}
        total_samples = len(y_train)
        n_classes = self.num_classes
        
        # Her sınıf için weight hesapla
        for i in range(n_classes):
            n_samples = np.sum(y_train[:, i])
            if n_samples > 0:
                class_weights[i] = (1 / n_samples) * (total_samples / 2.0)
            else:
                class_weights[i] = 1.0
                
        # Ağırlıkları normalize et
        max_weight = max(class_weights.values())
        for i in class_weights:
            class_weights[i] = class_weights[i] / max_weight
        
        # Eğitim
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        # Eğitim sonuçlarını kaydet
        pd.DataFrame(history.history).to_csv(save_dir / 'training_metrics.csv')
        
        # Test verisi üzerinde değerlendirme
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Test verisi üzerinde tahminler
        test_predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(test_predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Confusion Matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        report = classification_report(true_classes, predicted_classes)
        
        # ROC eğrileri
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], test_predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Sınıf bazlı doğruluk oranları
        class_accuracy = {}
        for class_id in range(self.num_classes):
            class_mask = true_classes == class_id
            if np.sum(class_mask) > 0:
                class_acc = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
                class_name = self.class_map.iloc[class_id]['TR']
                class_accuracy[class_name] = class_acc
        
        # Performans metriklerini kaydet
        performance_metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'class_accuracy': class_accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'roc_auc': {str(k): float(v) for k, v in roc_auc.items()}
        }
        
        with open(save_dir / 'test_performance.json', 'w', encoding='utf-8') as f:
            json.dump(performance_metrics, f, ensure_ascii=False, indent=4)
        
        # En iyi ve en kötü performans gösteren sınıfları bul
        best_class = max(class_accuracy.items(), key=lambda x: x[1])
        worst_class = min(class_accuracy.items(), key=lambda x: x[1])
        
        logger.info(f"Best performing class: {best_class[0]} (Accuracy: {best_class[1]:.4f})")
        logger.info(f"Worst performing class: {worst_class[0]} (Accuracy: {worst_class[1]:.4f})")
        logger.info("\nClassification Report:")
        logger.info(report)
        
        return history

    def predict(self):
        if len(self.landmark_buffer) < self.sequence_length:
            print("Yetersiz frame, tahmin yapılmadı.")
            return None

        sequence = np.array(self.landmark_buffer)
        dummy_rgb = np.zeros((1, self.sequence_length, 64, 64, 3), dtype=np.float32)
        predictions = self.model.predict([np.expand_dims(sequence, axis=0), dummy_rgb])
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        print(f"Model tahmini: {predicted_class}, Güven: {confidence:.2f}")

        class_info = self.class_map.iloc[predicted_class]
        if confidence < self.confidence_threshold:
            print(f"Güven eşiği ({self.confidence_threshold}) altında!")
            return None
        now = datetime.now().strftime('%H:%M:%S')
        print(f"Tahmin edilen işaret: {class_info['TR']} ({class_info['EN']}) - Güven: {confidence:.2f} - Saat: {now}")
        return {
            'class_id': predicted_class,
            'confidence': float(confidence),
            'tr_label': class_info['TR'],
            'en_label': class_info['EN']
        }

    def load_model(self, model_path):
        """Yüksek ağırlıklı modeli yükle"""
        logger.info(f"Yüksek ağırlıklı model yükleniyor: {model_path}")
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'LSTM': FixedLSTM,
                'CategoricalCrossentropy': FixedCategoricalCrossentropy
            }
        )
        logger.info(f"Model loaded from {model_path}") 