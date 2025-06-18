import tensorflow as tf
from models.sign_language_recognizer import SignLanguageRecognizer

def main():
    # GPU kullanımını kontrol et ve yapılandır
    print("\nGPU Bilgileri:")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("GPU Device Name:", tf.test.gpu_device_name())
    print("Is GPU available:", tf.test.is_built_with_cuda())
    
    # Metal API ile GPU bellek ayarlarını optimize et (bu satırı kaldırabiliriz)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Memory growth yönetimi kaldırıldı
            print("Metal API GPU kullanıma hazır.")
        except RuntimeError as e:
            print(e)
    else:
        print("GPU bulunamadı, CPU kullanılacak")
    
    print("\nEğitim başlıyor...\n")
    
    # Model konfigürasyonu
    config = {
        'train_landmarks_dir': 'archive/processed_train/landmarks',
        'val_landmarks_dir': 'archive/processed_val/landmarks',
        'test_landmarks_dir': 'archive/processed_test/landmarks',
        'train_labels_path': 'archive/train_labels.csv',
        'val_labels_path': 'archive/val_labels.csv',
        'test_labels_path': 'archive/test_labels.csv',
        'class_map_path': 'archive/SignList_ClassId_TR_EN.csv',
        'sequence_length': 30,  # Giriş verisi uzunluğu
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.0003
    }
    
    # Model nesnesini oluştur
    recognizer = SignLanguageRecognizer(config)
    
    # Modeli eğit
    history = recognizer.train(save_dir='models/sign_language')
    
    # Eğitim sırasında sonuçları takip edebiliriz
    print("\nEğitim Sonuçları:")
    print(f"Training Loss: {history.history['loss'][-1]}")
    print(f"Training Accuracy: {history.history['accuracy'][-1]}")
    
    # Test için örnek tahmin
    print("\nTest Sonuçları:")
    result = recognizer.predict('signer0_sample1')  # Burada doğru bir test örneği kullandığınızdan emin olun
    print(f"Predicted sign: {result['tr_label']} ({result['en_label']})")
    print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    main()
