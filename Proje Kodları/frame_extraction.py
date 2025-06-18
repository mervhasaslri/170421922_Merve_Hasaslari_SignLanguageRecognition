import cv2
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frame_extraction.log'),
        logging.StreamHandler()
    ]
)

class VideoProcessor:
    def __init__(self, input_dir='archive/cleaned_train', output_dir='frames', target_fps=30, frame_size=(224, 224), 
                 save_processed_video=False, jpg_quality=80):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_fps = target_fps
        self.frame_size = frame_size
        self.save_processed_video = save_processed_video
        self.jpg_quality = jpg_quality
        
        # Sadece frames klasörünü oluştur
        self.frames_dir = os.path.join(output_dir, 'frames')
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
        
        # Eğer processed_videos isteniyorsa o klasörü de oluştur
        if save_processed_video:
            self.processed_videos_dir = os.path.join(output_dir, 'processed_videos')
            if not os.path.exists(self.processed_videos_dir):
                os.makedirs(self.processed_videos_dir)
    
    def process_video(self, video_file):
        """Tek bir videoyu işler: FPS standardizasyonu ve frame extraction yapar."""
        try:
            video_path = os.path.join(self.input_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            
            # Video frame'lerini kaydetmek için klasör oluştur
            video_frames_dir = os.path.join(self.frames_dir, video_name)
            if not os.path.exists(video_frames_dir):
                os.makedirs(video_frames_dir)
            
            # Videoyu aç
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Video açılamadı: {video_file}")
                return False
            
            # Video özellikleri
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # FPS standardizasyonu için frame atlama/tekrarlama oranı
            fps_ratio = original_fps / self.target_fps
            
            # İşlenmiş video writer'ı (eğer isteniyorsa)
            out = None
            if self.save_processed_video:
                processed_video_path = os.path.join(self.processed_videos_dir, f"{video_name}_processed.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(processed_video_path, fourcc, self.target_fps, self.frame_size)
            
            frame_idx = 0
            saved_frames = 0
            
            with tqdm(total=frame_count, desc=f"İşleniyor: {video_file}", leave=False) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # FPS standardizasyonu için frame seçimi
                    if frame_idx % fps_ratio < 1.0:
                        # Frame'i yeniden boyutlandır
                        resized_frame = cv2.resize(frame, self.frame_size)
                        
                        # Frame'i kaydet (sıkıştırılmış JPG olarak)
                        frame_path = os.path.join(video_frames_dir, f"frame_{saved_frames:05d}.jpg")
                        cv2.imwrite(frame_path, resized_frame, 
                                  [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality])
                        
                        # İşlenmiş videoya frame'i ekle (eğer isteniyorsa)
                        if out is not None:
                            out.write(resized_frame)
                        
                        saved_frames += 1
                    
                    frame_idx += 1
                    pbar.update(1)
            
            cap.release()
            if out is not None:
                out.release()
            
            logging.info(f"Video işlendi: {video_file} - Kaydedilen frame sayısı: {saved_frames}")
            return True
            
        except Exception as e:
            logging.error(f"Hata oluştu ({video_file}): {str(e)}")
            return False
    
    def process_all_videos(self, max_workers=4):
        """Tüm videoları paralel olarak işler."""
        video_files = [f for f in os.listdir(self.input_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        total_videos = len(video_files)
        logging.info(f"Toplam {total_videos} video işlenecek")
        
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_video, video_file): video_file 
                      for video_file in video_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Toplam İlerleme"):
                video_file = futures[future]
                try:
                    if future.result():
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    logging.error(f"Video işlenirken hata oluştu ({video_file}): {str(e)}")
                    failed += 1
        
        logging.info(f"""
        İşlem tamamlandı:
        - Başarılı: {successful}
        - Başarısız: {failed}
        - Toplam: {total_videos}
        """)

if __name__ == "__main__":
    # Parametreler
    INPUT_DIR = "archive/cleaned_train"
    OUTPUT_DIR = "archive/processed_train"
    TARGET_FPS = 30
    FRAME_SIZE = (224, 224)
    MAX_WORKERS = 4
    SAVE_PROCESSED_VIDEO = False  # İşlenmiş videoları kaydetme
    JPG_QUALITY = 80  # JPG sıkıştırma kalitesi (0-100)
    
    processor = VideoProcessor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        target_fps=TARGET_FPS,
        frame_size=FRAME_SIZE,
        save_processed_video=SAVE_PROCESSED_VIDEO,
        jpg_quality=JPG_QUALITY
    )
    
    print(f"""
    Video İşleme Başlatılıyor:
    - Giriş klasörü: {INPUT_DIR}
    - Çıkış klasörü: {OUTPUT_DIR}
    - Hedef FPS: {TARGET_FPS}
    - Frame boyutu: {FRAME_SIZE}
    - Paralel işlem sayısı: {MAX_WORKERS}
    - İşlenmiş video kaydı: {'Evet' if SAVE_PROCESSED_VIDEO else 'Hayır'}
    - JPG kalitesi: {JPG_QUALITY}
    """)
    
    response = input("İşleme devam etmek istiyor musunuz? (e/h): ")
    if response.lower() == 'e':
        processor.process_all_videos(max_workers=MAX_WORKERS)
    else:
        print("İşlem iptal edildi.") 