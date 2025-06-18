import cv2
import numpy as np
import os
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction.log'),
        logging.StreamHandler()
    ]
)

class VideoExtractor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Sadece Depth için çıktı klasörü
        self.depth_dir = os.path.join(output_dir, 'depth')
        if not os.path.exists(self.depth_dir):
            os.makedirs(self.depth_dir)
    
    def process_video(self, video_name):
        """Video frame'lerini grayscale olarak ayırır."""
        try:
            video_path = os.path.join(self.input_dir, video_name)
            base_name = os.path.splitext(video_name)[0]
            
            # Depth için klasör
            depth_video_dir = os.path.join(self.depth_dir, base_name)
            if not os.path.exists(depth_video_dir):
                os.makedirs(depth_video_dir)
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            
            with tqdm(total=total_frames, desc=f"İşleniyor: {video_name}", leave=False) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Grayscale'e çevir ve kaydet
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    depth_path = os.path.join(depth_video_dir, f"depth_{frame_count:05d}.png")
                    cv2.imwrite(depth_path, gray)
                    
                    frame_count += 1
                    pbar.update(1)
            
            cap.release()
            logging.info(f"Video işlendi: {video_name} - Kaydedilen frame sayısı: {frame_count}")
            return True
            
        except Exception as e:
            logging.error(f"Hata oluştu ({video_name}): {str(e)}")
            return False
    
    def process_all_videos(self, max_workers=4):
        """Tüm videoları işler."""
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
        - İşlenen: {total_videos}
        """)

if __name__ == "__main__":
    INPUT_DIR = "archive/cleaned_train"
    OUTPUT_DIR = "archive/processed_train"
    MAX_WORKERS = 4
    
    extractor = VideoExtractor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR
    )
    
    print(f"""
    Depth Frame Ayıklama Başlatılıyor:
    - Giriş klasörü: {INPUT_DIR}
    - Çıkış klasörü: {OUTPUT_DIR}
    - Paralel işlem sayısı: {MAX_WORKERS}
    """)
    
    response = input("İşleme devam etmek istiyor musunuz? (e/h): ")
    if response.lower() == 'e':
        extractor.process_all_videos(max_workers=MAX_WORKERS)
    else:
        print("İşlem iptal edildi.") 