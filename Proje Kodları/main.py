import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil

def check_video_integrity(video_path):
    """Videoların bütünlüğünü kontrol eder."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        if not ret or frame is None:
            return False
        cap.release()
        return True
    except:
        return False

def check_video_duration(video_path, min_duration=1.0):
    """Video süresini kontrol eder."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration >= min_duration

def check_frame_quality(video_path, min_resolution=(224, 224)):
    """Frame kalitesini ve çözünürlüğünü kontrol eder."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return False
    
    height, width = frame.shape[:2]
    if height < min_resolution[0] or width < min_resolution[1]:
        cap.release()
        return False
    
    # Bozuk frame kontrolü
    while ret:
        if frame is None or frame.size == 0:
            cap.release()
            return False
        ret, frame = cap.read()
    
    cap.release()
    return True

def remove_static_frames(video_path, output_path, threshold=30):
    """Videoların başında ve sonundaki statik frameleri kaldırır."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer oluştur
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    # Baştan ve sondan statik frameleri tespit et
    start_idx = 0
    end_idx = len(frames) - 1
    
    # Baştan statik frame kontrolü
    for i in range(len(frames)-1):
        diff = cv2.absdiff(frames[i], frames[i+1])
        if np.mean(diff) > threshold:
            start_idx = i
            break
    
    # Sondan statik frame kontrolü
    for i in range(len(frames)-1, 0, -1):
        diff = cv2.absdiff(frames[i], frames[i-1])
        if np.mean(diff) > threshold:
            end_idx = i
            break
    
    # Statik olmayan frameleri yaz
    for frame in frames[start_idx:end_idx+1]:
        out.write(frame)
    
    cap.release()
    out.release()

def resize_video(video_path, output_path, target_size=(224, 224)):
    """Videoyu hedef çözünürlüğe ölçekler."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, target_size)
        out.write(resized_frame)
    
    cap.release()
    out.release()

def process_videos(input_dir, output_dir, filtered_dir, min_duration=1.0, min_resolution=(224, 224)):
    """Tüm video işleme adımlarını uygular."""
    # Gerekli klasörleri oluştur
    for directory in [output_dir, filtered_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Alt klasörler oluştur
    corrupted_dir = os.path.join(filtered_dir, "corrupted")
    short_dir = os.path.join(filtered_dir, "too_short")
    low_quality_dir = os.path.join(filtered_dir, "low_quality")
    
    for directory in [corrupted_dir, short_dir, low_quality_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        input_path = os.path.join(input_dir, video_file)
        temp_path = os.path.join(output_dir, f"temp_{video_file}")
        output_path = os.path.join(output_dir, video_file)
        
        # Video bütünlük kontrolü
        if not check_video_integrity(input_path):
            print(f"Corrupted video found: {video_file}")
            shutil.copy2(input_path, os.path.join(corrupted_dir, video_file))
            continue
        
        # Video süre kontrolü
        if not check_video_duration(input_path, min_duration):
            print(f"Too short video found: {video_file}")
            shutil.copy2(input_path, os.path.join(short_dir, video_file))
            continue
        
        # Frame kalite kontrolü
        if not check_frame_quality(input_path, min_resolution):
            print(f"Low quality video found: {video_file}")
            shutil.copy2(input_path, os.path.join(low_quality_dir, video_file))
            continue
        
        # Statik frameleri kaldır
        remove_static_frames(input_path, temp_path)
        
        # Hedef çözünürlüğe ölçekle
        resize_video(temp_path, output_path, target_size=min_resolution)
        
        # Geçici dosyayı sil
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    input_directory = "archive/train"  # Giriş klasörü
    output_directory = "archive/cleaned_train"  # Temizlenmiş videolar için çıkış klasörü
    filtered_directory = "archive/filtered_train"  # Ayıklanan videolar için çıkış klasörü
    
    # Video işleme parametreleri
    MIN_DURATION = 1.0  # saniye
    TARGET_RESOLUTION = (224, 224)  # piksel
    
    process_videos(
        input_directory,
        output_directory,
        filtered_directory,
        min_duration=MIN_DURATION,
        min_resolution=TARGET_RESOLUTION
    )
