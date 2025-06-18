import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe modellerini başlat
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def process_image(image_path, pose, hands):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Görüntü okunamadı")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Poz ve el algılama
        pose_results = pose.process(image_rgb)
        hands_results = hands.process(image_rgb)

        if not pose_results.pose_landmarks and not hands_results.multi_hand_landmarks:
            raise Exception("İskelet ve eller tespit edilemedi")

        # Görüntüye çizimler ekleme
        annotated_image = image.copy()

        # Vücut iskeleti çizimi
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, pose_results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2)
            )

        # El iskeleti çizimi (farklı renkte)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 0, 0), thickness=2, circle_radius=2)
                )

        return annotated_image, True

    except Exception as e:
        return None, False

def process_video_folder(input_folder, output_folder):
    success_count = 0
    fail_count = 0
    total_count = 0

    # MediaPipe modellerini başlat
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    video_folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

    for video_folder in tqdm(video_folders, desc="Toplam İlerleme"):
        input_video_path = os.path.join(input_folder, video_folder)
        output_video_path = os.path.join(output_folder, video_folder)

        os.makedirs(output_video_path, exist_ok=True)

        frames = sorted([f for f in os.listdir(input_video_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

        for frame in tqdm(frames, desc=f"İşleniyor: {video_folder}", leave=False):
            input_frame_path = os.path.join(input_video_path, frame)
            output_frame_path = os.path.join(output_video_path, frame)

            try:
                annotated_image, success = process_image(input_frame_path, pose, hands)

                if success:
                    cv2.imwrite(output_frame_path, annotated_image)
                    success_count += 1
                else:
                    fail_count += 1
                    logger.error(f"Hata oluştu ({video_folder}): İskelet veya el tespit edilemedi")

                total_count += 1

            except Exception as e:
                fail_count += 1
                total_count += 1
                logger.error(f"Hata oluştu ({video_folder}): {str(e)}")

    logger.info(f"""
        İşlem tamamlandı:
        - Başarılı: {success_count}
        - Başarısız: {fail_count}
        - İşlenen: {total_count}
        """)

if __name__ == "__main__":
    input_folder = "archive/processed_train/rgb"
    output_folder = "archive/processed_train/skeleton"

    os.makedirs(output_folder, exist_ok=True)
    
    process_video_folder(input_folder, output_folder)
