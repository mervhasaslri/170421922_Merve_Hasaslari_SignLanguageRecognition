import cv2
import mediapipe as mp
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseHandLandmarkExtractor:
    def __init__(self, input_dir="archive/processed_train/rgb", output_dir="archive/processed_train/landmarks", frame_step=1):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_step = frame_step
        
        # MediaPipe Hands için
        self.mp_hands = mp.solutions.hands
        # MediaPipe Pose için
        self.mp_pose = mp.solutions.pose

    def create_detectors(self):
        """Her frame için yeni Hands ve Pose nesneleri oluştur"""
        hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        return hands, pose

    def process_frame(self, frame_path):
        """Process a single frame and return hand and pose landmarks if detected."""
        try:
            hands, pose = self.create_detectors()
            
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.error(f"Could not read frame: {frame_path}")
                return None

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # El noktalarını tespit et
            hand_results = hands.process(frame_rgb)
            # Poz noktalarını tespit et
            pose_results = pose.process(frame_rgb)
            
            frame_data = {
                'hands': [],
                'pose': None
            }
            
            # El noktalarını işle
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    frame_data['hands'].append(landmarks)
            
            # Poz noktalarını işle
            if pose_results.pose_landmarks:
                pose_landmarks = []
                for landmark in pose_results.pose_landmarks.landmark:
                    pose_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                frame_data['pose'] = pose_landmarks
            
            hands.close()
            pose.close()
            
            if frame_data['hands'] or frame_data['pose']:
                return frame_data
            return None
            
        except Exception as e:
            logger.error(f"Frame processing error for {frame_path}: {str(e)}")
            return None

    def process_video_frames(self, video_dir):
        """Process all frames of a video from the rgb directory."""
        video_name = video_dir.name
        output_path = self.output_dir / f"{video_name}_landmarks.json"
        
        if output_path.exists():
            logger.info(f"Skipping {video_name} - already processed")
            return True

        try:
            frame_files = sorted([f for f in video_dir.glob('*.jpg')])
            total_frames = len(frame_files)
            
            landmarks_data = {
                'video_name': video_name,
                'frame_count': total_frames,
                'frames': {},
                'frame_step': self.frame_step
            }

            for frame_idx, frame_path in enumerate(frame_files):
                if frame_idx % self.frame_step == 0:
                    frame_data = self.process_frame(frame_path)
                    if frame_data:
                        landmarks_data['frames'][str(frame_idx)] = frame_data

                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames for {video_name}")

            # Save landmarks to JSON file
            with open(output_path, 'w') as f:
                json.dump(landmarks_data, f)

            logger.info(f"Successfully processed {video_name}")
            return True

        except Exception as e:
            logger.error(f"Error processing {video_name}: {str(e)}")
            return False

    def process_all_videos(self, num_workers=4):
        """Process all video directories in the input directory using multiple threads."""
        video_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        total_videos = len(video_dirs)
        logger.info(f"Found {total_videos} video directories to process")

        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self.process_video_frames, video_dirs))
            
            successful = sum(1 for r in results if r)
            failed = sum(1 for r in results if not r)

        logger.info(f"Processing complete. Successful: {successful}, Failed: {failed}")
        return successful, failed

def main():
    extractor = PoseHandLandmarkExtractor(frame_step=2)
    successful, failed = extractor.process_all_videos()
    
    if failed > 0:
        logger.warning(f"Failed to process {failed} videos")
    logger.info("Landmark extraction completed")

if __name__ == "__main__":
    main() 