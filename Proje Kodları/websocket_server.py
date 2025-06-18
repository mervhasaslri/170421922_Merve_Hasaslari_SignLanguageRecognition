import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import uvloop
import tensorflow as tf

from real_time_sign_language import RealTimeSignLanguageRecognizer
from gemini_sentence import kelimelerden_cumle_uret

recognizer = RealTimeSignLanguageRecognizer(
    model_path="models/sign_language/best_model.h5",
    class_map_path="archive/SignList_ClassId_TR_EN.csv"
)

uvloop.install()

async def process_frame(websocket):
    print("Yeni istemci bağlandı")
    best_prediction = None
    frame_count = 0
    recording_ended = False

    try:
        async for message in websocket:
            try:
                data = json.loads(message)

                if 'predictions' in data:
                    try:
                        print(f"Tahmin listesi alındı: {data['predictions']}")
                        cumle = kelimelerden_cumle_uret(data['predictions'])
                        response = {'sentence': cumle}
                        await websocket.send(json.dumps(response))
                        continue
                    except Exception as e:
                        print(f"Cümle oluşturma hatası: {str(e)}")
                        await websocket.send(json.dumps({'error': 'Cümle oluşturulamadı'}))
                        continue

                is_recording = data.get('isRecording', False)
                frame_count += 1
                print(f"Frame #{frame_count}, Kayıt durumu: {is_recording}")

                if not is_recording and recording_ended:
                    recording_ended = False
                    if best_prediction:
                        print(f"En iyi tahmin gönderiliyor: {best_prediction}")
                        response = {
                            'prediction': best_prediction['tr_label'],
                            'confidence': float(best_prediction['confidence']),
                            'en_label': best_prediction['en_label']
                        }
                        await websocket.send(json.dumps(response))
                        recognizer.landmark_buffer.clear()
                        best_prediction = None
                        frame_count = 0
                    else:
                        print("Kayıt bitti ama tahmin bulunamadı")
                    continue

                if not is_recording:
                    recording_ended = True
                    continue

                image_data = data['image'].split(',')[1]
                img_bytes = base64.b64decode(image_data)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is None:
                    print("Frame decode edilemedi")
                    continue

                # 🛠️ Tahmin yap (DÜZENLENMİŞ kısım burası!)
                landmarks_data = recognizer.extract_landmarks(frame)
                features = recognizer.prepare_sequence(landmarks_data)
                recognizer.landmark_buffer.append(features)

                # Buffer boyutunu kontrol et
                if len(recognizer.landmark_buffer) > recognizer.sequence_length:
                    recognizer.landmark_buffer.pop(0)

                # Yeterli frame biriktiyse tahmin yap
                if len(recognizer.landmark_buffer) == recognizer.sequence_length:
                    prediction = recognizer.predict()
                else:
                    prediction = None

                if prediction:
                    print(f"Tahmin yapıldı: {prediction}")
                    if best_prediction is None or prediction['confidence'] > best_prediction['confidence']:
                        best_prediction = prediction

            except Exception as e:
                print(f"Veri işleme hatası: {str(e)}")

    except websockets.exceptions.ConnectionClosed:
        print("Bağlantı kapatıldı.")

async def main():
    async with websockets.serve(process_frame, "localhost", 8767):
        print("WebSocket sunucusu başlatıldı ve 8767 portunda dinlemeye hazır")
        await asyncio.Future()  # Sunucunun sürekli açık kalmasını sağlar

if __name__ == "__main__":
    asyncio.run(main())
