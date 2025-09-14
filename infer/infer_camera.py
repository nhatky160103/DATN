import cv2
from models.Anti_spoof.FasNet_onnx import FasnetOnnx
import numpy as np
from collections import Counter
from .infer_image import getEmbedding
from .identity_person import find_closest_person
from playsound import playsound
import yaml
import threading
from .utils import load_onnx_model
import time
from .blazeFace import detect_face_and_nose
from PIL import Image, ImageSequence
from models.lightqnet.tf_face_quality_model_onnx import OnnxFaceQualityModel
# get recogn model
arcface_model = load_onnx_model()
antispoof_model = FasnetOnnx()
face_q_model = OnnxFaceQualityModel()



def yield_loading_gif_frames(gif_path):
    gif = Image.open(gif_path)

    for frame in ImageSequence.Iterator(gif):
        frame_np = np.array(frame)
        image_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

        _, buffer = cv2.imencode('.jpg', image_rgb)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')



def draw_camera_focus_box(frame, width, height, color=(0, 255, 255), thickness=1, length=20):
    overlay = frame.copy()
    alpha = 0.4

    x1 = int(width * 0.2)
    y1 = int(height * 0.1)
    x2 = int(width * 0.8)
    y2 = int(height * 0.9)
    
    # Làm tối vùng ngoài
    mask = np.zeros_like(frame)
    cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
    frame[:] = cv2.addWeighted(frame, alpha, cv2.bitwise_and(frame, mask), 1 - alpha, 0)

    # Góc trên trái
    cv2.line(overlay, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(overlay, (x1, y1), (x1, y1 + length), color, thickness)

    # Góc trên phải
    cv2.line(overlay, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(overlay, (x2, y1), (x2, y1 + length), color, thickness)

    # Góc dưới trái
    cv2.line(overlay, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(overlay, (x1, y2), (x1, y2 - length), color, thickness)

    # Góc dưới phải
    cv2.line(overlay, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(overlay, (x2, y2), (x2, y2 - length), color, thickness)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


    
def infer_camera(config = None, 
                 result_queue = None
                 ):
    min_face_area=config['infer_video']['min_face_area']
    bbox_threshold=config['infer_video']['bbox_threshold'] 
    required_images=config['infer_video']['required_images']
    qscore_threshold = config['infer_video']['qscore_threshold']
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở camera")
        return

    valid_images = [] 
    is_reals = []
    previous_message = 0 
    sound_delay = 2
    last_sound_time = 0
    while True:
        ret, frame = cap.read()
        origin_frame = frame.copy()
        if not ret:
            print("Không thể chụp được hình ảnh")
            break
        
        face , center_point, prob = detect_face_and_nose(frame)


        
        
        if face is None or prob is None or center_point is None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue

        center_x, center_y = map(int, center_point)
        height, width, _ = frame.shape
        x1, y1, x2, y2 = map(int, face)

        print(x1, y1, x2, y2)
        draw_camera_focus_box(frame, width, height)

        if prob > bbox_threshold:
            color = (255, 255, 255)

            cv2.line(frame, (x1, y1), (x1 + 20, y1), color, 1) 
            cv2.line(frame, (x1, y1), (x1, y1 + 20), color, 1) 
            cv2.line(frame, (x2, y1), (x2 - 20, y1), color, 1) 
            cv2.line(frame, (x2, y1), (x2, y1 + 20), color, 1) 
            cv2.line(frame, (x1, y2), (x1 + 20, y2), color, 1) 
            cv2.line(frame, (x1, y2), (x1, y2 - 20), color, 1)
            cv2.line(frame, (x2, y2), (x2 - 20, y2), color, 1) 
            cv2.line(frame, (x2, y2), (x2, y2 - 20), color, 1) 

            cv2.putText(frame, f"Face {prob:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            area = (face[2] - face[0]) * (face[3] - face[1])

            cv2.drawMarker(
                frame,
                (int(center_x), int(center_y)),
                color=(0, 0, 255),         
                markerType=cv2.MARKER_CROSS, 
                markerSize=10,             
                thickness=1,
                line_type=cv2.LINE_AA       
            )

            current_time = time.time()
            
            if area > min_face_area*height*width:
                if width * 0.2 < center_x < width * 0.8 and height * 0.2 < center_y < height * 0.8:
                
                    
                    
                    x_1 = int(width * 0.2)
                    y_1 = int(height * 0.1)
                    x_2 = int(width * 0.8)
                    y_2 = int(height * 0.9)
                
                    crop_face = origin_frame[y_1:y_2, x_1:x_2]
                    quality_score = face_q_model.inference(crop_face)
                    if quality_score >= qscore_threshold:  # quality threshold
                        if previous_message != 1 and current_time - last_sound_time > sound_delay:
                           threading.Thread(target=playsound, args=('audio/guide_keepface.mp3',), daemon=True).start()
                           last_sound_time = current_time
                           previous_message = 1

                        is_real, score = antispoof_model.analyze(origin_frame, map(int, face))
                        print('---->',is_real, score)
                        is_reals.append((is_real, score))
                        valid_images.append(origin_frame[y1:y2, x1:x2])

                else:
                    if previous_message != 2 and current_time - last_sound_time > sound_delay:
                        
                        threading.Thread(target=playsound, args=('audio/guide_centerface.mp3',), daemon=True).start()
                        
                        last_sound_time = current_time
                        previous_message = 2

            else:
                if previous_message != 3 and current_time - last_sound_time > sound_delay:
                    
                    threading.Thread(target=playsound, args=('audio/closer.mp3',), daemon=True).start()
                    
                    last_sound_time = current_time
                    previous_message = 3

  
        # cv2.imshow( "SMART CAMERA PREVIEW", frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        if len(valid_images) >= required_images:
            gif_path = "interface/static/assets/Loading.gif"
            yield from yield_loading_gif_frames(gif_path)

            break

    cap.release()
    cv2.destroyAllWindows()
    if result_queue:
                result_queue.put({
                    'valid_images': valid_images,
                    'is_reals': is_reals
                })


def check_validation(
        input, 
        embeddings, 
        image2class, 
        index2class,
        config = None,
        ):

    is_anti_spoof = config['infer_video']['is_anti_spoof']
    validation_threshold = config['infer_video']['validation_threshold']
    anti_spoof_threshold = config['infer_video']['anti_spoof_threshold'] 
    distance_mode = config['identity_person']['distance_mode']
    l2_threshold = config['identity_person']['l2_threshold']
    cosine_threshold = config['identity_person']['cosine_threshold']

    valid_images = input['valid_images']

    if valid_images is None or len(valid_images) == 0:
        print("Không có ảnh để xử lý.")
        try:
            playsound('audio/retry.mp3')
            return 'UNKNOWN'
        except Exception as e:
            print(f"Lỗi khi phát âm thanh: {e}")

    if embeddings is None or len(embeddings) == 0 or not image2class or not index2class:
        print("Không có nhân viên trong database.")
        try:
            playsound('audio/retry.mp3')
            return 'UNKNOWN'
        except Exception as e:
            print(f"Lỗi khi phát âm thanh: {e}")
    
    predict_class = []

    processed_faces = []


    print("START TESTING ......................................")
    for i, face in enumerate(valid_images):
        print(type(face))
        print(face.shape)
        if is_anti_spoof:
            if not input['is_reals'][i][0] and input['is_reals'][i][1] > anti_spoof_threshold:
                continue
        processed_faces.append(face)
    
    # Get embeddings for all processed faces in batch
    if processed_faces:
        pred_embeds = getEmbedding(arcface_model, processed_faces)
   
        for pred_embed in pred_embeds:
            result = find_closest_person(
                pred_embed, 
                embeddings, 
                image2class, 
                distance_mode=distance_mode, 
                l2_threshold=l2_threshold, 
                cosine_threshold=cosine_threshold
            )
            if result != -1:
                predict_class.append(result)
            print(result)
            print("__"*10)

    # Count predictions and check against validation threshold
    class_count = Counter(predict_class)
    majority_threshold = len(valid_images) * validation_threshold

    for cls, count in class_count.items():
        if count >= majority_threshold:
            person_id = index2class.get(cls, 'UNKNOWN')
            print(f"Người được nhận diện là: {person_id}")
            try:
                playsound('audio/greeting.mp3')
            except Exception as e:
                print(f"Lỗi khi phát âm thanh: {e}")
            return person_id

    print("Unknown person")
    try:
        playsound('audio/retry.mp3')
        return 'UNKNOWN'
    except Exception as e:
        print(f"Lỗi khi phát âm thanh: {e}")



