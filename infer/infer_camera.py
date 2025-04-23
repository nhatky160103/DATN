import torch
import cv2
from models.Anti_spoof.FasNet import Fasnet
import numpy as np
from collections import Counter
from .infer_image import getEmbedding, mtcnn
import os
from .identity_person import find_closest_person
from playsound import playsound
import yaml
import threading
from .utils import get_recogn_model
import time
from .get_embedding import EmbeddingManager
from .blazeFace import detect_face_and_nose

#use config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)['infer_video']

# get recogn model
arcface_model = get_recogn_model()
antispoof_model = Fasnet()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def draw_camera_focus_box(frame, width, height, color=(0, 255, 255), thickness=1, length=20):
    overlay = frame.copy()
    alpha = 0.4

    x1 = int(width * 0.2)
    y1 = int(height * 0.2)
    x2 = int(width * 0.8)
    y2 = int(height * 0.8)
    
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


    
def infer_camera(min_face_area=config['min_face_area'], 
                 bbox_threshold=config['bbox_threshold'], 
                 required_images=config['required_images']):

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
            continue

        center_x, center_y = map(int, center_point)
        height, width, _ = frame.shape
        x1, y1, x2, y2 = map(int, face)

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
                    if previous_message != 1 and current_time - last_sound_time > sound_delay:
                        threading.Thread(target=playsound, args=('audio/guide_keepface.mp3',), daemon=True).start()
                        
                        last_sound_time = current_time
                        previous_message = 1

                    is_real, score = antispoof_model.analyze(origin_frame, map(int, face)) 
                    print(is_real, score)
                    is_reals.append((is_real, score))
                    valid_images.append(origin_frame)

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
            print(f"Collect enough {required_images} valid images.")
            break
     
    cap.release()
    cv2.destroyAllWindows()
    result = {
        'valid_images': valid_images,
        'is_reals': is_reals
    }
    return result


def check_validation(
        input, 
        embeddings, 
        image2class, 
        index2class,
        recogn_model, 
        is_anti_spoof=config['is_anti_spoof'], 
        validation_threshold=config['validation_threshold'],
        distance_mode=config['distance_mode'], 
        anti_spoof_threshold=config['anti_spoof_threshold']):
    '''
    Validates and identifies a person based on input images and embeddings.

    Parameters:
        input (dict): A dictionary containing:
            - 'valid_images' (list): List of valid preprocessed face images (torch.Tensor).
            - 'is_reals' (list): List of tuples (bool, float) indicating if the image passed anti-spoof checks and its score.
        embeddings Tensor: Precomputed embeddings for known classes.
        image2class (dict): Mapping of embeddings to their respective class IDs.
        recogn_model (torch.nn.Module): The face recognition model used for inference.
        is_anti_spoof (bool): Whether to apply anti-spoofing validation.
        validation_threshold (float): The minimum ratio of valid votes required to confirm identification.
        is_vote (bool): Whether to use voting logic for identification.
        distance_mode (str): The distance metric used for embedding comparison ('cosine' or 'euclidean').
        anti_spoof_threshold (float): The score threshold for anti-spoof validation.

    Returns:
        str or bool: The name of the identified person if validation succeeds, otherwise False.

    '''
    valid_images = input['valid_images']

    if len(valid_images) == 0:
        print("Không có ảnh để xử lý.")
        return
    
    predict_class = []

    for i, raw_image in enumerate(valid_images):
        image = mtcnn(raw_image)
        if image is None:
            image = raw_image
        if is_anti_spoof:
            if not input['is_reals'][i][0] and input['is_reals'][i][1] > anti_spoof_threshold:
                continue

        pred_embed = getEmbedding(recogn_model, image)

        result = find_closest_person(pred_embed, embeddings, image2class, distance_mode=distance_mode)

        print(result)
        if result != -1:
            predict_class.append(result)

    class_count = Counter(predict_class)
    
    majority_threshold = len(valid_images) * validation_threshold

    person_identified = False

    for cls, count in class_count.items():
        if count >= majority_threshold:
            person_id = index2class.get(cls, 'UNKNOWN')
        
            print(f"Người được nhận diện là: {person_id}")

            try:
                playsound('audio/greeting.mp3')
            except Exception as e:
                print(f"Lỗi khi phát âm thanh: {e}")
      
            person_identified = True

    if not person_identified:
        valid_images_len= len(valid_images)
        print("Unknown person")
        try:
            playsound('audio/retry.mp3')
        except Exception as e:
            print(f"Lỗi khi phát âm thanh: {e}")
       
       
    if person_identified:
        return person_id
    return False


if __name__ == '__main__':

    manager = EmbeddingManager('Hust', 'glint360k_cosface')
    embeddings, image2class, index2class = manager.load()
    
    result = infer_camera()
    # check_validation(result, embeddings, image2class, index2class, arcface_model)
    # print(result['is_reals'])
    for image in result['valid_images']:
        image = mtcnn(image).numpy().transpose(1,2,0)
        print(type(image))
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


