import torch
import cv2
import torch.nn.functional as F
from .infer_image import infer
from .get_embedding import load_embeddings_and_names
from .getface import yolo
from torch.nn.modules.distance import PairwiseDistance
from PIL import Image
from models.spoofing.FasNet import Fasnet
import numpy as np
from collections import Counter
from .infer_image import infer, get_align
from .utils import get_model
import os
from gtts import gTTS
from .identity_person import find_closest_person_vote,find_closest_person
from playsound import playsound
import yaml
from datetime import datetime
import pandas as pd
import time
import threading

# use config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)['infer_video']

# get recogn model
recogn_model = get_model('inceptionresnetV1')
# define distance calculator
l2_distance = PairwiseDistance(p=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
antispoof_model = Fasnet()


def infer_camera(min_face_area=config['min_face_area'], 
                 bbox_threshold=config['bbox_threshold'], 
                 required_images=config['required_images']):
    
    '''
    Captures frames from the camera, detects faces, and processes valid images 
    based on face alignment, spoofing detection, and additional conditions.

    Parameters:
        min_face_area (int): Minimum face area (in pixels) required for a valid detection.
        bbox_threshold (float): Minimum confidence score required to consider a detected face as valid.
        required_images (int): The number of valid face images to collect before stopping.

    Returns:
        dict: A dictionary containing:
            - 'valid_images' (list): A list of preprocessed images (torch.Tensor) of valid faces.
    
    '''

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở camera")
        return

    valid_images = [] 
    is_reals = []
    last_sound_time = 0
    sound_delay = 2 
    previous_message = 0 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể chụp được hình ảnh")
            break

        input_image, face, prob, landmark = get_align(frame)
       
        if face is not None:
            x1, y1, x2, y2 = map(int, face)
            if prob > bbox_threshold:
                line_length = 20
                color = (0, 255, 0) 
                thickness = 1     

                cv2.line(frame, (x1, y1), (x1 + line_length, y1), color, thickness) 
                cv2.line(frame, (x1, y1), (x1, y1 + line_length), color, thickness) 

                cv2.line(frame, (x2, y1), (x2 - line_length, y1), color, thickness) 
                cv2.line(frame, (x2, y1), (x2, y1 + line_length), color, thickness) 

                cv2.line(frame, (x1, y2), (x1 + line_length, y2), color, thickness) 
                cv2.line(frame, (x1, y2), (x1, y2 - line_length), color, thickness)

                cv2.line(frame, (x2, y2), (x2 - line_length, y2), color, thickness) 
                cv2.line(frame, (x2, y2), (x2, y2 - line_length), color, thickness) 

                cv2.putText(frame, f"Face {prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

            area = (face[2] - face[0]) * (face[3] - face[1])

            if prob > bbox_threshold:
                center = np.mean(landmark, axis=0)
                height, width, _ = frame.shape
                center_x, center_y = center

                distance_from_center = np.sqrt((center_x - width / 2) ** 2 + (center_y - height / 2) ** 2)
                current_time = time.time()

                if area > min_face_area:
                    if width * 0.15 < center_x < width * 0.85 and height * 0.15 < center_y < height * 0.85 and distance_from_center < min(width, height) * 0.4:
                        if previous_message != 1 and current_time - last_sound_time > sound_delay:
                            
                            threading.Thread(target=playsound, args=('audio/guide_keepface.mp3',), daemon=True).start()
                           
                            last_sound_time = current_time
                            previous_message = 1

                        is_real, score = antispoof_model.analyze(frame, map(int, face)) 
                        print(is_real, score)
                        is_reals.append((is_real, score))
                        valid_images.append(input_image)

                    else:
                        if previous_message != 2 and current_time - last_sound_time > sound_delay:
                          
                            threading.Thread(target=playsound, args=('audio/guide_centerface.mp3',), daemon=True).start()
                         
                            last_sound_time = current_time
                            previous_message = 2

                else:
                    if previous_message != 3 and current_time - last_sound_time > sound_delay:
                        
                        threading.Thread(target=playsound, args=('audio/closer.mp3',), daemon=True).start()
                       
                        last_sound_time = current_time

        else:
            if previous_message != 0:
                print("Không phát hiện khuôn mặt")
                previous_message = 0

        cv2.imshow('FACE RECOGNITON', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(valid_images) >= required_images:
            print(f"Đã thu thập đủ {required_images} ảnh hợp lệ.")
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
        idx_to_class, 
        recogn_model, 
        is_anti_spoof=config['is_anti_spoof'], 
        validation_threshold=config['validation_threshold'], 
        is_vote=config['is_vote'],
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
        idx_to_class (dict): Mapping of class IDs to human-readable names.
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

    for i, image in enumerate(valid_images):

        if is_anti_spoof:
            if not input['is_reals'][i][0] and input['is_reals'][i][1] > anti_spoof_threshold:
                continue

        pred_embed = infer(recogn_model, image)

        if is_vote:
            result = find_closest_person_vote(pred_embed, embeddings, image2class, distance_mode=distance_mode)
        else:
            result = find_closest_person(pred_embed, embeddings, image2class, distance_mode=distance_mode)

        print(result)
        if result != -1:
            predict_class.append(result)

    class_count = Counter(predict_class)
    
    majority_threshold = len(valid_images) * validation_threshold

    person_identified = False

    log_data = []

    for cls, count in class_count.items():
        if count >= majority_threshold:
            person_name = idx_to_class.get(cls, 'Unknown')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            print(f"Người được nhận diện là: {person_name}")

            log_data.append({
                'Person Name': person_name,
                'Timestamp': timestamp,
                'Class Ratio': f"{count}/{len(valid_images)} ({count / len(valid_images):.2%})"
            })
            
            audio_dir = "audio"
            os.makedirs(audio_dir, exist_ok=True)

            audio_path = os.path.join(audio_dir, "greeting.mp3")
            if os.path.exists(audio_path):
                os.remove(audio_path)

            tts = gTTS(f"Xin chào {person_name}", lang='vi')
            tts.save(audio_path)
            try:
                playsound(audio_path)
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
       
 
        log_data.append({
        'Person Name': 'Unknown',
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Class Ratio': f"{valid_images_len - len(predict_class) }/{valid_images_len} ({(valid_images_len - len(predict_class)) / valid_images_len:.2%})"})
      

    log_file = "recognition_log.xlsx"
    if os.path.exists(log_file):
        existing_data = pd.read_excel(log_file)
        log_data = pd.concat([existing_data, pd.DataFrame(log_data)], ignore_index=True)
    else:
        log_data = pd.DataFrame(log_data)

    log_data.to_excel(log_file, index=False)
    
    if person_identified:
        return person_name
    return False


if __name__ == '__main__':

    recogn_model_name = 'inceptionresnetV1'
    embedding_file_path = f'data/data_source/db1/{recogn_model_name}_embeddings.npy'
    image2class_file_path = f'data/data_source/db1/{recogn_model_name}_image2class.pkl'
    index2class_file_path = f'data/data_source/db1/{recogn_model_name}_index2class.pkl'
    
    embeddings, image2class, index2class = load_embeddings_and_names(embedding_file_path, image2class_file_path, index2class_file_path)
    result = infer_camera()

    check_validation(result, embeddings, image2class, index2class, recogn_model)