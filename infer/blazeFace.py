import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MP_VERBOSE'] = '0'
os.environ['GLOG_minloglevel'] = '2'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('tensorflow.python').setLevel(logging.ERROR)
logging.getLogger('tensorflow.lite.python.lite').setLevel(logging.ERROR)
# Add these new logging configurations
logging.getLogger('mediapipe.python.solution_base').setLevel(logging.ERROR)
logging.getLogger('mediapipe.python.solutions.face_detection').setLevel(logging.ERROR)
logging.getLogger('mediapipe.python.solutions.face_mesh').setLevel(logging.ERROR)
logging.getLogger('mediapipe.python.solutions.face_detection_connections').setLevel(logging.ERROR)


import cv2
cv2.setLogLevel(0)  # hoặc dùng cv2.utils.logging.setLogLevel nếu cần

import mediapipe as mp
# Suppress MediaPipe warnings
mp.solutions.face_detection.FaceDetection._SUPPRESS_WARNINGS = True
mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils

def detect_face_and_nose(frame):

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                h, w, _ = frame.shape

                # Bounding box
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                # Nose tip (keypoint index 2)
                nose = detection.location_data.relative_keypoints[2]
                nose_x = int(nose.x * w)
                nose_y = int(nose.y * h)

                # Confidence score
                prob = float(detection.score[0])  # độ tin cậy đầu tiên

                return (x1, y1, x2, y2), (nose_x, nose_y), prob

    return None, None, None





