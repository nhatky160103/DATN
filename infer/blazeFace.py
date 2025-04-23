import mediapipe as mp
import cv2

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