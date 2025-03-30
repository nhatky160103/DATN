# built-in dependencies
import os
from typing import Any, List

# 3rd party dependencies
import cv2
import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np



# Notice that all facial detector models must be inherited from this class


# pylint: disable=unnecessary-pass, too-few-public-methods, too-many-instance-attributes
class Detector(ABC):
    @abstractmethod
    def detect_faces(self, img: np.ndarray) -> List["FacialAreaRegion"]:
        """
        Interface for detect and align face

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
                where each object contains:

            - facial_area (FacialAreaRegion): The facial area region represented
                as x, y, w, h, left_eye and right_eye. left eye and right eye are
                eyes on the left and right respectively with respect to the person
                instead of observer.
        """
        pass


@dataclass
class FacialAreaRegion:
    """
    Initialize a Face object.

    Args:
        x (int): The x-coordinate of the top-left corner of the bounding box.
        y (int): The y-coordinate of the top-left corner of the bounding box.
        w (int): The width of the bounding box.
        h (int): The height of the bounding box.
        left_eye (tuple): The coordinates (x, y) of the left eye with respect to
            the person instead of observer. Default is None.
        right_eye (tuple): The coordinates (x, y) of the right eye with respect to
            the person instead of observer. Default is None.
        confidence (float, optional): Confidence score associated with the face detection.
            Default is None.
    """

    x: int
    y: int
    w: int
    h: int
    left_eye: Optional[Tuple[int, int]] = None
    right_eye: Optional[Tuple[int, int]] = None
    confidence: Optional[float] = None
    nose: Optional[Tuple[int, int]] = None
    mouth_right: Optional[Tuple[int, int]] = None
    mouth_left: Optional[Tuple[int, int]] = None


@dataclass
class DetectedFace:
    """
    Initialize detected face object.

    Args:
        img (np.ndarray): detected face image as numpy array
        facial_area (FacialAreaRegion): detected face's metadata (e.g. bounding box)
        confidence (float): confidence score for face detection
    """

    img: np.ndarray
    facial_area: FacialAreaRegion
    confidence: float


class OpenCvClient(Detector):
    """
    Class to cover common face detection functionalitiy for OpenCv backend
    """

    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        """
        Build opencv's face and eye detector models
        Returns:
            model (dict): including face_detector and eye_detector keys
        """
        detector = {}
        detector["face_detector"] = self.__build_cascade("haarcascade")
        detector["eye_detector"] = self.__build_cascade("haarcascade_eye")
        return detector

    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Detect and align face with opencv

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        detected_face = None

        faces = []
        try:
            # faces = detector["face_detector"].detectMultiScale(img, 1.3, 5)

            # note that, by design, opencv's haarcascade scores are >0 but not capped at 1
            faces, _, scores = self.model["face_detector"].detectMultiScale3(
                img, 1.1, 10, outputRejectLevels=True
            )
        except:
            pass

        if len(faces) > 0:
            for (x, y, w, h), confidence in zip(faces, scores):
                detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
                left_eye, right_eye = self.find_eyes(img=detected_face)

                # eyes found in the detected face instead image itself
                # detected face's coordinates should be added
                if left_eye is not None:
                    left_eye = (int(x + left_eye[0]), int(y + left_eye[1]))
                if right_eye is not None:
                    right_eye = (int(x + right_eye[0]), int(y + right_eye[1]))

                facial_area = FacialAreaRegion(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    left_eye=left_eye,
                    right_eye=right_eye,
                    confidence=(100 - confidence) / 100,
                )
                resp.append(facial_area)

        return resp

    def find_eyes(self, img: np.ndarray) -> tuple:
        """
        Find the left and right eye coordinates of given image
        Args:
            img (np.ndarray): given image
        Returns:
            left and right eye (tuple)
        """
        left_eye = None
        right_eye = None

        # if image has unexpectedly 0 dimension then skip alignment
        if img.shape[0] == 0 or img.shape[1] == 0:
            return left_eye, right_eye

        detected_face_gray = cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY
        )  # eye detector expects gray scale image

        eyes = self.model["eye_detector"].detectMultiScale(detected_face_gray, 1.1, 10)

        # ----------------------------------------------------------------

        # opencv eye detection module is not strong. it might find more than 2 eyes!
        # besides, it returns eyes with different order in each call (issue 435)
        # this is an important issue because opencv is the default detector and ssd also uses this
        # find the largest 2 eye. Thanks to @thelostpeace

        eyes = sorted(eyes, key=lambda v: abs(v[2] * v[3]), reverse=True)

        # ----------------------------------------------------------------
        if len(eyes) >= 2:
            # decide left and right eye

            eye_1 = eyes[0]
            eye_2 = eyes[1]

            if eye_1[0] < eye_2[0]:
                right_eye = eye_1
                left_eye = eye_2
            else:
                right_eye = eye_2
                left_eye = eye_1

            # -----------------------
            # find center of eyes
            left_eye = (
                int(left_eye[0] + (left_eye[2] / 2)),
                int(left_eye[1] + (left_eye[3] / 2)),
            )
            right_eye = (
                int(right_eye[0] + (right_eye[2] / 2)),
                int(right_eye[1] + (right_eye[3] / 2)),
            )
        return left_eye, right_eye

    def __build_cascade(self, model_name="haarcascade") -> Any:
        """
        Build a opencv face&eye detector models
        Returns:
            model (Any)
        """
        opencv_path = self.__get_opencv_path()
        if model_name == "haarcascade":
            face_detector_path = os.path.join(opencv_path, "haarcascade_frontalface_default.xml")
            if not os.path.isfile(face_detector_path):
                raise ValueError(
                    "Confirm that opencv is installed on your environment! Expected path ",
                    face_detector_path,
                    " violated.",
                )
            detector = cv2.CascadeClassifier(face_detector_path)

        elif model_name == "haarcascade_eye":
            eye_detector_path = os.path.join(opencv_path, "haarcascade_eye.xml")
            if not os.path.isfile(eye_detector_path):
                raise ValueError(
                    "Confirm that opencv is installed on your environment! Expected path ",
                    eye_detector_path,
                    " violated.",
                )
            detector = cv2.CascadeClassifier(eye_detector_path)

        else:
            raise ValueError(f"unimplemented model_name for build_cascade - {model_name}")

        return detector

    def __get_opencv_path(self) -> str:
        """
        Returns where opencv installed
        Returns:
            installation_path (str)
        """
        return os.path.join(os.path.dirname(cv2.__file__), "data")
