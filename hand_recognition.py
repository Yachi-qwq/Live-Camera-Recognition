import os
import math
import time
from contextlib import contextmanager
from pathlib import Path
import cv2
import mediapipe as mp


HAND_COLOR = (0, 165, 255)
HAND_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


def distance(first, second):
    return sum((first[axis] - second[axis]) ** 2 for axis in range(3)) ** 0.5


def joint_angle(first, joint, last):
    first_vector = [first[axis] - joint[axis] for axis in range(3)]
    last_vector = [last[axis] - joint[axis] for axis in range(3)]
    first_length = distance(first, joint)
    last_length = distance(last, joint)
    if not first_length or not last_length:
        return 0.0
    cosine = sum(
        first_vector[axis] * last_vector[axis] for axis in range(3)
    ) / (first_length * last_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def count_extended_fingers(landmarks):
    # Estimate extension from joint angles and fingertip distance to the wrist.
    points = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks]
    wrist = points[0]
    finger_joints = ((6, 8), (10, 12), (14, 16), (18, 20))
    extended = sum(
        joint_angle(points[pip - 1], points[pip], points[tip]) >= 150
        and distance(points[tip], wrist) > distance(points[pip], wrist) * 1.05
        for pip, tip in finger_joints
    )

    thumb_extended = (
        joint_angle(points[2], points[3], points[4]) >= 145
        and distance(points[4], wrist) > distance(points[3], wrist) * 1.05
    )
    return extended + int(thumb_extended)


@contextmanager
def suppress_native_stderr():
    original_stderr = os.dup(2)
    null_stderr = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_stderr, 2)
        yield
    finally:
        os.dup2(original_stderr, 2)
        os.close(null_stderr)
        os.close(original_stderr)


class HandRecognizer:
    def __init__(self):
        if not HAND_MODEL_PATH.is_file():
            raise FileNotFoundError(f"Modelldatei nicht gefunden: {HAND_MODEL_PATH}")
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(HAND_MODEL_PATH)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        with suppress_native_stderr():
            self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
                options
            )
        self.last_timestamp_ms = 0
        self.startup_check = True

    def recognize(self, frame):
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # VIDEO mode requires strictly increasing timestamps for each frame.
        timestamp_ms = max(int(time.monotonic() * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        if self.startup_check:
            with suppress_native_stderr():
                results = self.landmarker.detect_for_video(image, timestamp_ms)
            self.startup_check = False
        else:
            results = self.landmarker.detect_for_video(image, timestamp_ms)
        hands = results.hand_landmarks or []

        finger_counts = []
        for landmarks in hands:
            finger_count = count_extended_fingers(landmarks)
            finger_counts.append(finger_count)
            points = [
                (
                    min(max(round(landmark.x * width), 0), width - 1),
                    min(max(round(landmark.y * height), 0), height - 1),
                )
                for landmark in landmarks
            ]
            # Draw MediaPipe's landmark connections in orange over the camera image.
            for connection in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[connection.start],
                    points[connection.end],
                    HAND_COLOR,
                    2,
                )
            for point in points:
                cv2.circle(frame, point, 4, HAND_COLOR, cv2.FILLED)

            left = min(point[0] for point in points)
            top = min(point[1] for point in points)
            right = max(point[0] for point in points)
            bottom = max(point[1] for point in points)
            cv2.rectangle(frame, (left, top), (right, bottom), HAND_COLOR, 2)
            cv2.putText(
                frame,
                f"{finger_count} Finger",
                (left, max(top - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                HAND_COLOR,
                2,
            )
        return frame, finger_counts

    def close(self):
        self.landmarker.close()
