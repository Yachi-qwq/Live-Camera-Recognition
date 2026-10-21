import time
from pathlib import Path
import cv2


GENDER_LABELS = ("male", "female")
GENDER_MODEL_MEAN = (78.4263, 87.7689, 114.8958)
GENDER_MODEL_DIR = Path(__file__).resolve().parent / "models"
MIN_FACE_SIZE = (48, 48)
FACE_OVERLAP_THRESHOLD = 0.35


def box_iou(first, second):
    # Measure overlap so two independent cascades must agree on a face.
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    intersection_width = max(
        0,
        min(first_x + first_width, second_x + second_width)
        - max(first_x, second_x),
    )
    intersection_height = max(
        0,
        min(first_y + first_height, second_y + second_height)
        - max(first_y, second_y),
    )
    intersection = intersection_width * intersection_height
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union else 0.0


def load_gender_model():
    prototxt = GENDER_MODEL_DIR / "deploy.prototxt"
    weights = GENDER_MODEL_DIR / "gender_net.caffemodel"
    for model_path in (prototxt, weights):
        if not model_path.is_file():
            raise FileNotFoundError(f"Modelldatei nicht gefunden: {model_path}")
    return cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))


def predict_gender(gender_model, frame, face):
    # Prepare the face crop in the input format expected by the Caffe network.
    x, y, width, height = face
    face_image = frame[y : y + height, x : x + width]
    blob = cv2.dnn.blobFromImage(
        face_image,
        1.0,
        (227, 227),
        GENDER_MODEL_MEAN,
        swapRB=False,
    )
    gender_model.setInput(blob)
    scores = gender_model.forward()[0]
    label_index = int(scores.argmax())
    return GENDER_LABELS[label_index], float(scores[label_index])


class FaceRecognizer:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.confirmation_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )
        if self.face_detector.empty() or self.confirmation_detector.empty():
            raise RuntimeError("Die OpenCV-Datei zur Gesichtserkennung konnte nicht geladen werden.")

        self.gender_model = load_gender_model()
        self.gender_results = []
        self.last_gender_check = 0.0

    def recognize(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)
        candidates = self.face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=MIN_FACE_SIZE,
        )
        confirmations = self.confirmation_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=MIN_FACE_SIZE,
        )
        # Keep only candidate boxes that are also confirmed by the second cascade.
        faces = sorted(
            (
                face
                for face in candidates
                if any(
                    box_iou(face, confirmation) >= FACE_OVERLAP_THRESHOLD
                    for confirmation in confirmations
                )
            ),
            key=lambda face: face[0],
        )
        now = time.monotonic()
        # Reuse estimates briefly to limit neural-network inference per frame.
        if len(faces) != len(self.gender_results) or now - self.last_gender_check >= 0.5:
            self.gender_results = [
                predict_gender(self.gender_model, frame, face) for face in faces
            ]
            self.last_gender_check = now

        for (x, y, width, height), (label, confidence) in zip(
            faces, self.gender_results
        ):
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 220, 0), 2)
            caption = f"{label} {confidence:.0%}"
            caption_y = max(y - 8, 20)
            (text_width, text_height), baseline = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            cv2.rectangle(
                frame,
                (x, caption_y - text_height - baseline),
                (x + text_width, caption_y + baseline),
                (0, 120, 0),
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                caption,
                (x, caption_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        return frame, self.gender_results
