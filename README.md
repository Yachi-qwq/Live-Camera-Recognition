<div align="center">

# Live Camera Recognition

**A lightweight desktop camera viewer with real-time face, hand, and finger detection.**

</div>

## Overview

Live Camera Recognition opens the first available camera in a Tkinter window, displays the live video beside a data panel, and overlays face and hand detections. Faces are outlined in green with a model-based `male` or `female` estimate; hands are outlined in orange with their estimated extended-finger count. These are experimental visual predictions, not reliable statements about a person.

## Installation

### Requirements

- Python 3.10 or newer
- A working camera
- An internet connection on the first start if the gender model weights are missing

From a terminal in the project directory, install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The application uses these model assets:

```text
models/
|-- deploy.prototxt
|-- gender_net.caffemodel
|-- hand_landmarker.task
```

`deploy.prototxt` and `hand_landmarker.task` must be present in `models/`.
If `gender_net.caffemodel` is missing, the application downloads it from its
public source into `models/` the first time face recognition is initialized.
An internet connection is only needed for that initial download.

Start the application with:

```bash
python main.py
```

The program checks camera indices `0` through `9` using the Windows DirectShow backend. Allow camera access in Windows if prompted.

## How it works

### Camera and interface

`main.py` opens the first camera that successfully returns a frame. Tkinter owns the application window and schedules subsequent frame reads with `after()`. Each frame is passed to the face and hand recognizers, converted from OpenCV's BGR color format to RGB, and displayed alongside live counts and per-face/per-hand results. Closing the window releases the camera and hand-landmarker resources.

### Face detection and visual estimation

`face_recognition.py` converts each camera frame to grayscale and equalizes its histogram to improve contrast. Two OpenCV Haar cascade classifiers (`haarcascade_frontalface_default.xml` and `haarcascade_frontalface_alt2.xml`) independently propose face rectangles. The application keeps a candidate only when a rectangle from the second detector overlaps it sufficiently, measured by intersection over union (IoU). A minimum face size and higher `minNeighbors` values make detection more selective.

For each accepted rectangle, the program crops the corresponding color image and resizes/normalizes it into the `227 x 227` input expected by the local Caffe gender-classification network. The network is defined by `models/deploy.prototxt` and its learned weights are in `models/gender_net.caffemodel`. Its two output scores are mapped to `male` and `female`; the larger score supplies both the displayed label and percentage. Predictions are refreshed at most every half second unless the number of detected faces changes. This model infers a binary appearance category from a face crop; it cannot establish a person's gender identity, and its score is not a calibrated measure of real-world accuracy.

### Hand landmarks and finger counting

`hand_recognition.py` uses MediaPipe Tasks' Hand Landmarker with the local `models/hand_landmarker.task` asset. The video running mode uses monotonically increasing timestamps to process frames and track up to two hands. For each detected hand, its 21 normalized landmarks are scaled to camera-frame coordinates, connected with the hand skeleton, and shown in orange.

Finger counting is a geometric heuristic over those landmarks. For the four non-thumb fingers, it checks whether the finger is sufficiently straight at its middle joint and whether its tip is farther from the wrist than that joint. The thumb uses its own joint-angle and wrist-distance checks. The resulting estimate is drawn above the hand and included in the total shown in the data panel. Hand orientation, occlusion, camera angle, and landmark quality can affect the count.

### Project layout

```text
main.py                 Tkinter window and camera lifecycle
face_recognition.py     Face detection and face-crop classification
hand_recognition.py     Hand landmarks, drawing, and finger counting
requirements.txt        Python package dependencies
models/                 Local model assets used at runtime
```

## Limitations

- This is a small computer-vision demo, not an identity, medical, or security system.
- Haar cascades can miss profiles or poorly lit faces and may still produce false detections.
- Face-category predictions and finger counts can be wrong, especially with occlusion or unusual poses.
- The hand-landmarker and Caffe network definition assets are required locally; missing
  gender-network weights are downloaded at first start when an internet connection is
  available.
