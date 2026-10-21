import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from face_recognition import FaceRecognizer
from hand_recognition import HandRecognizer


CAMERA_INDICES = range(10)
CAMERA_SIZE = (640, 480)
PANEL_WIDTH = 270


def find_camera():
    # Probe common Windows camera indices and keep the first readable device.
    for index in CAMERA_INDICES:
        camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if camera.isOpened():
            success, frame = camera.read()
            if success:
                return camera, frame
        camera.release()
    return None, None


class CameraWindow:
    def __init__(self, root, camera, face_recognizer, hand_recognizer, first_frame):
        self.root = root
        self.camera = camera
        self.face_recognizer = face_recognizer
        self.hand_recognizer = hand_recognizer

        self.image_label = tk.Label(
            root, bg="black", width=CAMERA_SIZE[0], height=CAMERA_SIZE[1]
        )
        self.image_label.pack(side=tk.LEFT, padx=(0, 12))

        self.data_panel = tk.Frame(root, width=PANEL_WIDTH)
        self.data_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.data_panel.pack_propagate(False)
        tk.Label(
            self.data_panel,
            text="Erkennung / Daten",
            font=("TkDefaultFont", 12, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 8))
        self.data_text = tk.Text(self.data_panel, wrap=tk.WORD, state=tk.DISABLED)
        self.data_text.pack(fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.show_frame(first_frame)
        self.update_frame()

    def show_frame(self, frame):
        # Apply both recognizers before converting the frame for Tkinter.
        frame, gender_results = self.face_recognizer.recognize(frame)
        frame, finger_counts = self.hand_recognizer.recognize(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image.thumbnail(CAMERA_SIZE)
        self.frame_image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.frame_image)

        self.data_text.configure(state=tk.NORMAL)
        self.data_text.delete("1.0", tk.END)
        face_lines = [
            f"Gesicht {index}: {label} ({confidence:.0%})"
            for index, (label, confidence) in enumerate(gender_results, start=1)
        ]
        if not face_lines:
            face_lines.append("Keine Gesichter erkannt.")
        hand_lines = [
            f"Hand {index}: {finger_count} Finger"
            for index, finger_count in enumerate(finger_counts, start=1)
        ]
        if not hand_lines:
            hand_lines.append("Keine Hände erkannt.")
        self.data_text.insert(
            tk.END,
            "Kamera: verbunden\n\n"
            f"Erkannte Gesichter: {len(gender_results)}\n"
            f"Erkannte Hände: {len(finger_counts)}\n"
            f"Gezeigte Finger insgesamt: {sum(finger_counts)}\n\n"
            + "\n".join(face_lines)
            + "\n\n"
            + "\n".join(hand_lines)
            + "\n\n"
            ""
            "",
        )
        self.data_text.configure(state=tk.DISABLED)

    def update_frame(self):
        # Schedule the next capture without blocking Tkinter's event loop.
        success, frame = self.camera.read()
        if success:
            self.show_frame(frame)
        else:
            self.close()
            messagebox.showerror(
                "Kamerafehler", "Das Kamerabild konnte nicht mehr gelesen werden."
            )
            return
        self.root.after(30, self.update_frame)

    def close(self):
        self.camera.release()
        self.hand_recognizer.close()
        self.root.destroy()


def show_error(title, message):
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, message)
    root.destroy()


def main():
    try:
        face_recognizer = FaceRecognizer()
        hand_recognizer = HandRecognizer()
    except (ImportError, OSError, cv2.error, RuntimeError) as error:
        show_error(
            "Erkennung nicht verfügbar",
            "Die Gesichts- oder Handerkennung konnte nicht initialisiert werden. "
            "Prüfe die Installation der Abhängigkeiten und die Internetverbindung "
            "beim ersten Start.\n\n"
            f"Details: {error}",
        )
        return

    camera, first_frame = find_camera()
    if camera is None:
        show_error(
            "Keine Kamera gefunden",
            "Es wurde keine verfügbare Kamera gefunden. "
            "Bitte verbinde eine Kamera und starte das Programm erneut.",
        )
        return

    root = tk.Tk()
    root.title("Kamera")
    root.geometry(f"{CAMERA_SIZE[0] + PANEL_WIDTH + 24}x{CAMERA_SIZE[1]}")
    root.resizable(False, False)
    CameraWindow(root, camera, face_recognizer, hand_recognizer, first_frame)
    root.mainloop()


if __name__ == "__main__":
    main()
