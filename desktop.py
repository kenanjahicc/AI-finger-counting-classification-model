import tkinter as tk
from tkinter import ttk
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image, ImageTk

model = load_model("finger_count_model")

class CameraApp:
    def __init__(self, window, window_title="Finger Classifier"):
        self.window = window
        self.window.title(window_title)
        self.info_label = tk.Label(window, text="Capture an image using the camera:")
        self.info_label.pack(pady=10)
        self.open_camera_button = tk.Button(window, text="Open Camera", command=self.open_camera)
        self.open_camera_button.pack(pady=10)
        self.img_label = tk.Label(window)
        self.img_label.pack(pady=10)
        self.capture_button = ttk.Button(window, text="Capture and Classify", command=self.capture_and_classify)
        self.capture_button.pack(pady=10)
        self.result_label = tk.Label(window, text="")
        self.result_label.pack(pady=20)

        self.cap = None
        self.is_camera_open = False
        self.current_frame = None



    def open_camera(self):
        if not self.is_camera_open:
            self.cap = cv2.VideoCapture(0)
            self.is_camera_open = True
            self.show_camera()

    def show_camera(self):
        if self.is_camera_open:
            ret, frame = self.cap.read()

            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img.thumbnail((400, 400))
                img = ImageTk.PhotoImage(img)

                self.img_label.img = img
                self.img_label.config(image=self.img_label.img)
                self.img_label.image = self.img_label.img

                self.current_frame = frame

                self.window.after(10, self.show_camera)
            else:
                self.close_camera()

    def close_camera(self):
        if self.is_camera_open:
            self.cap.release()
            self.is_camera_open = False
            self.img_label.config(image=None)

    def preprocess_image(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 30, 30], dtype=np.uint8)
        upper_skin = np.array([20, 240, 255], dtype=np.uint8)

        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        preprocessed_img = cv2.bitwise_and(img, img, mask=skin_mask)
        preprocessed_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(preprocessed_img, 1, 255, cv2.THRESH_BINARY)

        return binary_img

    def classify_image(self, img):
        img = self.preprocess_image(img)

        target_height = 400
        aspect_ratio = img.shape[1] / img.shape[0]
        target_width = int(target_height * aspect_ratio)

        resized_img = cv2.resize(img, (target_width, target_height))

        cv2.imwrite("preprocessed_image.png", resized_img)

        self.window.after(10, self.show_preprocessed_image)

        resized_img = cv2.resize(resized_img, (128, 128))
        resized_img = resized_img / 255.0
        resized_img = np.reshape(resized_img, (1, 128, 128, 1))
        predictions = model.predict(resized_img)
        predicted_class = np.argmax(predictions[0])
        return predicted_class


    def show_preprocessed_image(self):
        import subprocess
        subprocess.run(["start", "preprocessed_image.png"], shell=True)

    def capture_and_classify(self):
        if self.is_camera_open and self.current_frame is not None:
            predicted_class = self.classify_image(self.current_frame)
            self.result_label.config(text=f"Classified as: {predicted_class} fingers")


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)

    root.mainloop()



