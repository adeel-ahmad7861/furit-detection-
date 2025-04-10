import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
from PIL import Image, ImageTk

# Load YOLOv8 model (replace with 'best.pt' if you have trained it on fruits)
model = YOLO("yolov8n.pt")  # Or 'yolov8n.pt' for basic demo

# GUI App
class FruitDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Detector with YOLOv8")

        self.label = tk.Label(root, text="Choose an option", font=("Arial", 16))
        self.label.pack(pady=10)

        self.btn_realtime = tk.Button(root, text="📷 Real-time Detection", command=self.realtime_detection, width=30)
        self.btn_realtime.pack(pady=10)

        self.btn_image = tk.Button(root, text="🖼️ Detect from Image", command=self.detect_from_image, width=30)
        self.btn_image.pack(pady=10)

    def realtime_detection(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            messagebox.showerror("Error", "Webcam not accessible")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()
            cv2.imshow("Real-time Fruit Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def detect_from_image(self):
        file_path = filedialog.askopenfilename(title="Select Fruit Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return

        image = cv2.imread(file_path)
        results = model(image)
        annotated = results[0].plot()

        # Extract fruit names
        names = model.names
        fruit_names = [names[int(box.cls)] for box in results[0].boxes]
        unique_fruits = list(set(fruit_names))

        # Show result
        for fruit in unique_fruits:
            print(f"🍎 Detected: {fruit}")

        cv2.imshow("Image Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = FruitDetectorApp(root)
    root.mainloop()
