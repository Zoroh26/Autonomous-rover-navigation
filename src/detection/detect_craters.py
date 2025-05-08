import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from src.utils.pathfinder import find_safe_path
from config import TRAINED_MODEL, DATA_DIR


class CraterDetector:
    def __init__(self):
        self.image = None
        self.image_display = None
        self.points = []
        try:
            self.model = YOLO(TRAINED_MODEL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            exit(1)

    def select_point(self, event, x, y, flags, param):
        """Mouse callback for point selection"""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 2:
            self.points.append((x, y))
            cv2.circle(self.image_display, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow("Select Start & Destination", self.image_display)

    def detect_craters(self, image):
        """Run YOLO detection with error handling"""
        try:
            results = self.model(image)
            return [tuple(map(int, box.xyxy[0])) for box in results[0].boxes]
        except Exception as e:
            messagebox.showerror("Detection Error", str(e))
            return []

    def draw_results(self, crater_boxes, path=None):
        """Visualize craters and path with circular detection and offset safe path"""
        self.image_display = self.image.copy()

        # Draw craters as semi-transparent circles
        for box in crater_boxes:
            x1, y1, x2, y2 = box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            width = x2 - x1
            height = y2 - y1
            radius = min(width, height) // 2
            scale_factor = 1.2
            radius = int(radius * scale_factor)
            # Draw circle outline
            cv2.circle(self.image_display, center, radius, (0, 255, 0), 2)

        # Draw path if exists
        if path:
            offset = 5  # Offset to prevent overlap
            for i in range(1, len(path)):
                start_point = (path[i-1][0] + offset, path[i-1][1] + offset)
                end_point = (path[i][0] + offset, path[i][1] + offset)
                cv2.line(self.image_display, start_point, end_point, (0, 0, 255), 4)  # Thicker red line

        # Show with matplotlib for better quality
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(self.image_display, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Crater Detection with Safe Path" if path else "Crater Detection")
        plt.show()

    def process_image(self):
        """Main processing pipeline"""
        # Reset points for new image
        self.points = []

        # File selection
        file_path = filedialog.askopenfilename(
            initialdir=DATA_DIR,
            filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        # Load image
        self.image = cv2.imread(file_path)
        if self.image is None:
            messagebox.showerror("Error", "Could not load image!")
            return

        # Detect craters
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        crater_boxes = self.detect_craters(image_rgb)

        if not crater_boxes:
            messagebox.showwarning("Warning", "No craters detected!")
            return

        # Initial display of craters
        self.draw_results(crater_boxes)

        # Point selection
        self.image_display = self.image.copy()
        for box in crater_boxes:
            x1, y1, x2, y2 = box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            radius = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2)
            cv2.circle(self.image_display, center, radius, (0, 255, 0), 2)

        cv2.namedWindow("Select Start & Destination", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select Start & Destination", self.select_point)
        cv2.imshow("Select Start & Destination", self.image_display)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or len(self.points) == 2:  # ESC or 2 points selected
                break

        cv2.destroyAllWindows()

        # Pathfinding if 2 points selected
        if len(self.points) == 2:
            path = find_safe_path(
                start=self.points[0],
                end=self.points[1],
                crater_boxes=crater_boxes,
                image_shape=self.image.shape
            )

            if path:
                self.draw_results(crater_boxes, path)
            else:
                messagebox.showinfo("Info", "No safe path exists between points")


def main():
    root = tk.Tk()
    root.withdraw()

    detector = CraterDetector()
    detector.process_image()


if __name__ == "__main__":
    main()