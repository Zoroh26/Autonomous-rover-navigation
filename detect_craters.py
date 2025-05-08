import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import matplotlib.pyplot as plt
import networkx as nx

# Load trained model
model = YOLO("C:/Users/rohit/Downloads/MINI_PROJECT_GIT/MINI_PROJECT_GIT/runs/detect/train2/weights/best.pt")

global points
points = []

def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        cv2.circle(image_display, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("Select Start & Destination", image_display)

def upload_image():
    global image, image_display
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return
    
    # Load image
    image = cv2.imread(file_path)
    image_display = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run crater detection
    results = model(image_rgb)
    crater_boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crater_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(image_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow("Select Start & Destination", image_display)
    cv2.setMouseCallback("Select Start & Destination", select_point)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(points) == 2:
        find_safe_path(crater_boxes)
    

def find_safe_path(crater_boxes):
    start, destination = points
    h, w, _ = image.shape
    
    # Create a graph representation of the image
    graph = nx.grid_2d_graph(w, h)
    
    # Remove nodes that are inside craters
    for x1, y1, x2, y2 in crater_boxes:
        for i in range(x1, x2):
            for j in range(y1, y2):
                if (i, j) in graph:
                    graph.remove_node((i, j))
    
    # Find shortest path avoiding craters
    try:
        path = nx.astar_path(graph, start, destination)
        for (x, y) in path:
            cv2.circle(image_display, (x, y), 1, (0, 0, 255), -1)
    except:
        print("No safe path found!")
    
    # Show final output
    plt.imshow(cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# GUI to upload image
root = tk.Tk()
root.withdraw()
upload_image()
