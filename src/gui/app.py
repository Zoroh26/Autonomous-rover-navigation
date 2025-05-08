import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np

class PathTracingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Path Tracing System")
        
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()
        
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack()
        
        self.proceed_btn = tk.Button(root, text="Proceed", command=self.proceed, state=tk.DISABLED)
        self.proceed_btn.pack()
        
        self.image = None
        self.start_point = None
        self.end_point = None
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return
        
        self.image = cv2.imread(file_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.display_image()
        
        self.canvas.bind("<Button-1>", self.select_points)
        self.start_point = None
        self.end_point = None
        self.proceed_btn.config(state=tk.DISABLED)
        
    def display_image(self):
        img = cv2.resize(self.image, (800, 600))
        img = Image.fromarray(img)
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
    def select_points(self, event):
        x, y = event.x, event.y
        
        if self.start_point is None:
            self.start_point = (x, y)
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="blue", outline="blue")
        elif self.end_point is None:
            self.end_point = (x, y)
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="blue", outline="blue")
            self.proceed_btn.config(state=tk.NORMAL)
    
    def proceed(self):
        print(f"Start Point: {self.start_point}, End Point: {self.end_point}")
        # Here, you will pass the start and end points to the path tracing algorithm

if __name__ == "__main__":
    root = tk.Tk()
    app = PathTracingApp(root)
    root.mainloop()
