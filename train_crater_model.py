from ultralytics import YOLO

# Load YOLOv8 model (pre-trained on COCO, we will fine-tune it)
model = YOLO("yolov8n.pt")

# Train on your dataset (adjust epochs and batch size as needed)
results = model.train(
    data="Z:\miniprojdataset\data.yaml",  # Path to dataset config file
    epochs=50,  # Increase if needed for better accuracy
    batch=8,  # Adjust based on GPU memory
    imgsz=640  # Standard image size
)

# Save the trained model
model.export(format="onnx")  # Export as ONNX for easier inference
