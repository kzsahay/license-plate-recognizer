import os
from ultralytics import YOLO

# Download a lightweight, pre-trained license plate detector
# Source: Public YOLOv8 model trained on global plates
MODEL_URL = "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/best.pt"
MODEL_PATH = "plate_detector.pt"

if not os.path.exists(MODEL_PATH):
    print("Downloading pre-trained license plate detector...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
else:
    print(f"Model already exists: {MODEL_PATH}")

# Test load
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")
