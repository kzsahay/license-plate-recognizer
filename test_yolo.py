# test_yolo.py
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('plate_detector.pt')
print("YOLO loaded:", model is not None)

# Use the downloaded image
img_path = "thai_plate.jpg"
img = cv2.imread(img_path)

if img is None:
    print(f"Failed to load {img_path} — did you run wget?")
    exit()

print(f"Running YOLO on {img_path}...")
results = model(img, verbose=True)

# Count detections
boxes = results[0].boxes
detections = len(boxes) if boxes is not None else 0
print(f"Detected {detections} license plate(s)")

# Save annotated image
annotated = results[0].plot()
cv2.imwrite("detected_thai.jpg", annotated)
print("Saved: detected_thai.jpg → Open in VS Code!")