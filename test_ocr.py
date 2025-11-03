# test_ocr.py
import easyocr
import cv2
from ultralytics import YOLO

# Load Thai OCR
reader = easyocr.Reader(['th', 'en'], gpu=False)
print("Thai OCR loaded")

# Load YOLO to get bounding box
model = YOLO('plate_detector.pt')
results = model("thai_plate.jpg", verbose=False)
boxes = results[0].boxes

if len(boxes) > 0:
    x1, y1, x2, y2 = map(int, boxes[0].xyxy[0].tolist())
    img = cv2.imread("thai_plate.jpg")
    cropped = img[y1:y2, x1:x2]
    cv2.imwrite("cropped_thai.jpg", cropped)

    # Preprocess
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR
    result = reader.readtext(thresh, detail=1)
    print("OCR Results:")
    for (bbox, text, conf) in result:
        print(f"  → '{text}' (Confidence: {conf:.3f})")
else:
    print("No plate detected")