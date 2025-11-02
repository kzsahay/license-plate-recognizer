from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from io import BytesIO

app = FastAPI(title="License Plate & VIN OCR API", version="1.0")

# === CONFIGURATION ===
# Update this path to your trained or downloaded YOLO model
PLATE_MODEL_PATH = "plate_detector.pt"  # Will be downloaded later if missing

# Add more languages for global coverage (EasyOCR auto-downloads models)
# Full list: https://www.jaided.ai/easyocr/
OCR_LANGUAGES = ['en', 'ar', 'hi', 'ru', 'th', 'fr', 'de', 'es', 'pt', 'it', 'ja', 'ko', 'ch_sim', 'ch_tra']

# Load models at startup (once)
try:
    plate_model = YOLO(PLATE_MODEL_PATH)
    print(f"Loaded YOLO model: {PLATE_MODEL_PATH}")
except Exception as e:
    print(f"Could not load YOLO model: {e}")
    print("Run the download script below to get a pre-trained model.")
    plate_model = None

ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)  # CPU-only in Codespaces


# === ENDPOINTS ===
@app.get("/")
async def root():
    return {"message": "License Plate & VIN OCR API. Go to /docs for Swagger UI."}


@app.post("/recognize-plate/")
async def recognize_plate(file: UploadFile = File(...)):
    if plate_model is None:
        return {"error": "YOLO plate detection model not loaded."}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image"}

    results = plate_model(img, verbose=False)
    plates = []

    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  # class 0 = 'plate'
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cropped = img[y1:y2, x1:x2]

                # Optional: Enhance contrast
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                enhanced = cv2.equalizeHist(gray)
                _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # OCR
                ocr_result = ocr_reader.readtext(thresh, detail=0, paragraph=False)
                text = ' '.join(ocr_result).upper().strip()
                # Clean common OCR mistakes
                text = text.replace('O', '0').replace('I', '1').replace('S', '5')
                plates.append(text)

    return {"plates": plates if plates else "No plate detected"}


@app.post("/recognize-vin/")
async def recognize_vin(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image"}

    # Grayscale + threshold for VIN
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Restrict to VIN allowed chars
    ocr_result = ocr_reader.readtext(
        thresh,
        detail=0,
        allowlist='0123456789ABCDEFGHJKLMNPRSTUVWXYZ',  # No I, O, Q
        paragraph=False
    )
    vin = ''.join(ocr_result).upper().strip()

    # Basic VIN validation
    if len(vin) == 17 and all(c in '0123456789ABCDEFGHJKLMNPRSTUVWXYZ' for c in vin):
        return {"vin": vin, "valid": True}
    else:
        return {"vin": vin, "valid": False, "raw": ocr_result}


# === Health Check ===
@app.get("/health")
async def health():
    return {"status": "OK", "yolo_loaded": plate_model is not None}
