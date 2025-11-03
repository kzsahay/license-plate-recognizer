# --------------------------------------------------------------
# main.py  (FINAL VERSION – UI WORKS 100%)
# --------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from io import BytesIO
import threading
from typing import List, Dict, Any
import os

app = FastAPI(title="Global OCR API", version="1.0")

# ------------------- 1. CONFIG -------------------
PLATE_MODEL_PATH = "plate_detector.pt"

# List all desired languages
DESIRED_LANGUAGES = [
    'en', 'ar', 'hi', 'ru', 'fr', 'de', 'es', 'pt', 'it',
    'th', 'ja', 'ko', 'ch_sim', 'ch_tra',
    'bg', 'cs', 'da', 'nl', 'fi', 'el', 'he', 'hu', 'id',
    'no', 'pl', 'ro', 'sk', 'sv', 'tr', 'uk', 'vi'
]

RESTRICTED_LANGS = {'th', 'ja', 'ko', 'ch_sim', 'ch_tra'}

# ------------------- 2. AUTO-SPLIT LANGUAGES -------------------
main_langs = ['en'] + [l for l in DESIRED_LANGUAGES if l not in RESTRICTED_LANGS]
restricted_groups = [[lang, 'en'] for lang in RESTRICTED_LANGS if lang in DESIRED_LANGUAGES]

# ------------------- 3. OCR READERS -------------------
ocr_readers: Dict[str, Any] = {}
lock = threading.Lock()

def load_reader(langs: List[str], name: str):
    try:
        reader = easyocr.Reader(langs, gpu=False, download_enabled=True)
        with lock:
            ocr_readers[name] = reader
        print(f"{name.upper()} OCR loaded: {langs}")
    except Exception as e:
        print(f"Failed to load {name} OCR: {e}")

load_reader(main_langs, 'main')
for i, langs in enumerate(restricted_groups):
    threading.Thread(target=load_reader, args=(langs, langs[0]), daemon=True).start()

# ------------------- 4. YOLO -------------------
try:
    plate_model = YOLO(PLATE_MODEL_PATH)
    print(f"YOLO loaded: {PLATE_MODEL_PATH}")
except Exception as e:
    print(f"YOLO failed: {e}")
    plate_model = None

# ------------------- 5. OCR HELPER -------------------
def run_ocr(image: np.ndarray) -> str:
    best = ""
    for name, reader in list(ocr_readers.items()):
        try:
            res = reader.readtext(image, detail=0, paragraph=False)
            txt = ' '.join(res).upper().strip()
            txt = txt.replace('O', '0').replace('I', '1').replace('S', '5').replace('B', '8')
            if len(txt) > len(best):
                best = txt
        except:
            continue
    return best

# ------------------- 6. API ENDPOINTS -------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve UI at root"""
    if os.path.exists("ui.html"):
        with open("ui.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("""
    <h1>Global OCR API</h1>
    <p>UI not found. Create <code>ui.html</code> or go to <a href="/ui">/ui</a></p>
    """)

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    if os.path.exists("ui.html"):
        with open("ui.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>ui.html not found</h1>")

@app.post("/recognize-plate/")
async def recognize_plate(file: UploadFile = File(...)):
    if plate_model is None:
        return {"error": "YOLO not loaded"}
    if not ocr_readers:
        return {"error": "OCR loading... wait 10s"}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image"}

    results = plate_model(img, verbose=False)
    plates = []

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cropped = img[y1:y2, x1:x2]
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text = run_ocr(thresh)
                if text:
                    plates.append(text)

    return {"plates": plates if plates else "No plate detected"}

@app.post("/recognize-vin/")
async def recognize_vin(file: UploadFile = File(...)):
    if 'main' not in ocr_readers:
        return {"error": "OCR not ready"}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    res = ocr_readers['main'].readtext(
        thresh,
        detail=0,
        allowlist='0123456789ABCDEFGHJKLMNPRSTUVWXYZ',
        paragraph=False
    )
    vin = ''.join(res).upper().strip()
    valid = len(vin) == 17 and all(c in '0123456789ABCDEFGHJKLMNPRSTUVWXYZ' for c in vin)
    return {"vin": vin, "valid": valid}

@app.get("/health")
async def health():
    return {
        "status": "OK",
        "yolo": plate_model is not None,
        "ocr_loaded": list(ocr_readers.keys()),
        "total_ocr": len(ocr_readers)
    }

# ------------------- 7. STATIC FILES (Optional) -------------------
# Only needed if you want to serve other files
app.mount("/static", StaticFiles(directory=".", html=True), name="static")