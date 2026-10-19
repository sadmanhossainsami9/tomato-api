import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import pipeline

MODEL_ID = "wellCh4n/tomato-leaf-disease-classification-resnet50"
classifier = pipeline("image-classification", model=MODEL_ID)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def home():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    results = classifier(img, top_k=3)
    return {"predictions": [{"label": r["label"], "confidence": float(r["score"])} for r in results]}
