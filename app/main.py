from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from app.detector import FruitDetector

app = FastAPI()
detector = FruitDetector()

@app.get("/")
def home():
    return {"message": "Fruit Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = detector.detect(image)

    return {
        "status": "success",
        "predictions": results
    }
