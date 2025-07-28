from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from io import StringIO
import os

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the features model
class SongFeatures(BaseModel):
    spotify: int
    shazam: int
    tiktok: int
    youtube: int
    airplay_score: int
    tiktok_streams: int
    mentions: int

class SongsRequest(BaseModel):
    songs: List[SongFeatures]

# Predict endpoint
@app.post("/predict")
def predict(request: SongsRequest):
    try:
        model = joblib.load("model.pkl")
    except:
        return {"error": "No trained model available."}

    input_data = pd.DataFrame([song.dict() for song in request.songs])

    try:
        proba = model.predict_proba(input_data)
        # Normalize to 0–1 scale using class weights (0, 1, 2 → scaled)
        probabilities = []
        for p in proba:
            # Weighted sum: 0×p0 + 0.5×p1 + 1×p2
            score = round((0.5 * p[1] + 1.0 * p[2]), 3)
            probabilities.append(min(score, 1.0))
    except:
        return {"error": "Prediction failed."}

    return {"probabilities": probabilities}

# Retrain endpoint
@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    required_columns = ["spotify", "shazam", "tiktok", "youtube", "airplay_score", "tiktok_streams", "mentions", "label"]
    if not all(col in df.columns for col in required_columns):
        return {"error": "Missing required columns."}

    X = df[required_columns[:-1]]
    y = df["label"]

    # Use multiclass classifier (for 0, 1, 2 classes)
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, "model.pkl")

    return {"message": "Model retrained successfully."}
