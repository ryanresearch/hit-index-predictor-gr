from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import os

app = FastAPI()

# CORS (dev mode)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input structure
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

# Load trained model if available
model_path = "model.pkl"
model = joblib.load(model_path) if os.path.exists(model_path) else None

@app.post("/predict")
def predict(request: SongsRequest):
    global model
    if not model:
        return {"error": "No trained model available."}
    
    df = pd.DataFrame([song.dict() for song in request.songs])
    predictions = model.predict_proba(df)[:, 1]
    return {"probabilities": predictions.tolist()}

@app.get("/retrain")
def retrain():
    global model
    try:
        df = pd.read_csv("training_data.csv")
        if "label" not in df.columns:
            return {"error": "'label' column missing from training_data.csv"}

        X = df.drop(columns=["label"])
        y = df["label"]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)

        return {"status": "Retrained successfully", "samples": len(df)}
    except Exception as e:
        return {"error": str(e)}
