from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

app = FastAPI()

# Allow all CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input model
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

# Global ML model
model = None

# Predict endpoint
@app.post("/predict")
def predict(request: SongsRequest):
    global model
    if model is None:
        return {"error": "No trained model available."}

    X = pd.DataFrame([song.dict() for song in request.songs])
    probabilities = model.predict_proba(X)[:, 1].tolist()
    return {"probabilities": probabilities}

# Retrain endpoint
@app.post("/retrain")
def retrain(file: UploadFile = File(...)):
    global model
    try:
        contents = file.file.read().decode('utf-8')
        from io import StringIO
        df = pd.read_csv(StringIO(contents))

        required_cols = ['spotify', 'shazam', 'tiktok', 'youtube', 'airplay_score', 'tiktok_streams', 'mentions', 'hit']
        if not all(col in df.columns for col in required_cols):
            return {"error": "Missing required columns."}

        X = df[required_cols[:-1]]
        y = df['hit']

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        joblib.dump(model, "model.pkl")

        return {"message": "Model retrained successfully."}
    except Exception as e:
        return {"error": str(e)}    
