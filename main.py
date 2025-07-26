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
    probabilities = model.predict_proba(input_data)[:, 1]
    return {"probabilities": probabilities.tolist()}

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

    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, "model.pkl")

    return {"message": "Model retrained successfully."}
