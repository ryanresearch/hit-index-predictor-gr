from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

class SongFeatures(BaseModel):
    spotify: int
    shazam: int
    tiktok: int
    youtube: int
    airplay_score: int
    tiktok_streams: int
    mentions: int

class SongRequest(BaseModel):
    songs: list[SongFeatures]

model = joblib.load("model.pkl")

@app.post("/predict")
def predict_probabilities(request: SongRequest):
    X = [[
        song.spotify,
        song.shazam,
        song.tiktok,
        song.youtube,
        song.airplay_score,
        song.tiktok_streams,
        song.mentions
    ] for song in request.songs]

    probabilities = model.predict_proba(np.array(X))[:, 1]
    return {"probabilities": probabilities.tolist()}
