from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/predict")
def predict(request: SongsRequest):
    # Dummy example - Replace with real ML logic
    probabilities = []
    for song in request.songs:
        score = (
            0.2 * song.spotify +
            0.2 * song.shazam +
            0.2 * song.tiktok +
            0.2 * song.youtube +
            0.1 * song.airplay_score +
            0.05 * song.tiktok_streams +
            0.05 * song.mentions
        ) / 10
        probabilities.append(min(score, 1.0))  # Make sure it's 0–1

    return {"probabilities": probabilities}

