from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load or define model (example: manually weighted logistic-style)
def predict_score(features):
    # Feature order: [spotify, shazam, tiktok, youtube, airplay_score, tiktok_streams]
    weights = [0.25, 0.25, 0.2, 0.15, 0.1, 0.05]
    max_streams = 500000  # Normalization cap
    features = np.array(features, dtype=float)
    features[5] = min(features[5], max_streams) / max_streams  # Normalize TikTok streams
    score = np.dot(weights, features)
    return round(float(score), 4)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    features = data['features']  # Must be list of 6 items
    if len(features) != 6:
        return jsonify({'error': 'Expected 6 features'}), 400

    probability = predict_score(features)
    return jsonify({'probability': probability})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
