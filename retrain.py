import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your generated dataset
df = pd.read_csv("training_data.csv")

# Features and label
X = df.drop(columns=['label'])
y = df['label']

# Split for validation (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

# (Optional) Show validation accuracy
print(f"Validation Accuracy: {model.score(X_test, y_test):.2f}")
