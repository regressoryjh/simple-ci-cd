import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

print("Starting model testing...")

# Load data and model
df = pd.read_csv('data/data.csv')
model = joblib.load('model.pkl')

X = df[['feature1', 'feature2']]
y = df['target']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model accuracy: {accuracy}")

# Check if accuracy meets threshold
# Note: Ini adalah contoh, sesuaikan ambang batas Anda
if accuracy < 0.85:
    print("Model performance is too low! Exiting...")
    exit(1) # Keluar dengan kode error
else:
    print("Model performance is satisfactory!")