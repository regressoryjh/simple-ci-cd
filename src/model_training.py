import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

print("Starting model training...")

# Load data
df = pd.read_csv('data/data.csv')
X = df[['feature1', 'feature2']] # contoh fitur
y = df['target'] # contoh target

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model.pkl')
print("Model trained and saved as model.pkl")