from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

class HousePriceModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X, y):
        """Train the model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, filepath):
        """Save model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)

    @classmethod
    def load(cls, filepath):
        """Load model from file"""
        model_data = joblib.load(filepath)
        instance = cls()
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.is_fitted = model_data['is_fitted']
        return instance