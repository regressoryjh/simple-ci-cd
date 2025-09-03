import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from src.model import HousePriceModel  # Fixed import

def main():
    print("🏠 Starting model training...")

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Load dataset (menggunakan California housing dataset)
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {housing.feature_names}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = HousePriceModel()
    model.fit(X_train, y_train)

    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"📊 Training RMSE: {train_rmse:.3f}")
    print(f"📊 Training R²: {train_r2:.3f}")
    print(f"📊 Test RMSE: {test_rmse:.3f}")
    print(f"📊 Test R²: {test_r2:.3f}")

    # Save model using joblib.dump() to be consistent with loading
    joblib.dump(model, 'models/model.pkl')

    # Save test data for validation
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': housing.feature_names
    }
    joblib.dump(test_data, 'models/test_data.pkl')

    print("✅ Model training completed and saved!")

if __name__ == "__main__":
    main()
