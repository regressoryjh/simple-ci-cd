import pytest
import numpy as np
from sklearn.datasets import make_regression
from src.model import HousePriceModel  # Fixed import

class TestHousePriceModel:

    @pytest.fixture
    def sample_data(self):
        """Generate sample regression data for testing"""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        return X, y

    def test_model_initialization(self):
        """Test model can be initialized"""
        model = HousePriceModel()
        assert model.is_fitted == False
        assert model.model is not None
        assert model.scaler is not None

    def test_model_fitting(self, sample_data):
        """Test model can be fitted"""
        X, y = sample_data
        model = HousePriceModel()

        fitted_model = model.fit(X, y)
        assert fitted_model.is_fitted == True
        assert fitted_model is model  # Should return self

    def test_model_prediction(self, sample_data):
        """Test model can make predictions"""
        X, y = sample_data
        model = HousePriceModel()
        model.fit(X, y)

        predictions = model.predict(X[:10])
        assert len(predictions) == 10
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)

    def test_prediction_without_fitting(self, sample_data):
        """Test that prediction fails when model is not fitted"""
        X, y = sample_data
        model = HousePriceModel()

        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            model.predict(X[:10])

    def test_model_save_load(self, sample_data, tmp_path):
        """Test model can be saved and loaded"""
        X, y = sample_data

        # Train and save model
        model = HousePriceModel()
        model.fit(X, y)

        model_path = tmp_path / "test_model.pkl"
        model.save(str(model_path))

        # Load model and test
        loaded_model = HousePriceModel.load(str(model_path))
        assert loaded_model.is_fitted == True

        # Predictions should be the same
        original_pred = model.predict(X[:5])
        loaded_pred = loaded_model.predict(X[:5])
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)