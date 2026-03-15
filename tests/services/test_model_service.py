import pytest
from unittest.mock import Mock, patch, MagicMock
import joblib
import pandas as pd
from api.services.model_service import ModelService


class TestModelService:
    """Test cases for ModelService."""

    def test_model_service_initialization(self):
        """Test that ModelService initializes correctly."""
        service = ModelService()
        assert service.model is None
        assert service.scaler is None
        assert service.is_loaded is False
        assert "linear_reg_baseline_model.pkl" in service.model_path
        assert "linear_reg_scaler.pkl" in service.scaler_path

    @patch("api.services.model_service.joblib.load")
    @patch("api.services.model_service.os.path.exists")
    def test_load_model_success(
        self, mock_exists, mock_load, mock_sklearn_model, mock_scaler
    ):
        """Test successful model loading."""
        mock_exists.return_value = True
        # Configure mock to return different values for different calls
        mock_load.side_effect = [mock_sklearn_model, mock_scaler]

        service = ModelService()
        result = service.load_model()

        assert result is True
        assert service.is_loaded is True
        assert service.model == mock_sklearn_model
        assert service.scaler == mock_scaler

    @patch("api.services.model_service.os.path.exists")
    def test_load_model_missing_model_file(self, mock_exists):
        """Test model loading when model file is missing."""
        mock_exists.return_value = False

        service = ModelService()
        result = service.load_model()

        assert result is False
        assert service.is_loaded is False

    @patch("api.services.model_service.joblib.load")
    @patch("api.services.model_service.os.path.exists")
    def test_load_model_exception(self, mock_exists, mock_load):
        """Test model loading when an exception occurs."""
        mock_exists.return_value = True
        mock_load.side_effect = Exception("Failed to load model")

        service = ModelService()
        result = service.load_model()

        assert result is False
        assert service.is_loaded is False

    def test_predict_without_loading_model(self, sample_transaction):
        """Test prediction when model is not loaded."""
        service = ModelService()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            service.predict(sample_transaction)

    def test_predict_success(
        self, mock_model_service, sample_transaction, mock_sklearn_model, mock_scaler
    ):
        """Test successful prediction."""
        mock_model_service.model = mock_sklearn_model
        mock_model_service.scaler = mock_scaler

        result = mock_model_service.predict(sample_transaction)

        assert result["prediction"] == 1
        assert result["probability"] == 0.8
        assert result["is_fraud"] is True

    def test_predict_missing_feature(self, mock_model_service, sample_transaction):
        """Test prediction with missing required feature."""
        del sample_transaction["V1"]  # Remove a required feature

        with pytest.raises(ValueError, match="Missing required feature"):
            mock_model_service.predict(sample_transaction)

    def test_predict_batch_success(
        self, mock_model_service, sample_transaction, mock_sklearn_model, mock_scaler
    ):
        """Test successful batch prediction."""
        import numpy as np

        mock_model_service.model = mock_sklearn_model
        mock_model_service.scaler = mock_scaler

        # Mock predict and predict_proba for batch (return numpy arrays)
        mock_model_service.model.predict.return_value = np.array([0, 1])
        mock_model_service.model.predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.2, 0.8]]
        )

        transactions = [sample_transaction, sample_transaction]
        result = mock_model_service.predict_batch(transactions)

        assert len(result) == 2
        assert result[0]["prediction"] == 0
        assert result[0]["probability"] == 0.1
        assert result[0]["is_fraud"] is False
        assert result[1]["prediction"] == 1
        assert result[1]["probability"] == 0.8
        assert result[1]["is_fraud"] is True

    def test_predict_batch_without_loading_model(self, sample_transaction):
        """Test batch prediction when model is not loaded."""
        service = ModelService()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            service.predict_batch([sample_transaction])
