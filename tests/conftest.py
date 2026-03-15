import pytest
import joblib
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from api.services.model_service import ModelService


@pytest.fixture
def sample_transaction():
    """Sample transaction data for testing."""
    return {
        "Time": 10000.0,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 0.538343,
        "V4": -0.092781,
        "V5": 0.089271,
        "V6": 0.363787,
        "V7": 0.239599,
        "V8": 0.168037,
        "V9": 0.057921,
        "V10": 0.239599,
        "V11": -0.338321,
        "V12": 0.103463,
        "V13": 0.156522,
        "V14": 0.059921,
        "V15": -0.039493,
        "V16": -0.024221,
        "V17": -0.067321,
        "V18": -0.051221,
        "V19": -0.089271,
        "V20": 0.089271,
        "V21": 0.057921,
        "V22": -0.039493,
        "V23": -0.024221,
        "V24": -0.067321,
        "V25": -0.051221,
        "V26": -0.089271,
        "V27": 0.089271,
        "V28": 0.057921,
        "Amount": 149.62,
    }


@pytest.fixture
def mock_model_service():
    """Create a mock model service with mocked dependencies."""
    service = ModelService()
    service.model = Mock()
    service.scaler = Mock()
    service.is_loaded = True
    return service


@pytest.fixture
def mock_sklearn_model():
    """Create a mock sklearn model."""
    model = Mock()
    model.predict.return_value = [1]
    model.predict_proba.return_value = [[0.2, 0.8]]  # 80% fraud probability
    return model


@pytest.fixture
def mock_scaler():
    """Create a mock scaler."""
    scaler = Mock()
    scaler.transform.return_value = [[0.1] * 30]  # 30 features
    return scaler


# Note: mock_joblib_files fixture removed as it's not needed with mocked joblib.load
