import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ModelService:
    """
    Service for loading the fraud detection model and performing predictions.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = "models/linear_regression/linear_reg_baseline_model.pkl"
        self.scaler_path = "models/linear_regression/linear_reg_scaler.pkl"
        self.is_loaded = False

    def load_model(self) -> bool:
        """Load the model and scaler from disk."""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False

            if not os.path.exists(self.scaler_path):
                logger.error(f"Scaler file not found: {self.scaler_path}")
                return False

            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_loaded = True
            logger.info("Model and scaler loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a single transaction.

        Args:
            transaction_data: Dictionary containing transaction features

        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Convert to DataFrame with correct column order
            features = [
                "Time",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
                "V7",
                "V8",
                "V9",
                "V10",
                "V11",
                "V12",
                "V13",
                "V14",
                "V15",
                "V16",
                "V17",
                "V18",
                "V19",
                "V20",
                "V21",
                "V22",
                "V23",
                "V24",
                "V25",
                "V26",
                "V27",
                "V28",
                "Amount",
            ]

            # Ensure all features are present
            for feature in features:
                if feature not in transaction_data:
                    raise ValueError(f"Missing required feature: {feature}")

            # Create DataFrame
            df = pd.DataFrame([transaction_data])[features]

            # Scale features
            scaled_features = self.scaler.transform(df)

            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            probability = self.model.predict_proba(scaled_features)[0][1]

            return {
                "prediction": int(prediction),
                "probability": float(probability),
                "is_fraud": bool(prediction == 1),
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    def predict_batch(self, transactions_data: list) -> list:
        """
        Make predictions for multiple transactions.

        Args:
            transactions_data: List of transaction dictionaries

        Returns:
            List of prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            features = [
                "Time",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
                "V7",
                "V8",
                "V9",
                "V10",
                "V11",
                "V12",
                "V13",
                "V14",
                "V15",
                "V16",
                "V17",
                "V18",
                "V19",
                "V20",
                "V21",
                "V22",
                "V23",
                "V24",
                "V25",
                "V26",
                "V27",
                "V28",
                "Amount",
            ]

            # Create DataFrame
            df = pd.DataFrame(transactions_data)[features]

            # Scale features
            scaled_features = self.scaler.transform(df)

            # Make predictions
            predictions = self.model.predict(scaled_features)
            probabilities = self.model.predict_proba(scaled_features)[:, 1]

            results = []
            for i in range(len(predictions)):
                results.append(
                    {
                        "prediction": int(predictions[i]),
                        "probability": float(probabilities[i]),
                        "is_fraud": bool(predictions[i] == 1),
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise


# Global model service instance
model_service = ModelService()
