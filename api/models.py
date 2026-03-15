from pydantic import BaseModel, Field
from typing import List, Optional


class Transaction(BaseModel):
    """
    Model for a single credit card transaction.
    Features match the dataset: Time, V1-V28, Amount.
    """

    Time: float = Field(
        ..., description="Seconds elapsed between transaction and first transaction"
    )
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount")


class PredictionResponse(BaseModel):
    """
    Model for prediction response.
    """

    prediction: int = Field(
        ..., description="Predicted class (0 = legitimate, 1 = fraud)"
    )
    probability: float = Field(..., description="Probability of fraud (class 1)")
    is_fraud: bool = Field(..., description="True if predicted as fraud")


class HealthResponse(BaseModel):
    """
    Model for health check response.
    """

    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
