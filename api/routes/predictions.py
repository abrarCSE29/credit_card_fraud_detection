from fastapi import APIRouter, HTTPException, status
from api.models import Transaction, PredictionResponse
from api.services.model_service import model_service

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post(
    "/",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict fraud for a single transaction",
    description="Analyze a credit card transaction and predict if it's fraudulent",
)
async def predict_single(transaction: Transaction):
    """
    Predict whether a single credit card transaction is fraudulent.

    - **Time**: Seconds elapsed between transaction and first transaction
    - **V1-V28**: Anonymized features from PCA transformation
    - **Amount**: Transaction amount

    Returns:
    - **prediction**: 0 for legitimate, 1 for fraud
    - **probability**: Probability of fraud (0-1)
    - **is_fraud**: Boolean indicating if predicted as fraud
    """
    try:
        result = model_service.predict(transaction.model_dump())
        return PredictionResponse(**result)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction error: {str(e)}",
        )


@router.post(
    "/batch",
    status_code=status.HTTP_200_OK,
    summary="Predict fraud for multiple transactions",
    description="Analyze multiple credit card transactions and predict which are fraudulent",
)
async def predict_batch(transactions: list[Transaction]):
    """
    Predict whether multiple credit card transactions are fraudulent.

    Accepts a list of transaction objects and returns predictions for each.
    """
    try:
        transactions_data = [t.model_dump() for t in transactions]
        results = model_service.predict_batch(transactions_data)
        return {"predictions": results, "count": len(results)}
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction error: {str(e)}",
        )
