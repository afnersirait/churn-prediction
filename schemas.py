from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime


class CustomerInput(BaseModel):
    customer_id: str
    tenure: int = Field(..., ge=0, description="Number of months the customer has been with the company")
    monthly_charges: float = Field(..., gt=0, description="Monthly charges in dollars")
    total_charges: float = Field(..., ge=0, description="Total charges in dollars")
    contract_type: str = Field(..., description="Month-to-month, One year, or Two year")
    payment_method: str = Field(..., description="Electronic check, Mailed check, Bank transfer, or Credit card")
    internet_service: str = Field(..., description="DSL, Fiber optic, or No")
    online_security: str = Field(..., description="Yes, No, or No internet service")
    tech_support: str = Field(..., description="Yes, No, or No internet service")
    streaming_tv: str = Field(..., description="Yes, No, or No internet service")
    streaming_movies: str = Field(..., description="Yes, No, or No internet service")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST001234",
                "tenure": 24,
                "monthly_charges": 79.99,
                "total_charges": 1919.76,
                "contract_type": "One year",
                "payment_method": "Credit card",
                "internet_service": "Fiber optic",
                "online_security": "Yes",
                "tech_support": "Yes",
                "streaming_tv": "Yes",
                "streaming_movies": "No"
            }
        }


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    will_churn: bool
    risk_level: str
    model_version: Optional[str] = None
    timestamp: datetime


class ExplainedPredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    will_churn: bool
    risk_level: str
    explanation: Dict
    top_risk_factors: List[Dict[str, float]]
    model_version: Optional[str] = None
    timestamp: datetime


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerInput]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    timestamp: datetime


class ModelMetricsResponse(BaseModel):
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    trained_at: datetime


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_engineer_loaded: bool
    database_connected: bool
    mlflow_connected: bool
    timestamp: datetime
