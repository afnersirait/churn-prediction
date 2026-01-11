from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    database_url: str = "postgresql://user:password@localhost:5432/churn_db"
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_registry_name: str = "churn_prediction_model"
    retrain_threshold: float = 0.85
    
    model_path: str = "models/churn_model.pkl"
    data_path: str = "data"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
