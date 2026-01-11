from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np
import os

from schemas import (
    CustomerInput, PredictionResponse, ExplainedPredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse, ModelMetricsResponse,
    HealthResponse
)
from model import ChurnPredictor
from feature_engineering import AutomatedFeatureEngineer, prepare_features
from explainability import ExplainableAI
from database import get_db, init_db, Prediction, ModelMetrics, Customer
from retrain_pipeline import RetrainingPipeline
from config import settings

app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML-powered API for predicting customer churn with explainable AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: ChurnPredictor = None
feature_engineer: AutomatedFeatureEngineer = None
explainer: ExplainableAI = None


@app.on_event("startup")
async def startup_event():
    global model, feature_engineer, explainer
    
    init_db()
    print("Database initialized")
    
    try:
        model = ChurnPredictor.load()
        feature_engineer = AutomatedFeatureEngineer.load()
        print("Model and feature engineer loaded successfully")
        
        if os.path.exists("data/train.csv"):
            train_data = pd.read_csv("data/train.csv")
            X_train, _, _ = prepare_features(train_data, feature_engineer, fit=False)
            explainer = ExplainableAI(model, X_train[:100])
            print("Explainer initialized")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Please train a model first by running: python retrain_pipeline.py")


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


def customer_to_dataframe(customer: CustomerInput) -> pd.DataFrame:
    return pd.DataFrame([{
        "customer_id": customer.customer_id,
        "tenure": customer.tenure,
        "monthly_charges": customer.monthly_charges,
        "total_charges": customer.total_charges,
        "contract_type": customer.contract_type,
        "payment_method": customer.payment_method,
        "internet_service": customer.internet_service,
        "online_security": customer.online_security,
        "tech_support": customer.tech_support,
        "streaming_tv": customer.streaming_tv,
        "streaming_movies": customer.streaming_movies
    }])


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_with_explanation": "/predict/explain",
            "batch_predict": "/predict/batch",
            "retrain": "/model/retrain",
            "metrics": "/model/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check(db: Session = Depends(get_db)):
    database_connected = True
    try:
        db.execute("SELECT 1")
    except:
        database_connected = False
    
    mlflow_connected = True
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    except:
        mlflow_connected = False
    
    return HealthResponse(
        status="healthy" if model and feature_engineer else "degraded",
        model_loaded=model is not None,
        feature_engineer_loaded=feature_engineer is not None,
        database_connected=database_connected,
        mlflow_connected=mlflow_connected,
        timestamp=datetime.utcnow()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_churn(customer: CustomerInput, db: Session = Depends(get_db)):
    if model is None or feature_engineer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        df = customer_to_dataframe(customer)
        X, _, _ = prepare_features(df, feature_engineer, fit=False)
        
        churn_prob = float(model.predict_proba(X)[0])
        will_churn = bool(model.predict(X)[0])
        risk_level = get_risk_level(churn_prob)
        
        prediction = Prediction(
            customer_id=customer.customer_id,
            churn_probability=churn_prob,
            prediction=will_churn,
            model_version=model.model_version or "unknown"
        )
        db.add(prediction)
        db.commit()
        
        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=churn_prob,
            will_churn=will_churn,
            risk_level=risk_level,
            model_version=model.model_version,
            timestamp=datetime.utcnow()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/explain", response_model=ExplainedPredictionResponse, tags=["Predictions"])
async def predict_with_explanation(customer: CustomerInput, db: Session = Depends(get_db)):
    if model is None or feature_engineer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    if explainer is None:
        raise HTTPException(status_code=503, detail="Explainer not initialized.")
    
    try:
        df = customer_to_dataframe(customer)
        X, _, _ = prepare_features(df, feature_engineer, fit=False)
        
        churn_prob = float(model.predict_proba(X)[0])
        will_churn = bool(model.predict(X)[0])
        risk_level = get_risk_level(churn_prob)
        
        explanation = explainer.explain_prediction(X, feature_engineer.feature_names)
        
        top_factors = [
            {"feature": k, "impact": v}
            for k, v in list(explanation["top_features"].items())[:5]
        ]
        
        prediction = Prediction(
            customer_id=customer.customer_id,
            churn_probability=churn_prob,
            prediction=will_churn,
            model_version=model.model_version or "unknown"
        )
        db.add(prediction)
        db.commit()
        
        return ExplainedPredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=churn_prob,
            will_churn=will_churn,
            risk_level=risk_level,
            explanation=explanation,
            top_risk_factors=top_factors,
            model_version=model.model_version,
            timestamp=datetime.utcnow()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction with explanation failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict(request: BatchPredictionRequest, db: Session = Depends(get_db)):
    if model is None or feature_engineer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        predictions = []
        
        for customer in request.customers:
            df = customer_to_dataframe(customer)
            X, _, _ = prepare_features(df, feature_engineer, fit=False)
            
            churn_prob = float(model.predict_proba(X)[0])
            will_churn = bool(model.predict(X)[0])
            risk_level = get_risk_level(churn_prob)
            
            prediction = Prediction(
                customer_id=customer.customer_id,
                churn_probability=churn_prob,
                prediction=will_churn,
                model_version=model.model_version or "unknown"
            )
            db.add(prediction)
            
            predictions.append(PredictionResponse(
                customer_id=customer.customer_id,
                churn_probability=churn_prob,
                will_churn=will_churn,
                risk_level=risk_level,
                model_version=model.model_version,
                timestamp=datetime.utcnow()
            ))
        
        db.commit()
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            timestamp=datetime.utcnow()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/model/retrain", tags=["Model Management"])
async def trigger_retrain(background_tasks: BackgroundTasks):
    def retrain_task():
        pipeline = RetrainingPipeline()
        pipeline.run_pipeline(generate_new_data=True, n_samples=10000)
    
    background_tasks.add_task(retrain_task)
    
    return {
        "message": "Model retraining started in background",
        "status": "processing"
    }


@app.get("/model/metrics", response_model=List[ModelMetricsResponse], tags=["Model Management"])
async def get_model_metrics(limit: int = 10, db: Session = Depends(get_db)):
    try:
        metrics = db.query(ModelMetrics).order_by(ModelMetrics.trained_at.desc()).limit(limit).all()
        
        return [
            ModelMetricsResponse(
                model_version=m.model_version,
                accuracy=m.accuracy,
                precision=m.precision,
                recall=m.recall,
                f1_score=m.f1_score,
                roc_auc=m.roc_auc,
                trained_at=m.trained_at
            )
            for m in metrics
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


@app.get("/predictions/history/{customer_id}", tags=["Predictions"])
async def get_prediction_history(customer_id: str, limit: int = 10, db: Session = Depends(get_db)):
    try:
        predictions = db.query(Prediction).filter(
            Prediction.customer_id == customer_id
        ).order_by(Prediction.created_at.desc()).limit(limit).all()
        
        return [
            {
                "churn_probability": p.churn_probability,
                "prediction": p.prediction,
                "model_version": p.model_version,
                "created_at": p.created_at
            }
            for p in predictions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve prediction history: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
