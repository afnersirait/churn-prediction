import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
import mlflow
from sqlalchemy.orm import Session

from data_generator import generate_synthetic_data, split_data
from feature_engineering import AutomatedFeatureEngineer, prepare_features
from model import ChurnPredictor, train_model
from database import SessionLocal, ModelMetrics
from config import settings


class RetrainingPipeline:
    def __init__(self, retrain_threshold: float = None):
        self.retrain_threshold = retrain_threshold or settings.retrain_threshold
        self.current_model = None
        self.feature_engineer = None
        
    def load_current_model(self):
        try:
            self.current_model = ChurnPredictor.load()
            self.feature_engineer = AutomatedFeatureEngineer.load()
            print("Loaded existing model and feature engineer")
            return True
        except Exception as e:
            print(f"No existing model found: {e}")
            return False
    
    def evaluate_model_performance(self, test_data: pd.DataFrame) -> Dict:
        if self.current_model is None or self.feature_engineer is None:
            raise ValueError("Model not loaded. Call load_current_model() first.")
        
        X_test, y_test, _ = prepare_features(test_data, self.feature_engineer, fit=False)
        metrics = self.current_model.evaluate(X_test, y_test)
        
        return metrics
    
    def should_retrain(self, current_metrics: Dict) -> bool:
        accuracy = current_metrics.get("accuracy", 0)
        f1_score = current_metrics.get("f1_score", 0)
        
        performance_score = (accuracy + f1_score) / 2
        
        should_retrain = performance_score < self.retrain_threshold
        
        print(f"Current performance score: {performance_score:.4f}")
        print(f"Retrain threshold: {self.retrain_threshold}")
        print(f"Should retrain: {should_retrain}")
        
        return should_retrain
    
    def retrain_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Tuple[ChurnPredictor, Dict]:
        print("Starting model retraining...")
        
        feature_engineer = AutomatedFeatureEngineer()
        X_train, y_train, feature_engineer = prepare_features(train_data, feature_engineer, fit=True)
        
        predictor, metrics = train_model(train_data, val_data, feature_engineer, log_mlflow=True)
        
        predictor.save()
        feature_engineer.save()
        
        print("Model retrained and saved successfully")
        
        return predictor, metrics
    
    def save_metrics_to_db(self, metrics: Dict, model_version: str):
        db = SessionLocal()
        try:
            model_metrics = ModelMetrics(
                model_version=model_version,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                roc_auc=metrics["roc_auc"],
                trained_at=datetime.utcnow()
            )
            db.add(model_metrics)
            db.commit()
            print(f"Metrics saved to database for model version {model_version}")
        except Exception as e:
            print(f"Error saving metrics to database: {e}")
            db.rollback()
        finally:
            db.close()
    
    def run_pipeline(self, generate_new_data: bool = True, n_samples: int = 10000):
        print("=" * 50)
        print("Starting Automated Retraining Pipeline")
        print("=" * 50)
        
        if generate_new_data:
            print("\nGenerating new synthetic data...")
            df = generate_synthetic_data(n_samples=n_samples)
            train_data, val_data, test_data = split_data(df)
        else:
            print("\nLoading existing data...")
            train_data = pd.read_csv("data/train.csv")
            val_data = pd.read_csv("data/val.csv")
            test_data = pd.read_csv("data/test.csv")
        
        model_exists = self.load_current_model()
        
        if model_exists:
            print("\nEvaluating current model performance...")
            current_metrics = self.evaluate_model_performance(test_data)
            
            print("\nCurrent Model Metrics:")
            for metric, value in current_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            if self.should_retrain(current_metrics):
                print("\nPerformance below threshold. Retraining model...")
                new_model, new_metrics = self.retrain_model(train_data, val_data)
                
                print("\nNew Model Metrics:")
                for metric, value in new_metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
                self.save_metrics_to_db(new_metrics, new_model.model_version or "latest")
            else:
                print("\nModel performance is satisfactory. No retraining needed.")
        else:
            print("\nNo existing model found. Training initial model...")
            feature_engineer = AutomatedFeatureEngineer()
            X_train, y_train, feature_engineer = prepare_features(train_data, feature_engineer, fit=True)
            
            new_model, new_metrics = train_model(train_data, val_data, feature_engineer, log_mlflow=True)
            
            print("\nInitial Model Metrics:")
            for metric, value in new_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            new_model.save()
            feature_engineer.save()
            
            self.save_metrics_to_db(new_metrics, new_model.model_version or "initial")
        
        print("\n" + "=" * 50)
        print("Retraining Pipeline Completed")
        print("=" * 50)


if __name__ == "__main__":
    pipeline = RetrainingPipeline()
    pipeline.run_pipeline(generate_new_data=True, n_samples=10000)
