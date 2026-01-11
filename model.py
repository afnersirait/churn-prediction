import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import joblib
import os
from datetime import datetime
from config import settings


class ChurnPredictor:
    def __init__(self, params: Dict = None):
        self.params = params or {
            "objective": "binary:logistic",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "min_child_weight": 1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "eval_metric": "logloss"
        }
        self.model = None
        self.model_version = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              log_mlflow: bool = True) -> Dict:
        
        if log_mlflow:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment("churn_prediction")
            
        with mlflow.start_run() as run:
            self.model = xgb.XGBClassifier(**self.params)
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred = self.model.predict(X_val)
            y_pred_proba = self.model.predict_proba(X_val)[:, 1]
            
            metrics = {
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred),
                "recall": recall_score(y_val, y_pred),
                "f1_score": f1_score(y_val, y_pred),
                "roc_auc": roc_auc_score(y_val, y_pred_proba)
            }
            
            if log_mlflow:
                mlflow.log_params(self.params)
                mlflow.log_metrics(metrics)
                
                mlflow.xgboost.log_model(
                    self.model,
                    "model",
                    registered_model_name=settings.model_registry_name
                )
                
                feature_importance = pd.DataFrame({
                    "feature": [f"feature_{i}" for i in range(X_train.shape[1])],
                    "importance": self.model.feature_importances_
                }).sort_values("importance", ascending=False)
                
                feature_importance.to_csv("feature_importance.csv", index=False)
                mlflow.log_artifact("feature_importance.csv")
                os.remove("feature_importance.csv")
                
                self.model_version = run.info.run_id
            
            return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1_score": f1_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_pred_proba)
        }
        
        return metrics
    
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        return importance_df
    
    def save(self, path: str = None):
        if path is None:
            path = settings.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved to {path}")
    
    @staticmethod
    def load(path: str = None):
        if path is None:
            path = settings.model_path
        return joblib.load(path)


def train_model(train_data: pd.DataFrame, val_data: pd.DataFrame, 
                feature_engineer, log_mlflow: bool = True) -> Tuple[ChurnPredictor, Dict]:
    from feature_engineering import prepare_features
    
    X_train, y_train, _ = prepare_features(train_data, feature_engineer, fit=False)
    X_val, y_val, _ = prepare_features(val_data, feature_engineer, fit=False)
    
    predictor = ChurnPredictor()
    metrics = predictor.train(X_train, y_train, X_val, y_val, log_mlflow=log_mlflow)
    
    print(f"Model trained with metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return predictor, metrics
