import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple
import joblib
import os


class AutomatedFeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.categorical_features = [
            "contract_type", "payment_method", "internet_service",
            "online_security", "tech_support", "streaming_tv", "streaming_movies"
        ]
        self.numerical_features = ["tenure", "monthly_charges", "total_charges"]
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df["charges_per_month"] = df["total_charges"] / (df["tenure"] + 1)
        
        df["tenure_monthly_ratio"] = df["tenure"] / (df["monthly_charges"] + 1)
        
        df["high_value_customer"] = (
            (df["monthly_charges"] > df["monthly_charges"].quantile(0.75)) &
            (df["tenure"] > 24)
        ).astype(int)
        
        df["at_risk_customer"] = (
            (df["tenure"] < 12) &
            (df["monthly_charges"] > df["monthly_charges"].median())
        ).astype(int)
        
        df["service_count"] = (
            (df["online_security"] == "Yes").astype(int) +
            (df["tech_support"] == "Yes").astype(int) +
            (df["streaming_tv"] == "Yes").astype(int) +
            (df["streaming_movies"] == "Yes").astype(int)
        )
        
        df["has_premium_services"] = (df["service_count"] >= 2).astype(int)
        
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 100],
            labels=["0-12", "12-24", "24-48", "48+"]
        )
        
        df["charges_group"] = pd.cut(
            df["monthly_charges"],
            bins=[0, 35, 70, 100, 200],
            labels=["Low", "Medium", "High", "Very High"]
        )
        
        return df
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        
        for col in self.categorical_features:
            if col in df.columns:
                if fit:
                    self.encoders[col] = LabelEncoder()
                    df[f"{col}_encoded"] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.encoders:
                        df[f"{col}_encoded"] = self.encoders[col].transform(df[col].astype(str))
        
        if "tenure_group" in df.columns:
            if fit:
                self.encoders["tenure_group"] = LabelEncoder()
                df["tenure_group_encoded"] = self.encoders["tenure_group"].fit_transform(df["tenure_group"].astype(str))
            else:
                if "tenure_group" in self.encoders:
                    df["tenure_group_encoded"] = self.encoders["tenure_group"].transform(df["tenure_group"].astype(str))
        
        if "charges_group" in df.columns:
            if fit:
                self.encoders["charges_group"] = LabelEncoder()
                df["charges_group_encoded"] = self.encoders["charges_group"].fit_transform(df["charges_group"].astype(str))
            else:
                if "charges_group" in self.encoders:
                    df["charges_group_encoded"] = self.encoders["charges_group"].transform(df["charges_group"].astype(str))
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        
        numerical_cols = self.numerical_features + [
            "charges_per_month", "tenure_monthly_ratio"
        ]
        
        for col in numerical_cols:
            if col in df.columns:
                if fit:
                    self.scalers[col] = StandardScaler()
                    df[f"{col}_scaled"] = self.scalers[col].fit_transform(df[[col]])
                else:
                    if col in self.scalers:
                        df[f"{col}_scaled"] = self.scalers[col].transform(df[[col]])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.create_interaction_features(df)
        df = self.encode_features(df, fit=True)
        df = self.scale_features(df, fit=True)
        
        self.feature_names = [col for col in df.columns if col not in 
                             ["customer_id", "churned", "tenure_group", "charges_group"] + 
                             self.categorical_features]
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.create_interaction_features(df)
        df = self.encode_features(df, fit=False)
        df = self.scale_features(df, fit=False)
        return df
    
    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feature_names].values
    
    def save(self, path: str = "models/feature_engineer.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
    
    @staticmethod
    def load(path: str = "models/feature_engineer.pkl"):
        return joblib.load(path)


def prepare_features(df: pd.DataFrame, engineer: AutomatedFeatureEngineer = None, fit: bool = True) -> Tuple[np.ndarray, np.ndarray, AutomatedFeatureEngineer]:
    if engineer is None:
        engineer = AutomatedFeatureEngineer()
    
    if fit:
        df_transformed = engineer.fit_transform(df)
    else:
        df_transformed = engineer.transform(df)
    
    X = engineer.get_feature_matrix(df_transformed)
    y = df["churned"].values if "churned" in df.columns else None
    
    return X, y, engineer
