import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple


def generate_synthetic_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    customer_ids = [f"CUST{str(i).zfill(6)}" for i in range(n_samples)]
    
    tenure = np.random.randint(0, 73, n_samples)
    
    monthly_charges = np.random.uniform(20, 120, n_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 50, n_samples)
    total_charges = np.maximum(total_charges, 0)
    
    contract_types = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        n_samples,
        p=[0.55, 0.25, 0.20]
    )
    
    payment_methods = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_samples,
        p=[0.35, 0.20, 0.25, 0.20]
    )
    
    internet_services = np.random.choice(
        ["DSL", "Fiber optic", "No"],
        n_samples,
        p=[0.35, 0.45, 0.20]
    )
    
    online_security = np.where(
        internet_services == "No",
        "No internet service",
        np.random.choice(["Yes", "No"], n_samples, p=[0.35, 0.65])
    )
    
    tech_support = np.where(
        internet_services == "No",
        "No internet service",
        np.random.choice(["Yes", "No"], n_samples, p=[0.35, 0.65])
    )
    
    streaming_tv = np.where(
        internet_services == "No",
        "No internet service",
        np.random.choice(["Yes", "No"], n_samples, p=[0.45, 0.55])
    )
    
    streaming_movies = np.where(
        internet_services == "No",
        "No internet service",
        np.random.choice(["Yes", "No"], n_samples, p=[0.45, 0.55])
    )
    
    churn_prob = np.zeros(n_samples)
    churn_prob += (tenure < 12) * 0.3
    churn_prob += (contract_types == "Month-to-month") * 0.25
    churn_prob += (payment_methods == "Electronic check") * 0.15
    churn_prob += (monthly_charges > 80) * 0.15
    churn_prob += (online_security == "No") * 0.10
    churn_prob += (tech_support == "No") * 0.10
    churn_prob += (internet_services == "Fiber optic") * 0.05
    
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    churned = np.random.binomial(1, churn_prob, n_samples).astype(bool)
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_types,
        "payment_method": payment_methods,
        "internet_service": internet_services,
        "online_security": online_security,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "churned": churned
    })
    
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    
    train_val, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df["churned"])
    
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_size_adjusted, random_state=42, stratify=train_val["churned"])
    
    return train, val, test


if __name__ == "__main__":
    df = generate_synthetic_data(n_samples=10000)
    train, val, test = split_data(df)
    
    import os
    os.makedirs("data", exist_ok=True)
    
    df.to_csv("data/full_data.csv", index=False)
    train.to_csv("data/train.csv", index=False)
    val.to_csv("data/val.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    
    print(f"Generated {len(df)} samples")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Churn rate: {df['churned'].mean():.2%}")
