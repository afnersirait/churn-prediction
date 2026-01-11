import pandas as pd
from data_generator import generate_synthetic_data, split_data
from feature_engineering import AutomatedFeatureEngineer, prepare_features
from model import train_model
import os


def main():
    print("=" * 60)
    print("Customer Churn Prediction - Model Training")
    print("=" * 60)
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("\n1. Generating synthetic customer data...")
    df = generate_synthetic_data(n_samples=10000)
    print(f"   Generated {len(df)} customer records")
    print(f"   Churn rate: {df['churned'].mean():.2%}")
    
    print("\n2. Splitting data into train/val/test sets...")
    train_data, val_data, test_data = split_data(df)
    print(f"   Train: {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples")
    print(f"   Test: {len(test_data)} samples")
    
    df.to_csv("data/full_data.csv", index=False)
    train_data.to_csv("data/train.csv", index=False)
    val_data.to_csv("data/val.csv", index=False)
    test_data.to_csv("data/test.csv", index=False)
    print("   Data saved to data/ directory")
    
    print("\n3. Engineering features with automated pipeline...")
    feature_engineer = AutomatedFeatureEngineer()
    X_train, y_train, feature_engineer = prepare_features(train_data, feature_engineer, fit=True)
    print(f"   Created {X_train.shape[1]} features")
    print(f"   Feature names: {len(feature_engineer.feature_names)}")
    
    print("\n4. Training XGBoost model with MLflow tracking...")
    predictor, metrics = train_model(train_data, val_data, feature_engineer, log_mlflow=False)
    
    print("\n5. Evaluating on test set...")
    X_test, y_test, _ = prepare_features(test_data, feature_engineer, fit=False)
    test_metrics = predictor.evaluate(X_test, y_test)
    
    print("\n" + "=" * 60)
    print("Test Set Performance:")
    print("=" * 60)
    for metric, value in test_metrics.items():
        print(f"  {metric.capitalize():15s}: {value:.4f}")
    
    print("\n6. Saving model and feature engineer...")
    predictor.save()
    feature_engineer.save()
    print("   Model saved to models/churn_model.pkl")
    print("   Feature engineer saved to models/feature_engineer.pkl")
    
    print("\n7. Feature importance (Top 10):")
    importance_df = predictor.get_feature_importance(feature_engineer.feature_names)
    print(importance_df.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start the API server: python main.py")
    print("  2. Or use uvicorn: uvicorn main:app --reload")
    print("  3. View API docs at: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
