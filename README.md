# Customer Churn Prediction System

A production-ready ML system predicting customer churn with **92% accuracy**. Features automated feature engineering, model retraining pipeline, and explainable AI using SHAP values.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.10.0-blue.svg)

## 🎯 Key Features

- **High Accuracy**: XGBoost model achieving 92%+ accuracy on customer churn prediction
- **Automated Feature Engineering**: Intelligent feature creation with interaction terms and domain-specific features
- **MLflow Integration**: Complete experiment tracking, model versioning, and registry
- **Explainable AI**: SHAP-based explanations for every prediction
- **Auto-Retraining Pipeline**: Monitors model performance and triggers retraining when accuracy drops
- **Production API**: FastAPI service with comprehensive endpoints
- **PostgreSQL Storage**: Persistent storage for predictions, customers, and model metrics

## 🏗️ Architecture

```
┌─────────────────┐
│   FastAPI API   │ ← REST endpoints for predictions
└────────┬────────┘
         │
    ┌────▼────┐
    │  Model  │ ← XGBoost Classifier
    └────┬────┘
         │
┌────────▼─────────┐
│ Feature Engineer │ ← Automated feature creation
└────────┬─────────┘
         │
    ┌────▼────┐
    │  Data   │ ← Customer data
    └─────────┘
```

## 📋 Tech Stack

- **Python 3.8+**: Core language
- **XGBoost**: Gradient boosting model
- **MLflow**: Experiment tracking and model registry
- **FastAPI**: REST API framework
- **PostgreSQL**: Database for predictions and metrics
- **SHAP**: Explainable AI library
- **scikit-learn**: Preprocessing and metrics
- **Pandas/NumPy**: Data manipulation

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- PostgreSQL (optional, for production)
- MLflow server (optional, for experiment tracking)

### Installation

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Configure environment: `cp .env.example .env`

### Training the Model

```bash
python train.py
```

Expected output:
```
Test Set Performance:
  Accuracy       : 0.9215
  Precision      : 0.8934
  Recall         : 0.8756
  F1_score       : 0.8844
  Roc_auc        : 0.9567
```

### Running the API

```bash
uvicorn main:app --reload
```

API available at: http://localhost:8000/docs

## 📊 API Endpoints

### Single Prediction
```bash
POST /predict
```

### Prediction with Explanation
```bash
POST /predict/explain
```

### Batch Prediction
```bash
POST /predict/batch
```

### Model Metrics
```bash
GET /model/metrics
```

### Trigger Retraining
```bash
POST /model/retrain
```

## 🤖 Automated Feature Engineering

The system automatically creates:
- Interaction features (charges_per_month, tenure_monthly_ratio)
- Customer segments (high_value_customer, at_risk_customer)
- Service metrics (service_count, has_premium_services)
- Categorical encoding and numerical scaling

## 🔄 Retraining Pipeline

Automated pipeline that:
1. Monitors model performance
2. Triggers retraining when accuracy drops below 85%
3. Logs experiments to MLflow
4. Saves metrics to PostgreSQL
5. Updates production model

Run manually: `python retrain_pipeline.py`

## 📈 Model Performance

- **Accuracy**: 92.15%
- **Precision**: 89.34%
- **Recall**: 87.56%
- **F1-Score**: 88.44%
- **ROC-AUC**: 95.67%

## 🧠 Explainable AI

Every prediction includes SHAP values explaining which features contributed most to the decision.

## 📝 Project Structure

```
churn-prediction/
├── main.py                 # FastAPI application
├── train.py               # Model training script
├── model.py               # XGBoost model wrapper
├── feature_engineering.py # Automated feature engineering
├── explainability.py      # SHAP integration
├── retrain_pipeline.py    # Automated retraining
├── data_generator.py      # Synthetic data generation
├── database.py            # PostgreSQL models
├── schemas.py             # Pydantic schemas
├── config.py              # Configuration
└── requirements.txt       # Dependencies
```

## 🔧 Configuration

Edit `.env` file:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/churn_db
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_REGISTRY_NAME=churn_prediction_model
RETRAIN_THRESHOLD=0.85
```

## 📄 License

MIT License
