# Usage Guide

## Getting Started

### 1. Train Your First Model

```bash
python train.py
```

This generates synthetic data and trains the model. You should see output like:

```
Test Set Performance:
  Accuracy       : 0.9215
  Precision      : 0.8934
  Recall         : 0.8756
  F1_score       : 0.8844
  Roc_auc        : 0.9567
```

### 2. Start the API Server

```bash
uvicorn main:app --reload
```

Visit http://localhost:8000/docs for interactive API documentation.

### 3. Make Your First Prediction

Using curl:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Using Python:
```python
import requests

customer = {
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

response = requests.post("http://localhost:8000/predict", json=customer)
print(response.json())
```

## Advanced Usage

### Get Explainable Predictions

```bash
curl -X POST "http://localhost:8000/predict/explain" \
  -H "Content-Type: application/json" \
  -d '{...customer_data...}'
```

This returns SHAP values explaining which features influenced the prediction.

### Batch Predictions

```python
import requests

customers = {
    "customers": [
        {...customer1...},
        {...customer2...},
        {...customer3...}
    ]
}

response = requests.post("http://localhost:8000/predict/batch", json=customers)
print(response.json())
```

### Monitor Model Performance

```bash
curl http://localhost:8000/model/metrics
```

### Trigger Retraining

```bash
curl -X POST http://localhost:8000/model/retrain
```

## Using with Docker

### Build and Run

```bash
docker-compose up -d
```

This starts:
- PostgreSQL database on port 5432
- MLflow server on port 5000
- FastAPI application on port 8000

### Train Model in Docker

```bash
docker-compose exec api python train.py
```

### View Logs

```bash
docker-compose logs -f api
```

## Database Setup

### Using PostgreSQL

1. Install PostgreSQL
2. Create database:
```sql
CREATE DATABASE churn_db;
CREATE USER churn_user WITH PASSWORD 'churn_password';
GRANT ALL PRIVILEGES ON DATABASE churn_db TO churn_user;
```

3. Update `.env`:
```env
DATABASE_URL=postgresql://churn_user:churn_password@localhost:5432/churn_db
```

### Using SQLite (Development)

Update `.env`:
```env
DATABASE_URL=sqlite:///./churn.db
```

## MLflow Setup

### Local MLflow Server

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Visit http://localhost:5000 to view experiments.

### Configure in Application

Update `.env`:
```env
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Customization

### Adjust Model Parameters

Edit `model.py`:
```python
self.params = {
    "max_depth": 8,  # Increase for more complex models
    "learning_rate": 0.05,  # Decrease for better accuracy
    "n_estimators": 300,  # Increase for better performance
    ...
}
```

### Change Retraining Threshold

Update `.env`:
```env
RETRAIN_THRESHOLD=0.90  # Retrain if accuracy drops below 90%
```

### Add Custom Features

Edit `feature_engineering.py`:
```python
def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Add your custom features here
    df["my_custom_feature"] = df["tenure"] * df["monthly_charges"]
    
    return df
```

## Troubleshooting

### Model Not Loading

Ensure you've trained the model first:
```bash
python train.py
```

### Database Connection Error

Check PostgreSQL is running:
```bash
pg_isready -h localhost -p 5432
```

### MLflow Connection Error

Verify MLflow server is running:
```bash
curl http://localhost:5000/health
```

## Production Deployment

### Environment Variables

Set these in production:
```env
DATABASE_URL=postgresql://user:pass@prod-db:5432/churn_db
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MODEL_REGISTRY_NAME=churn_prediction_model_prod
RETRAIN_THRESHOLD=0.88
```

### Security

- Use strong database passwords
- Enable HTTPS for API
- Implement authentication
- Rate limit API endpoints

### Monitoring

- Set up logging aggregation
- Monitor API response times
- Track model performance metrics
- Alert on prediction errors
