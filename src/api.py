"""
FastAPI Scoring Service for Fraud Detection

REST API endpoints for:
- Single transaction scoring
- Batch transaction scoring
- Health check
- Model information
"""

import os
import json
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================================
# SCHEMAS
# ============================================================================

class TransactionRequest(BaseModel):
    """Single transaction for scoring."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    amount: float = Field(..., gt=0, description="Transaction amount in INR")
    transaction_type: str = Field(..., description="Type: UPI, DEBIT_CARD, CREDIT_CARD, etc.")
    merchant_name: str = Field(..., description="Merchant name")
    merchant_category: str = Field(..., description="Merchant category")
    city: str = Field(..., description="Transaction city")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    balance_before: float = Field(..., ge=0, description="Balance before transaction")
    device_id: str = Field(..., description="Device identifier")
    
    # Optional context features (for better predictions)
    hour_of_day: Optional[int] = Field(None, ge=0, le=23)
    time_since_last_txn: Optional[float] = Field(None, ge=0)
    txn_count_24h: Optional[int] = Field(None, ge=0)
    amount_sum_24h: Optional[float] = Field(None, ge=0)
    is_new_merchant: Optional[bool] = None
    is_new_city: Optional[bool] = None


class TransactionResponse(BaseModel):
    """Scoring response for a transaction."""
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    prediction: str = Field(..., description="FRAUD or LEGITIMATE")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, CRITICAL")
    confidence: float = Field(..., ge=0, le=1)
    top_risk_factors: list[dict] = Field(default_factory=list)


class BatchTransactionRequest(BaseModel):
    """Batch of transactions for scoring."""
    transactions: list[TransactionRequest]


class BatchTransactionResponse(BaseModel):
    """Batch scoring response."""
    results: list[TransactionResponse]
    total: int
    fraud_count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_file: str
    num_features: int
    feature_names: list[str]
    training_metrics: dict
    training_date: str


# ============================================================================
# FEATURE ENGINEERING FOR SCORING
# ============================================================================

# Feature columns expected by the model
FEATURE_COLUMNS = [
    "amount", "amount_zscore", "is_high_amount", "is_low_amount", "amount_change",
    "amount_vs_avg_7d", "amount_vs_max_30d", "hour_of_day", "day_of_week", "day_of_month",
    "is_weekend", "is_night", "is_salary_week", "time_since_last_txn", "is_city_change",
    "is_same_merchant", "txn_sequence", "txn_count_1h", "txn_count_6h", "txn_count_24h",
    "txn_count_7d", "amount_sum_1h", "amount_sum_6h", "amount_sum_24h", "amount_sum_7d",
    "unique_merchants_7d", "unique_cities_7d", "balance_before", "balance_after",
    "balance_change_pct", "balance_impact_ratio", "is_low_balance", "is_large_withdrawal",
    "merchant_txn_count", "merchant_avg_amount", "is_first_merchant_txn", "category_risk_score",
    "customer_total_txns", "transaction_type_encoded", "status_encoded", "city_tier_encoded",
    "is_credit_encoded"
]

# Transaction type encoding
TXN_TYPE_MAP = {
    "UPI": 1, "DEBIT_CARD": 2, "CREDIT_CARD": 3, "NEFT": 4,
    "IMPS": 5, "ATM": 6, "SALARY": 7, "PENSION": 8, "EMI": 9,
    "REFUND": 10, "PARENT_TRANSFER": 11
}

# Category risk scores
CATEGORY_RISK = {
    "grocery": 0.1, "food_delivery": 0.15, "ecommerce": 0.3,
    "fuel": 0.15, "utilities": 0.05, "travel": 0.25,
    "entertainment": 0.2, "healthcare": 0.1, "education": 0.05,
    "rent": 0.05, "emi": 0.05, "investment": 0.3, "p2p_transfer": 0.35,
}

# Tier 1 cities
TIER1_CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Pune", "Ahmedabad"]
TIER2_CITIES = ["Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Visakhapatnam", 
                "Patna", "Vadodara", "Coimbatore", "Ludhiana", "Agra", "Nashik", "Ranchi", "Guwahati"]


def prepare_features(txn: TransactionRequest) -> dict:
    """Convert a transaction request to model features."""
    # Basic features
    hour = txn.hour_of_day if txn.hour_of_day is not None else txn.timestamp.hour
    day_of_week = txn.timestamp.weekday() + 1  # 1=Monday, 7=Sunday
    day_of_month = txn.timestamp.day
    
    # Balance calculations
    balance_after = txn.balance_before - txn.amount
    balance_change_pct = -txn.amount / txn.balance_before if txn.balance_before > 0 else 0
    balance_impact_ratio = txn.amount / txn.balance_before if txn.balance_before > 0 else 1
    
    # Encodings
    txn_type_encoded = TXN_TYPE_MAP.get(txn.transaction_type, 0)
    category_risk = CATEGORY_RISK.get(txn.merchant_category, 0.2)
    
    if txn.city in TIER1_CITIES:
        city_tier = 1
    elif txn.city in TIER2_CITIES:
        city_tier = 2
    else:
        city_tier = 3
    
    # Build feature dict with defaults for missing context
    features = {
        "amount": txn.amount,
        "amount_zscore": 0,  # Would need historical data
        "is_high_amount": 0,
        "is_low_amount": 0,
        "amount_change": 0,
        "amount_vs_avg_7d": 1.0,
        "amount_vs_max_30d": 0.5,
        "hour_of_day": hour,
        "day_of_week": day_of_week,
        "day_of_month": day_of_month,
        "is_weekend": 1 if day_of_week >= 6 else 0,
        "is_night": 1 if 0 <= hour < 6 else 0,
        "is_salary_week": 1 if day_of_month <= 7 else 0,
        "time_since_last_txn": txn.time_since_last_txn if txn.time_since_last_txn else 3600,
        "is_city_change": 1 if txn.is_new_city else 0,
        "is_same_merchant": 0,
        "txn_sequence": 100,  # Default
        "txn_count_1h": 1,
        "txn_count_6h": 3,
        "txn_count_24h": txn.txn_count_24h if txn.txn_count_24h else 5,
        "txn_count_7d": 30,
        "amount_sum_1h": txn.amount,
        "amount_sum_6h": txn.amount * 2,
        "amount_sum_24h": txn.amount_sum_24h if txn.amount_sum_24h else txn.amount * 3,
        "amount_sum_7d": txn.amount * 15,
        "unique_merchants_7d": 10,
        "unique_cities_7d": 1,
        "balance_before": txn.balance_before,
        "balance_after": balance_after,
        "balance_change_pct": balance_change_pct,
        "balance_impact_ratio": balance_impact_ratio,
        "is_low_balance": 1 if balance_after < 10000 else 0,
        "is_large_withdrawal": 1 if balance_impact_ratio > 0.3 else 0,
        "merchant_txn_count": 1000,  # Default
        "merchant_avg_amount": txn.amount,
        "is_first_merchant_txn": 1 if txn.is_new_merchant else 0,
        "category_risk_score": category_risk,
        "customer_total_txns": 300,  # Default
        "transaction_type_encoded": txn_type_encoded,
        "status_encoded": 1,  # SUCCESS
        "city_tier_encoded": city_tier,
        "is_credit_encoded": 0,  # Debit transaction
    }
    
    return features


def get_risk_level(probability: float) -> str:
    """Convert probability to risk level."""
    if probability < 0.1:
        return "LOW"
    elif probability < 0.3:
        return "MEDIUM"
    elif probability < 0.7:
        return "HIGH"
    else:
        return "CRITICAL"


def get_top_risk_factors(features: dict, model: xgb.XGBClassifier, n: int = 3) -> list[dict]:
    """Get top contributing features to the prediction."""
    # Use feature importance as proxy for risk factors
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    
    # Filter to features with non-default values
    risk_factors = []
    for feat, importance in sorted(importances.items(), key=lambda x: -x[1]):
        if len(risk_factors) >= n:
            break
        if importance > 0.01:  # Only significant features
            # Convert numpy types to Python native types for JSON serialization
            value = features.get(feat, 0)
            if hasattr(value, 'item'):  # numpy scalar
                value = value.item()
            risk_factors.append({
                "feature": feat,
                "value": float(value) if isinstance(value, (int, float)) else value,
                "importance": float(importance)
            })
    
    return risk_factors


# ============================================================================
# APPLICATION
# ============================================================================

# Global model reference
model: Optional[xgb.XGBClassifier] = None
metrics: Optional[dict] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, metrics
    
    model_path = os.environ.get("MODEL_PATH", "models/model.json")
    metrics_path = os.environ.get("METRICS_PATH", "models/metrics.json")
    
    # Load model
    if os.path.exists(model_path):
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print(f"WARNING: Model not found at {model_path}")
    
    # Load metrics
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"Metrics loaded from {metrics_path}")
    
    yield
    
    # Cleanup
    model = None
    metrics = None


app = FastAPI(
    title="Fraud Detection API",
    description="XGBoost-based fraud detection scoring service for Indian bank transactions",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_file=os.environ.get("MODEL_PATH", "models/model.json"),
        num_features=len(FEATURE_COLUMNS),
        feature_names=FEATURE_COLUMNS,
        training_metrics={
            "roc_auc": metrics.get("roc_auc", 0) if metrics else 0,
            "precision": metrics.get("precision", 0) if metrics else 0,
            "recall": metrics.get("recall", 0) if metrics else 0,
            "f1": metrics.get("f1", 0) if metrics else 0,
        },
        training_date=metrics.get("training_date", "unknown") if metrics else "unknown"
    )


@app.post("/score", response_model=TransactionResponse)
async def score_transaction(txn: TransactionRequest):
    """Score a single transaction for fraud probability."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Prepare features
    features = prepare_features(txn)
    
    # Create DataFrame for prediction
    X = pd.DataFrame([features])[FEATURE_COLUMNS]
    
    # Predict
    proba = model.predict_proba(X)[0, 1]
    prediction = "FRAUD" if proba >= 0.5 else "LEGITIMATE"
    
    return TransactionResponse(
        transaction_id=txn.transaction_id,
        fraud_probability=round(float(proba), 4),
        prediction=prediction,
        risk_level=get_risk_level(proba),
        confidence=round(abs(proba - 0.5) * 2, 4),  # Distance from decision boundary
        top_risk_factors=get_top_risk_factors(features, model)
    )


@app.post("/score/batch", response_model=BatchTransactionResponse)
async def score_batch(request: BatchTransactionRequest):
    """Score multiple transactions in batch."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    results = []
    fraud_count = 0
    
    for txn in request.transactions:
        # Prepare features
        features = prepare_features(txn)
        X = pd.DataFrame([features])[FEATURE_COLUMNS]
        
        # Predict
        proba = model.predict_proba(X)[0, 1]
        prediction = "FRAUD" if proba >= 0.5 else "LEGITIMATE"
        
        if prediction == "FRAUD":
            fraud_count += 1
        
        results.append(TransactionResponse(
            transaction_id=txn.transaction_id,
            fraud_probability=round(float(proba), 4),
            prediction=prediction,
            risk_level=get_risk_level(proba),
            confidence=round(abs(proba - 0.5) * 2, 4),
            top_risk_factors=get_top_risk_factors(features, model)
        ))
    
    processing_time = (time.time() - start_time) * 1000
    
    return BatchTransactionResponse(
        results=results,
        total=len(results),
        fraud_count=fraud_count,
        processing_time_ms=round(processing_time, 2)
    )


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
