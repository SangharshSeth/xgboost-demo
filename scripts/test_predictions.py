"""
Test Script: Validate Model Predictions Against Feature Data

Picks sample transactions from features.parquet, runs them through
the trained model, and compares predictions vs actual labels.
"""

import os
import json
import pandas as pd
import xgboost as xgb


def load_model(model_dir: str = "models"):
    """Load the trained XGBoost model."""
    model_path = os.path.join(model_dir, "model.json")
    metadata_path = os.path.join(model_dir, "metrics.json")
    
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    return model, metadata


def test_single_transaction(features_df: pd.DataFrame, model: xgb.XGBClassifier, 
                            feature_columns: list, idx: int = 0):
    """Test a single transaction and show prediction vs actual."""
    row = features_df.iloc[idx]
    
    # Get actual label
    actual = row['is_fraud']
    txn_id = row.get('transaction_id', f'row_{idx}')
    
    # Prepare features
    X = row[feature_columns].values.reshape(1, -1)
    
    # Predict
    proba = model.predict_proba(X)[0, 1]
    prediction = "FRAUD" if proba >= 0.5 else "LEGITIMATE"
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Transaction: {txn_id}")
    print(f"{'='*60}")
    print(f"\n📊 PREDICTION vs ACTUAL:")
    print(f"   Model Fraud Probability: {proba:.4f} ({proba*100:.2f}%)")
    print(f"   Model Prediction:        {prediction}")
    print(f"   Actual Label:            {'FRAUD' if actual else 'LEGITIMATE'}")
    print(f"   Match:                   {'✅ CORRECT' if (proba >= 0.5) == actual else '❌ WRONG'}")
    
    # Show key features
    print(f"\n📋 KEY FEATURES:")
    print(f"   Amount:           ₹{row.get('amount', 0):,.2f}")
    print(f"   Amount Z-Score:   {row.get('amount_zscore', 0):.2f}")
    print(f"   Txn Count 24h:    {row.get('txn_count_24h', 0)}")
    print(f"   Is City Change:   {row.get('is_city_change', False)}")
    print(f"   Is Night Txn:     {row.get('is_night_txn', False)}")
    print(f"   Hour of Day:      {row.get('hour_of_day', 0)}")
    
    return {
        'transaction_id': txn_id,
        'probability': float(proba),
        'prediction': prediction,
        'actual': bool(actual),
        'correct': (proba >= 0.5) == actual
    }


def test_random_samples(n_samples: int = 5, include_fraud: bool = True):
    """Test random samples including both fraud and non-fraud."""
    # Load model
    print("Loading model...")
    model, metadata = load_model()
    feature_columns = metadata['feature_columns']
    
    # Load features
    print("Loading features...")
    features_df = pd.read_parquet("data/features.parquet")
    print(f"Total transactions: {len(features_df)}")
    print(f"Fraud transactions: {features_df['is_fraud'].sum()}")
    
    results = []
    
    # Test some fraud cases
    if include_fraud:
        fraud_df = features_df[features_df['is_fraud'] == True]
        if len(fraud_df) > 0:
            print(f"\n{'#'*60}")
            print("TESTING FRAUD TRANSACTIONS")
            print(f"{'#'*60}")
            for i in range(min(n_samples // 2, len(fraud_df))):
                result = test_single_transaction(fraud_df, model, feature_columns, i)
                results.append(result)
    
    # Test some legitimate cases
    legit_df = features_df[features_df['is_fraud'] == False]
    print(f"\n{'#'*60}")
    print("TESTING LEGITIMATE TRANSACTIONS")
    print(f"{'#'*60}")
    for i in range(min(n_samples // 2, len(legit_df))):
        result = test_single_transaction(legit_df, model, feature_columns, i)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    correct = sum(1 for r in results if r['correct'])
    print(f"Tested: {len(results)} transactions")
    print(f"Correct: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    
    return results


if __name__ == "__main__":
    test_random_samples(n_samples=10, include_fraud=True)
