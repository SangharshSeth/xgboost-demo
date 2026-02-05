"""
XGBoost Model Training for Fraud Detection

Trains an XGBoost classifier on prepared features with:
- CPU-based training (GPU optional)
- Class imbalance handling
- Comprehensive metrics capture
- Model and artifact saving
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# Features to use for training (excluding identifiers and target)
FEATURE_COLUMNS = [
    # Amount features
    "amount",
    "amount_zscore",
    "is_high_amount",
    "is_low_amount",
    "amount_change",
    "amount_vs_avg_7d",
    "amount_vs_max_30d",
    
    # Temporal features
    "hour_of_day",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "is_night",
    "is_salary_week",
    
    # Lag features
    "time_since_last_txn",
    "is_city_change",
    "is_same_merchant",
    "txn_sequence",
    
    # Velocity features
    "txn_count_1h",
    "txn_count_6h",
    "txn_count_24h",
    "txn_count_7d",
    "amount_sum_1h",
    "amount_sum_6h",
    "amount_sum_24h",
    "amount_sum_7d",
    "unique_merchants_7d",
    "unique_cities_7d",
    
    # Balance features
    "balance_before",
    "balance_after",
    "balance_change_pct",
    "balance_impact_ratio",
    "is_low_balance",
    "is_large_withdrawal",
    
    # Merchant features
    "merchant_txn_count",
    "merchant_avg_amount",
    "is_first_merchant_txn",
    "category_risk_score",
    
    # Customer features
    "customer_total_txns",
    
    # Encoded categoricals
    "transaction_type_encoded",
    "status_encoded",
    "city_tier_encoded",
    "is_credit_encoded",
]

TARGET_COLUMN = "is_fraud"


def load_features(data_dir: str = "data") -> pd.DataFrame:
    """Load prepared features from parquet."""
    features_path = os.path.join(data_dir, "features.parquet")
    df = pd.read_parquet(features_path)
    return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
    """Split data into train, validation, and test sets."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_adjusted, random_state=42, stratify=y_trainval
    )
    
    print(f"Dataset splits:")
    print(f"  Train: {len(X_train)} samples ({y_train.sum()} fraud)")
    print(f"  Validation: {len(X_val)} samples ({y_val.sum()} fraud)")
    print(f"  Test: {len(X_test)} samples ({y_test.sum()} fraud)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, X_val, y_train, y_val, use_gpu: bool = False) -> xgb.XGBClassifier:
    """Train XGBoost classifier."""
    # Calculate class weight for imbalanced data
    neg_count = len(y_train) - y_train.sum()
    pos_count = y_train.sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    print(f"\nClass imbalance ratio: {scale_pos_weight:.1f}:1")
    
    # Model parameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr", "logloss"],
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,  # Use all CPU cores
        "early_stopping_rounds": 30,
    }
    
    # GPU configuration
    if use_gpu:
        params["device"] = "cuda"
        params["tree_method"] = "hist"
        print("Using GPU for training...")
    else:
        params["device"] = "cpu"
        params["tree_method"] = "hist"
        print("Using CPU for training...")
    
    # Create and train model
    model = xgb.XGBClassifier(**params)
    
    print("\nTraining XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )
    
    return model


def evaluate_model(model: xgb.XGBClassifier, X_test, y_test) -> dict:
    """Evaluate model and return metrics."""
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics (convert numpy types to Python native types)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["true_negatives"] = int(cm[0, 0])
    metrics["false_positives"] = int(cm[0, 1])
    metrics["false_negatives"] = int(cm[1, 0])
    metrics["true_positives"] = int(cm[1, 1])
    
    return metrics, y_proba


def save_feature_importance(model: xgb.XGBClassifier, output_dir: str) -> pd.DataFrame:
    """Save feature importance to CSV."""
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    importance_path = os.path.join(output_dir, "feature_importance.csv")
    importance.to_csv(importance_path, index=False)
    
    return importance


def save_plots(y_test, y_proba, metrics: dict, output_dir: str):
    """Save evaluation plots."""
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, 'g-', linewidth=2, label=f'PR (AUC = {metrics["pr_auc"]:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curves.png"), dpi=150)
    plt.close()
    
    # Confusion Matrix
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Legitimate', 'Fraud']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def run_training(data_dir: str = "data", output_dir: str = "models", use_gpu: bool = False):
    """Run the complete training pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("XGBoost Fraud Detection Model Training")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading features...")
    df = load_features(data_dir)
    print(f"Loaded {len(df)} records with {len(FEATURE_COLUMNS)} features")
    
    # Prepare data
    print("\n[2/5] Preparing train/val/test splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df)
    
    # Train model
    print("\n[3/5] Training model...")
    model = train_model(X_train, X_val, y_train, y_val, use_gpu=use_gpu)
    
    # Evaluate
    print("\n[4/5] Evaluating model...")
    metrics, y_proba = evaluate_model(model, X_test, y_test)
    
    # Save artifacts
    print("\n[5/5] Saving artifacts...")
    
    # Save model
    model_path = os.path.join(output_dir, "model.json")
    model.save_model(model_path)
    
    # Save feature importance
    importance_df = save_feature_importance(model, output_dir)
    
    # Save plots
    save_plots(y_test, y_proba, metrics, output_dir)
    
    # Save metrics
    metrics["training_date"] = datetime.now().isoformat()
    metrics["num_features"] = len(FEATURE_COLUMNS)
    metrics["train_samples"] = len(X_train)
    metrics["test_samples"] = len(X_test)
    metrics["feature_columns"] = FEATURE_COLUMNS
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR AUC:    {metrics['pr_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  True Positives:  {metrics['true_positives']}")
    
    print(f"\nTop 10 Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    print(f"\nArtifacts saved to: {output_dir}/")
    print(f"  - model.json")
    print(f"  - metrics.json")
    print(f"  - feature_importance.csv")
    print(f"  - curves.png")
    print(f"  - confusion_matrix.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train XGBoost fraud detection model")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory with features.parquet")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model and artifacts")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    
    args = parser.parse_args()
    
    run_training(data_dir=args.data_dir, output_dir=args.output_dir, use_gpu=args.gpu)
