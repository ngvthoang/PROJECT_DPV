"""
Model training, evaluation, and utility functions.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, precision_score, recall_score
)
from .config import TRAIN_RATIO, VAL_RATIO, THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP


def time_split(df, target_col, feature_cols):
    """
    Perform time-based train/val/test split.
    
    Args:
        df: DataFrame sorted by date
        target_col: Name of target column
        feature_cols: List of feature column names
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    df = df.sort_values('MatchDate').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def find_best_threshold(pipeline, X_val, y_val):
    """
    Find optimal classification threshold on validation set.
    
    Args:
        pipeline: Fitted sklearn pipeline
        X_val: Validation features
        y_val: Validation targets
        
    Returns:
        Best threshold value
    """
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP)
    f1_scores = [f1_score(y_val, (y_val_proba >= t).astype(int)) for t in thresholds]
    best_thresh = thresholds[np.argmax(f1_scores)]
    return best_thresh


def evaluate_model(pipeline, X_test, y_test, X_val, y_val, model_name='Model'):
    """
    Evaluate fitted pipeline and return metrics.
    
    Args:
        pipeline: Fitted sklearn pipeline
        X_test: Test features
        y_test: Test targets
        X_val: Validation features
        y_val: Validation targets
        model_name: Name for display
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\n📊 Evaluating {model_name}...")
    
    # Find best threshold on validation set
    best_thresh = find_best_threshold(pipeline, X_val, y_val)
    print(f"   Best threshold: {best_thresh:.3f}")
    
    # Predict on test set
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_test_proba >= best_thresh).astype(int)
    
    # Calculate metrics
    results = {
        'model_name': model_name,
        'threshold': best_thresh,
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'pr_auc': average_precision_score(y_test, y_test_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Print results
    print(f"   F1-Score:  {results['f1']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"   PR-AUC:    {results['pr_auc']:.4f}")
    
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    return results
