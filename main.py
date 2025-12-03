"""
Main execution script for Cinderella Story Project.
Orchestrates the entire pipeline: data loading, processing, training, and evaluation.
"""
import pandas as pd
from src.config import RAW_DATA_PATH
from src.data_processing import load_data, filter_data, clean_data
from src.feature_engineering import create_target_feature, feature_engineering, extract_features
from src.pipelines import create_raw_pipeline, create_prepared_pipeline
from src.modeling import time_split, evaluate_model


def main():
    """Main execution function."""
    print("="*80)
    print("🎭 CINDERELLA STORY PROJECT - PRODUCTION PIPELINE")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load and Process Data
    # ========================================================================
    print("\n📥 STEP 1: Loading and Processing Data")
    print("-"*80)
    
    df = load_data(RAW_DATA_PATH)
    df = filter_data(df)
    df = clean_data(df)
    df = create_target_feature(df)
    
    # Create copy for RAW model
    df_raw = df.copy()
    
    # ========================================================================
    # STEP 2: Train and Evaluate RAW Model
    # ========================================================================
    print("\n" + "="*80)
    print("🔷 STEP 2: RAW DATA MODEL")
    print("="*80)
    
    raw_features = extract_features(df_raw, feature_set='raw')
    print(f"\n📋 RAW features: {raw_features}")
    
    X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = time_split(
        df_raw, 'is_cinderella', raw_features
    )
    
    print(f"\n📊 Data split:")
    print(f"   Train: {X_train_raw.shape}, Cinderella: {y_train_raw.mean():.2%}")
    print(f"   Val:   {X_val_raw.shape}, Cinderella: {y_val_raw.mean():.2%}")
    print(f"   Test:  {X_test_raw.shape}, Cinderella: {y_test_raw.mean():.2%}")
    
    print("\n🚀 Training RAW model...")
    raw_pipeline = create_raw_pipeline(raw_features)
    raw_pipeline.fit(X_train_raw, y_train_raw)
    print("   ✓ Training complete")
    
    results_raw = evaluate_model(
        raw_pipeline, X_test_raw, y_test_raw, X_val_raw, y_val_raw, 
        model_name='RAW Model'
    )
    
    # ========================================================================
    # STEP 3: Train and Evaluate PREPARED Model
    # ========================================================================
    print("\n" + "="*80)
    print("🔶 STEP 3: PREPARED DATA MODEL")
    print("="*80)
    
    print("\n⚙️  Running feature engineering...")
    df_prepared = feature_engineering(df.copy())
    
    prep_features = extract_features(df_prepared, feature_set='set2')
    print(f"\n📋 PREPARED features ({len(prep_features)} total)")
    
    X_train_prep, y_train_prep, X_val_prep, y_val_prep, X_test_prep, y_test_prep = time_split(
        df_prepared, 'is_cinderella', prep_features
    )
    
    print(f"\n📊 Data split:")
    print(f"   Train: {X_train_prep.shape}, Cinderella: {y_train_prep.mean():.2%}")
    print(f"   Val:   {X_val_prep.shape}, Cinderella: {y_val_prep.mean():.2%}")
    print(f"   Test:  {X_test_prep.shape}, Cinderella: {y_test_prep.mean():.2%}")
    
    print("\n🚀 Training PREPARED model...")
    prep_pipeline = create_prepared_pipeline(prep_features)
    prep_pipeline.fit(X_train_prep, y_train_prep)
    print("   ✓ Training complete")
    
    results_prep = evaluate_model(
        prep_pipeline, X_test_prep, y_test_prep, X_val_prep, y_val_prep,
        model_name='PREPARED Model'
    )
    
    # ========================================================================
    # STEP 4: Compare Results
    # ========================================================================
    print("\n" + "="*80)
    print("📊 STEP 4: MODEL COMPARISON")
    print("="*80)
    
    comparison = pd.DataFrame([
        {
            'Model': 'RAW',
            'F1-Score': results_raw['f1'],
            'Precision': results_raw['precision'],
            'Recall': results_raw['recall'],
            'ROC-AUC': results_raw['roc_auc'],
            'PR-AUC': results_raw['pr_auc']
        },
        {
            'Model': 'PREPARED',
            'F1-Score': results_prep['f1'],
            'Precision': results_prep['precision'],
            'Recall': results_prep['recall'],
            'ROC-AUC': results_prep['roc_auc'],
            'PR-AUC': results_prep['pr_auc']
        }
    ])
    
    print("\n📈 Performance Comparison:")
    print(comparison.to_string(index=False))
    
    # Calculate improvements
    print("\n🎯 Improvements (PREPARED vs RAW):")
    for metric in ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC']:
        raw_val = comparison[comparison['Model'] == 'RAW'][metric].values[0]
        prep_val = comparison[comparison['Model'] == 'PREPARED'][metric].values[0]
        improvement = ((prep_val - raw_val) / raw_val) * 100
        print(f"   {metric:12s}: {improvement:+.2f}%")
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    main()
