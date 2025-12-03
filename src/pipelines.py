"""
Scikit-learn pipeline definitions for RAW and PREPARED models.
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier
from .config import LGBM_PARAMS


def create_raw_pipeline(feature_cols):
    """
    Create pipeline for RAW data model.
    
    Args:
        feature_cols: List of feature column names
        
    Returns:
        Sklearn Pipeline object
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', RobustScaler())
            ]), feature_cols)
        ],
        remainder='drop'
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(**LGBM_PARAMS))
    ])
    
    return pipeline


def create_prepared_pipeline(feature_cols):
    """
    Create pipeline for PREPARED data model.
    
    Args:
        feature_cols: List of feature column names
        
    Returns:
        Sklearn Pipeline object
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', RobustScaler())
            ]), feature_cols)
        ],
        remainder='drop'
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(**LGBM_PARAMS))
    ])
    
    return pipeline
