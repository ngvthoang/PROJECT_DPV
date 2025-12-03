"""
Data loading, filtering, and cleaning functions.
"""
import pandas as pd
import numpy as np
from .config import TOP_LEAGUES


def load_data(path):
    """
    Load raw match data from CSV.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    print(f"📂 Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"   Initial shape: {df.shape}")
    return df


def filter_data(df):
    """
    Filter and select relevant columns for analysis.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Filtered DataFrame with relevant columns
    """
    print(f"🔍 Filtering data:")
    
    # Filter for top leagues only
    df = df[df['Division'].isin(TOP_LEAGUES)].copy()
    
    # Define relevant columns
    relevant_columns = [
        'Division', 'MatchDate', 'MatchTime', 'HomeTeam', 'AwayTeam', 
        'HomeElo', 'AwayElo', 'Form3Home', 'Form3Away', 'Form5Home', 'Form5Away',
        'FTHome', 'FTAway', 'FTResult',
        'HomeShots', 'AwayShots', 'HomeTarget', 'AwayTarget',
        'HomeCorners', 'AwayCorners', 'HomeFouls', 'AwayFouls',
        'HomeYellow', 'AwayYellow', 'HomeRed', 'AwayRed'
    ]
    
    # Filter columns that exist
    existing_cols = [col for col in relevant_columns if col in df.columns]
    df_filtered = df[existing_cols].copy()
    
    print(f"   After filtering: {df_filtered.shape}")
    return df_filtered


def clean_data(df):
    """
    Clean and preprocess the data.
    
    Args:
        df: Filtered DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("🧹 Cleaning data:")
    
    # Convert MatchDate to datetime
    df['MatchDate'] = pd.to_datetime(df['MatchDate'], dayfirst=True, errors='coerce')
    print("   ✓ Converted MatchDate to datetime")
    
    # Sort by date
    df = df.sort_values(by='MatchDate').reset_index(drop=True)
    print("   ✓ Sorted data by MatchDate")
    
    # Drop MatchTime (high missing rate)
    if 'MatchTime' in df.columns:
        df = df.drop(columns=['MatchTime'])
        print("   ✓ Dropped MatchTime column")
    
    # Remove invalid rows
    initial_len = len(df)
    condition_valid_score = (df['FTHome'] >= 0) & (df['FTAway'] >= 0)
    condition_valid_elo = (df['HomeElo'] > 0) & (df['AwayElo'] > 0)
    df = df[condition_valid_score & condition_valid_elo].copy()
    print(f"   ✓ Removed {initial_len - len(df)} rows with invalid scores/Elo")
    
    # Remove duplicates
    before_dedup = len(df)
    df = df.drop_duplicates()
    if len(df) < before_dedup:
        print(f"   ✓ Removed {before_dedup - len(df)} duplicate rows")
    
    print(f"   Final shape: {df.shape}")
    return df
