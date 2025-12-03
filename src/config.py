"""
Configuration file for Cinderella Story Project.
Contains all constants, parameters, and hyperparameters.
"""
import numpy as np

# ============================================================================
# FILE PATHS
# ============================================================================
RAW_DATA_PATH = "data\\Matches.csv"
PROCESSED_DATA_PATH = "data\\Matches_Prepared.csv"

# ============================================================================
# DATA FILTERING
# ============================================================================
TOP_LEAGUES = ['E0', 'SP1', 'D1', 'I1', 'F1']  # Premier League, La Liga, Bundesliga, Serie A, Ligue 1

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================
HOME_ADVANTAGE = 100  # Elo points added to home team
PROB_THRESHOLD = 0.3  # Maximum underdog win probability to classify as "cinderella"
ROLLING_WINDOWS = [3, 5]  # Window sizes for rolling statistics

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# LightGBM Hyperparameters
LGBM_PARAMS = {
    'n_estimators': 600,
    'learning_rate': 0.03,
    'num_leaves': 50,
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    'class_weight': 'balanced',
    'is_unbalance': True,
    'random_state': RANDOM_SEED,
    'objective': 'binary',
    'verbose': -1,
}

# ============================================================================
# TRAIN/VAL/TEST SPLIT RATIOS
# ============================================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================================
# THRESHOLD SEARCH PARAMETERS
# ============================================================================
THRESHOLD_MIN = 0.2
THRESHOLD_MAX = 0.8
THRESHOLD_STEP = 0.02

# ============================================================================
# FEATURE SET DEFINITIONS
# ============================================================================
RAW_FEATURES = ['HomeElo', 'AwayElo', 'Form3Home', 'Form3Away', 'Form5Home', 'Form5Away']

# Feature set 2 (best performing from notebook): Home/Away with manual selection
PREPARED_FEATURES_SET2 = [
    'Elo_diff',
    # Elo changes
    'EloChange1Home', 'EloChange1Away',
    # Rest days
    'RestDaysHome', 'RestDaysAway',
    # Momentum
    'MomentumHome', 'MomentumAway',
    # Congestion
    'CongestionHome_10d', 'CongestionAway_10d',
    # Form 3
    'Form3Home', 'Form3Away',
    # Rolling stats 3
    'HomeGF3', 'AwayGF3', 'HomeGA3', 'AwayGA3',
    'HomeShots3', 'AwayShots3', 'HomeTarget3', 'AwayTarget3',
    'HomeCorners3', 'AwayCorners3', 'HomeFouls3', 'AwayFouls3',
    'HomeYellow3', 'AwayYellow3', 'HomeRed3', 'AwayRed3',
    # H2H 3
    'H2H_Home_Wins3', 'H2H_Away_Wins3'
]
