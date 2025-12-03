"""
Feature engineering functions for creating target variable and derived features.
"""
import pandas as pd
import numpy as np
from .config import HOME_ADVANTAGE, PROB_THRESHOLD


def create_target_feature(df, home_advantage=HOME_ADVANTAGE, prob_threshold=PROB_THRESHOLD):
    """
    Create target feature 'is_cinderella' based on underdog wins with low probability.
    
    Args:
        df: DataFrame with match data
        home_advantage: Elo points added to home team
        prob_threshold: Maximum underdog win probability threshold
        
    Returns:
        DataFrame with target feature added
    """
    print("🎯 Creating target feature:")
    
    df['Elo_diff'] = (df['HomeElo'] + home_advantage - df['AwayElo'])
    df['Prob_HomeWin'] = 1 / (1 + 10 ** (-df['Elo_diff'] / 400))
    df['Prob_AwayWin'] = 1 - df['Prob_HomeWin']
    
    # Determine favorite and underdog
    df['favorite'] = np.where(df['Prob_HomeWin'] >= 0.5, 'Home', 'Away')
    df['underdog'] = np.where(df['favorite'] == 'Home', 'Away', 'Home')
    
    # Get underdog win probability
    df['Prob_UnderdogWin'] = np.where(
        df['underdog'] == 'Home', 
        df['Prob_HomeWin'], 
        df['Prob_AwayWin']
    )
    
    # Determine if result is a cinderella win
    underdog_wins = (
        ((df['underdog'] == 'Home') & (df['FTResult'] == 'H')) |
        ((df['underdog'] == 'Away') & (df['FTResult'] == 'A'))
    )
    is_low_prob = (df['Prob_UnderdogWin'] < prob_threshold)
    df['is_cinderella'] = (underdog_wins & is_low_prob).astype(int)
    
    # Reset Elo_diff (remove home advantage for modeling)
    df['Elo_diff'] = df['Elo_diff'] - home_advantage
    
    print(f"   ✓ Target created. Cinderella rate: {df['is_cinderella'].mean() * 100:.2f}%")
    return df


def count_matches_in_window(g, days=10):
    """Helper function to count matches in a time window."""
    g = g.sort_values('MatchDate').copy()
    counts = []
    for idx in g.index:
        current_date = g.at[idx, 'MatchDate']
        window_start = current_date - pd.Timedelta(days=days)
        matches_before = g[(g['MatchDate'] >= window_start) & (g['MatchDate'] < current_date)]
        counts.append(len(matches_before))
    return pd.Series(counts, index=g.index)


def feature_engineering(df):
    """
    Create advanced engineered features.
    
    Args:
        df: DataFrame with cleaned data
        
    Returns:
        DataFrame with engineered features
    """
    print("\n⚙️  Starting feature engineering...")
    
    df = df.sort_values('MatchDate').reset_index(drop=True)
    df['_match_id'] = np.arange(len(df))
    
    # ===================================================================
    # === A. Home/Away Features ===
    # ===================================================================

    print(" A. Creating Home/Away Features...")

    # ===================================================================
    # === 1. Rest Days and Congestion ===
    # ===================================================================

    # Number of days since last match for home and away teams
    # and congestion in last 10 days (number of matches played in last 10 days)
    print("   1.1. Calculating Rest Days...")
    team_dates_home = df[['_match_id', 'MatchDate', 'HomeTeam']].copy()
    team_dates_home.columns = ['_match_id', 'MatchDate', 'team']
    team_dates_home['is_home'] = 1

    team_dates_away = df[['_match_id', 'MatchDate', 'AwayTeam']].copy()
    team_dates_away.columns = ['_match_id', 'MatchDate', 'team']
    team_dates_away['is_home'] = 0

    team_dates = pd.concat([team_dates_home, team_dates_away], ignore_index=True)
    team_dates = team_dates.sort_values(['team', 'MatchDate']).reset_index(drop=True)
    team_dates['last_match_date'] = team_dates.groupby('team')['MatchDate'].shift(1)
    team_dates['rest_days'] = (team_dates['MatchDate'] - team_dates['last_match_date']).dt.days

    home_rest = team_dates[team_dates['is_home'] == 1].set_index('_match_id')['rest_days']
    away_rest = team_dates[team_dates['is_home'] == 0].set_index('_match_id')['rest_days']

    df['RestDaysHome'] = df['_match_id'].map(home_rest)
    df['RestDaysAway'] = df['_match_id'].map(away_rest)
    print("      ✓ Rest Days calculated.")

    print("   1.2. Calculating Congestion (matches in last 10 days)...")
    team_dates_home = df[['_match_id', 'MatchDate', 'HomeTeam']].copy()
    team_dates_home.columns = ['_match_id', 'MatchDate', 'team']
    team_dates_home['is_home'] = 1
    
    team_dates_away = df[['_match_id', 'MatchDate', 'AwayTeam']].copy()
    team_dates_away.columns = ['_match_id', 'MatchDate', 'team']
    team_dates_away['is_home'] = 0
    
    team_dates = pd.concat([team_dates_home, team_dates_away], ignore_index=True)
    team_dates = team_dates.sort_values(['team', 'MatchDate']).reset_index(drop=True)
    
    team_dates['congestion_10d'] = team_dates.groupby('team', group_keys=False).apply(
        lambda g: count_matches_in_window(g, 10)
    )
    home_congestion = team_dates[team_dates['is_home'] == 1].set_index('_match_id')['congestion_10d']
    away_congestion = team_dates[team_dates['is_home'] == 0].set_index('_match_id')['congestion_10d']
    
    df['CongestionHome_10d'] = df['_match_id'].map(home_congestion)
    df['CongestionAway_10d'] = df['_match_id'].map(away_congestion)
    print("      ✓ Congestion calculated.")


    # ===================================================================
    # === 2. Elo changes ===
    # ===================================================================

    # Calcuate Elo changes over past 1 month and 2 months for home and away teams
    print("   2. Calculating Elo Changes (1 month & 2 months)...")
    home_elo_data = df[['MatchDate', 'HomeTeam', 'HomeElo', '_match_id']].copy()
    home_elo_data.columns = ['MatchDate', 'team', 'elo', '_match_id']
    home_elo_data['is_home'] = 1

    away_elo_data = df[['MatchDate', 'AwayTeam', 'AwayElo', '_match_id']].copy()
    away_elo_data.columns = ['MatchDate', 'team', 'elo', '_match_id']
    away_elo_data['is_home'] = 0

    team_elo = pd.concat([home_elo_data, away_elo_data], ignore_index=True)
    team_elo = team_elo.sort_values(['team', 'MatchDate']).reset_index(drop=True)

    team_elo['MatchDate_ref_1m'] = team_elo['MatchDate'] - pd.Timedelta(days=30)
    team_elo['MatchDate_ref_2m'] = team_elo['MatchDate'] - pd.Timedelta(days=60)

    right_df_sorted = team_elo[['team', 'MatchDate', 'elo']].rename(columns={'elo': 'past_elo'}).sort_values('MatchDate')

    merged_1m = pd.merge_asof(
        team_elo.sort_values('MatchDate_ref_1m'), # Sắp xếp theo cột bên trái
        right_df_sorted,
        left_on='MatchDate_ref_1m',
        right_on='MatchDate',
        by='team',
        direction='backward' # Tìm giá trị gần nhất TRƯỚC ngày tham chiếu
    )
    merged_1m['EloChange1'] = merged_1m['elo'] - merged_1m['past_elo']

    merged_2m = pd.merge_asof(
        team_elo.sort_values('MatchDate_ref_2m'),
        right_df_sorted,
        left_on='MatchDate_ref_2m',
        right_on='MatchDate',
        by='team',
        direction='backward'
    )
    merged_2m['EloChange2'] = merged_2m['elo'] - merged_2m['past_elo']

    team_elo_indexed = team_elo.set_index(['_match_id', 'team'])

    # Set index cho các kết quả merge để join
    merged_1m = merged_1m.set_index(['_match_id', 'team'])
    merged_2m = merged_2m.set_index(['_match_id', 'team'])

    # Join các kết quả
    team_elo_indexed['EloChange1'] = merged_1m['EloChange1']
    team_elo_indexed['EloChange2'] = merged_2m['EloChange2']

    # Reset index 
    final_elo_changes = team_elo_indexed.reset_index()

    # Tách home/away và merge về df gốc
    home_changes = final_elo_changes[final_elo_changes['is_home'] == 1][['_match_id', 'EloChange1', 'EloChange2']].rename(
        columns={'EloChange1': 'EloChange1Home', 'EloChange2': 'EloChange2Home'})
    away_changes = final_elo_changes[final_elo_changes['is_home'] == 0][['_match_id', 'EloChange1', 'EloChange2']].rename(
        columns={'EloChange1': 'EloChange1Away', 'EloChange2': 'EloChange2Away'})

    df = df.merge(home_changes, on='_match_id', how='left')
    df = df.merge(away_changes, on='_match_id', how='left')

    print("      ✓ Elo Changes calculated.")

    # ===================================================================
    # === 3. Rolling Stats ===
    # ===================================================================

    print("   3. Calculating Rolling Stats...")
    # Prepare data for rolling stats calculation
    home_base = df[['_match_id', 'MatchDate', 'HomeTeam', 'FTHome', 'FTAway']].rename(
        columns={'HomeTeam': 'team', 'FTHome': 'GoalsFor', 'FTAway': 'GoalsAgainst'}
    )
    home_base['is_home'] = 1

    away_base = df[['_match_id', 'MatchDate', 'AwayTeam', 'FTAway', 'FTHome']].rename(
        columns={'AwayTeam': 'team', 'FTAway': 'GoalsFor', 'FTHome': 'GoalsAgainst'}
    )
    away_base['is_home'] = 0

    team_stats = pd.concat([home_base, away_base])
    team_stats['points'] = np.where(
        team_stats['GoalsFor'] > team_stats['GoalsAgainst'], 3,
        np.where(team_stats['GoalsFor'] == team_stats['GoalsAgainst'], 1, 0)
    )

    advanced_stats_map = {
        'HomeShots': 'shots', 'AwayShots': 'shots',
        'HomeTarget': 'target', 'AwayTarget': 'target',
        'HomeCorners': 'corners', 'AwayCorners': 'corners',
        'HomeFouls': 'fouls', 'AwayFouls': 'fouls',
        'HomeYellow': 'yellow', 'AwayYellow': 'yellow',
        'HomeRed': 'red', 'AwayRed': 'red'
    }

    existing_adv_cols = [col for col in advanced_stats_map.keys() if col in df.columns]
    if existing_adv_cols:
        advanced_df = df[['_match_id'] + existing_adv_cols].copy()

        for stat_name in set(advanced_stats_map.values()):
            home_col = next((k for k, v in advanced_stats_map.items() if v == stat_name and 'Home' in k), None)
            away_col = next((k for k, v in advanced_stats_map.items() if v == stat_name and 'Away' in k), None)
            if home_col and away_col and home_col in advanced_df.columns and away_col in advanced_df.columns:
                advanced_df[f'{stat_name}_total'] = advanced_df[home_col] + advanced_df[away_col]

        team_stats = team_stats.merge(advanced_df, on='_match_id', how='left')

        for stat_name in set(advanced_stats_map.values()):
            home_col = next((k for k, v in advanced_stats_map.items() if v == stat_name and 'Home' in k), None)
            away_col = next((k for k, v in advanced_stats_map.items() if v == stat_name and 'Away' in k), None)
            if home_col and away_col and home_col in team_stats.columns and away_col in team_stats.columns:
                team_stats[f'{stat_name}_for'] = np.where(team_stats['is_home'] == 1, team_stats[home_col], team_stats[away_col])
                team_stats[f'{stat_name}_conceded'] = team_stats[f'{stat_name}_total'] - team_stats[f'{stat_name}_for']

    team_stats = team_stats.sort_values(['team', 'MatchDate']).reset_index(drop=True)

    # === 3.1. Basic Rolling Stats ===
    print("      3.1. Basic Rolling Stats (Forms, Goals For/Against)...")
    basic_cols_to_shift = ['GoalsFor', 'GoalsAgainst', 'points']
    for col in basic_cols_to_shift:
        team_stats[f'{col}_shifted'] = team_stats.groupby('team')[col].shift(1)

    for w in [3, 5]:
        team_stats[f'GF{w}'] = team_stats.groupby('team')['GoalsFor_shifted'].rolling(w, min_periods=0).sum().reset_index(level=0, drop=True)
        team_stats[f'GA{w}'] = team_stats.groupby('team')['GoalsAgainst_shifted'].rolling(w, min_periods=0).sum().reset_index(level=0, drop=True)
        team_stats[f'Form{w}'] = team_stats.groupby('team')['points_shifted'].rolling(w, min_periods=0).sum().reset_index(level=0, drop=True)
    print("          ✓ Basic Rolling Stats calculated.")

    # === 3.2. Advanced Rolling Stats ===
    print("      3.2. Advanced Rolling Stats (Shots, Target, Corners, Fouls, Yellow, Red)...")
    advanced_cols_to_shift = [c for c in team_stats.columns if '_for' in c or '_conceded' in c]
    for col in advanced_cols_to_shift:
        team_stats[f'{col}_shifted'] = team_stats.groupby('team')[col].shift(1)

    agg_dict = {}
    stat_types = ['shots', 'target', 'corners', 'fouls', 'yellow', 'red']
    for stat in stat_types:
        for_col_shifted = f'{stat}_for_shifted'
        conceded_col_shifted = f'{stat}_conceded_shifted'
        
        if for_col_shifted in team_stats.columns:
            for w in [3, 5]:
                team_stats[f'{stat.capitalize()}{w}'] = team_stats.groupby('team')[for_col_shifted].rolling(w, min_periods=0).sum().reset_index(level=0, drop=True)

        if conceded_col_shifted in team_stats.columns:
            for w in [3, 5]:
                team_stats[f'{stat.capitalize()}Conceded{w}'] = team_stats.groupby('team')[conceded_col_shifted].rolling(w, min_periods=0).sum().reset_index(level=0, drop=True)

    print("          ✓ Advanced Rolling Stats calculated.")

    # Merge rolling stats back to main dataframe
    home_final = team_stats[team_stats['is_home'] == 1].add_prefix('Home')
    away_final = team_stats[team_stats['is_home'] == 0].add_prefix('Away')

    df = df.merge(home_final, left_on=['_match_id'], right_on=['Home_match_id'], how='left')
    df = df.merge(away_final, left_on=['_match_id'], right_on=['Away_match_id'], how='left')

    # ===================================================================
    # === 4. Head-to-Head Features ===
    # ===================================================================

    print("   4. Calculating Head-to-Head Features...")
    df['team_pair'] = df.apply(lambda row: tuple(sorted([row['HomeTeam'], row['AwayTeam']])), axis=1)
    df = df.sort_values(['team_pair', 'MatchDate'])

    # calculate points and wins for home and away teams
    df['HomePoints'] = np.where(df['FTHome'] > df['FTAway'], 3, np.where(df['FTHome'] == df['FTAway'], 1, 0))
    df['AwayPoints'] = np.where(df['FTAway'] > df['FTHome'], 3, np.where(df['FTAway'] == df['FTHome'], 1, 0))
    df['HomeWin'] = np.where(df['FTHome'] > df['FTAway'], 1, 0)
    df['AwayWin'] = np.where(df['FTAway'] > df['FTHome'], 1, 0)

    grouped = df.groupby('team_pair')
    df['prev_home_pts'] = grouped['HomePoints'].shift(1)
    df['prev_away_pts'] = grouped['AwayPoints'].shift(1)
    df['prev_home_wins'] = grouped['HomeWin'].shift(1)
    df['prev_away_wins'] = grouped['AwayWin'].shift(1)
    df['prev_home'] = grouped['HomeTeam'].shift(1)

    # group_shifted = df.groupby('team_pair')
    for w in [3, 5]:
        df[f'h2h_Home_Points{w}'] = df.groupby('team_pair')['prev_home_pts'].rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)
        df[f'h2h_Away_Points{w}'] = df.groupby('team_pair')['prev_away_pts'].rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)
        df[f'h2h_Home_Wins{w}'] = df.groupby('team_pair')['prev_home_wins'].rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)
        df[f'h2h_Away_Wins{w}'] = df.groupby('team_pair')['prev_away_wins'].rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)

    for w in [3, 5]:
        is_home_team_A = df['HomeTeam'] == df['team_pair'].str[0]
        
        df[f'H2H_Home_Points{w}'] = np.where(is_home_team_A, df[f'h2h_Home_Points{w}'], df[f'h2h_Away_Points{w}'])
        df[f'H2H_Away_Points{w}'] = np.where(is_home_team_A, df[f'h2h_Away_Points{w}'], df[f'h2h_Home_Points{w}'])
        df[f'H2H_Home_Wins{w}'] = np.where(is_home_team_A, df[f'h2h_Home_Wins{w}'], df[f'h2h_Away_Wins{w}'])
        df[f'H2H_Away_Wins{w}'] = np.where(is_home_team_A, df[f'h2h_Away_Wins{w}'], df[f'h2h_Home_Wins{w}'])
   
    # fill NaN with 0 (created by shift)
    h2h_cols = [c for c in df.columns if c.startswith('H2H_')]
    df[h2h_cols] = df[h2h_cols].fillna(0)

    df = df.sort_values('MatchDate').reset_index(drop=True)
    print("      ✓ Head-to-Head Features calculated.")

    # ===================================================================
    # === 5. Momentum Features ===
    # ===================================================================

    # The difference between short-term and long-term form (3-match form minus 5-match form)
    print("   5. Calculating Momentum Features...")
    df['MomentumHome'] = (df['Form3Home'] / 3) - (df['Form5Home'] / 5)
    df['MomentumAway'] = (df['Form3Away'] / 3) - (df['Form5Away'] / 5)
    print("      ✓ Momentum Features calculated.")


    # ===================================================================
    # === B. Underdog/Favorite Features ===
    # ===================================================================

    print("B. Creating underdog/favorite and difference features...")
    # create boolean masks for underdog and favorite status
    is_home_underdog = (df['underdog'] == 'Home')
    is_away_underdog = (df['underdog'] == 'Away')
    is_home_favorite = (df['favorite'] == 'Home')
    is_away_favorite = (df['favorite'] == 'Away')

    # ===================================================================
    # === 1. Rest Days ===
    # ===================================================================

    print("   1. Creating Rest Days features...")
    df['RestDays_underdog'] = np.where(is_home_underdog, df['RestDaysHome'],
                                np.where(is_away_underdog, df['RestDaysAway'], np.nan))
    df['RestDays_fav'] = np.where(is_home_favorite, df['RestDaysHome'],
                            np.where(is_away_favorite, df['RestDaysAway'], np.nan))
    df['RestDays_diff'] = df['RestDays_fav'] - df['RestDays_underdog']
    print("      ✓ Rest Days features created.")

    # ===================================================================
    # === 2. Congestion ===
    # ===================================================================

    print("   2. Creating Congestion features...")
    df['Congestion_underdog_10d'] = np.where(is_home_underdog, df['CongestionHome_10d'],
                                    np.where(is_away_underdog, df['CongestionAway_10d'], np.nan))
    df['Congestion_fav_10d'] = np.where(is_home_favorite, df['CongestionHome_10d'],
                                np.where(is_away_favorite, df['CongestionAway_10d'], np.nan))
    df['Congestion_diff_10d'] = df['Congestion_fav_10d'] - df['Congestion_underdog_10d']
    print("      ✓ Congestion features created.")

    # ===================================================================
    # === 3.1. Form 3 ===
    # ===================================================================
    print("   3. Creating Form features...")
    df['Form3_underdog'] = np.where(is_home_underdog, df['Form3Home'],
                            np.where(is_away_underdog, df['Form3Away'], np.nan))
    df['Form3_fav'] = np.where(is_home_favorite, df['Form3Home'],
                        np.where(is_away_favorite, df['Form3Away'], np.nan))
    df['Form3_diff'] = df['Form3_fav'] - df['Form3_underdog']
    
    # ===================================================================
    # === 3.2. Form 5 ===
    # ===================================================================

    df['Form5_underdog'] = np.where(is_home_underdog, df['Form5Home'],
                            np.where(is_away_underdog, df['Form5Away'], np.nan))
    df['Form5_fav'] = np.where(is_home_favorite, df['Form5Home'],
                        np.where(is_away_favorite, df['Form5Away'], np.nan))
    df['Form5_diff'] = df['Form5_fav'] - df['Form5_underdog']
    print("      ✓ Form features created.")

    # ===================================================================   
    # === 4. Elo Change ===
    # ===================================================================   

    print("   4. Creating Elo Change features...")
    df['EloChange1_underdog'] = np.where(is_home_underdog, df['EloChange1Home'],
                                np.where(is_away_underdog, df['EloChange1Away'], np.nan))
    df['EloChange1_fav'] = np.where(is_home_favorite, df['EloChange1Home'],
                            np.where(is_away_favorite, df['EloChange1Away'], np.nan))
    df['EloChange1_diff'] = df['EloChange1_fav'] - df['EloChange1_underdog']

    df['EloChange2_underdog'] = np.where(is_home_underdog, df['EloChange2Home'],
                                np.where(is_away_underdog, df['EloChange2Away'], np.nan))
    df['EloChange2_fav'] = np.where(is_home_favorite, df['EloChange2Home'],
                            np.where(is_away_favorite, df['EloChange2Away'], np.nan))
    df['EloChange2_diff'] = df['EloChange2_fav'] - df['EloChange2_underdog']
    print("      ✓ Elo Change features created.")

    # ===================================================================
    # === 5. H2H ===
    # ===================================================================

    print("   5. Creating Head-to-Head features...")
    df['H2H_underdog_Wins3'] = np.where(is_home_underdog, df['H2H_Home_Wins3'],
                                np.where(is_away_underdog, df['H2H_Away_Wins3'], np.nan))
    df['H2H_fav_Wins3'] = np.where(is_home_favorite, df['H2H_Home_Wins3'],
                            np.where(is_away_favorite, df['H2H_Away_Wins3'], np.nan))
    df['H2H_Wins3_diff'] = df['H2H_fav_Wins3'] - df['H2H_underdog_Wins3']

    df['H2H_underdog_Wins5'] = np.where(is_home_underdog, df['H2H_Home_Wins5'],
                                np.where(is_away_underdog, df['H2H_Away_Wins5'], np.nan))
    df['H2H_fav_Wins5'] = np.where(is_home_favorite, df['H2H_Home_Wins5'],
                            np.where(is_away_favorite, df['H2H_Away_Wins5'], np.nan))
    df['H2H_Wins5_diff'] = df['H2H_fav_Wins5'] - df['H2H_underdog_Wins5']
    
    df['H2H_underdog_Points3'] = np.where(is_home_underdog, df['H2H_Home_Points3'],
                                np.where(is_away_underdog, df['H2H_Away_Points3'], np.nan))
    df['H2H_fav_Points3'] = np.where(is_home_favorite, df['H2H_Home_Points3'],
                            np.where(is_away_favorite, df['H2H_Away_Points3'], np.nan))
    df['H2H_Points3_diff'] = df['H2H_fav_Points3'] - df['H2H_underdog_Points3']
    
    df['H2H_underdog_Points5'] = np.where(is_home_underdog, df['H2H_Home_Points5'],
                                np.where(is_away_underdog, df['H2H_Away_Points5'], np.nan))
    df['H2H_fav_Points5'] = np.where(is_home_favorite, df['H2H_Home_Points5'],
                            np.where(is_away_favorite, df['H2H_Away_Points5'], np.nan))
    df['H2H_Points5_diff'] = df['H2H_fav_Points5'] - df['H2H_underdog_Points5']
    print("      ✓ Head-to-Head features created.")

    # ===================================================================
    # === 6.1 Rolling Stats (for stats) ===
    # ===================================================================

    print("   6. Creating Rolling Stats features...")
    stat_bases = ['GF3', 'GA3', 'Shots3', 'Target3', 'Corners3', 'Fouls3', 'Yellow3', 'Red3',
                  'GF5', 'GA5', 'Shots5', 'Target5', 'Corners5', 'Fouls5', 'Yellow5', 'Red5']
    for stat_base in stat_bases:
        home_col = f'Home{stat_base}'
        away_col = f'Away{stat_base}'

        if home_col in df.columns and away_col in df.columns:
            # Underdog
            df[f'{stat_base}_underdog'] = np.where(
                is_home_underdog, df[home_col],
                np.where(is_away_underdog, df[away_col], np.nan)
            )

            # Favorite
            df[f'{stat_base}_fav'] = np.where(
                is_home_favorite, df[home_col],
                np.where(is_away_favorite, df[away_col], np.nan)
            )

            # Difference (fav - underdog)
            df[f'{stat_base}_diff'] = (
                df[f'{stat_base}_fav'] - df[f'{stat_base}_underdog']
            )

    # ===================================================================
    # === 6.2 Rolling Stats (for conceded stats) ===
    # ===================================================================

    conceded_bases = ['GA3', 'ShotsConceded3', 'TargetConceded3', 'CornersConceded3',
                      'FoulsConceded3', 'YellowConceded3', 'RedConceded3',
                      'GA5', 'ShotsConceded5', 'TargetConceded5', 'CornersConceded5',
                      'FoulsConceded5', 'YellowConceded5', 'RedConceded5']
    for conceded_base in conceded_bases:
        home_col = f'Home{conceded_base}'
        away_col = f'Away{conceded_base}'
    
        if home_col in df.columns and away_col in df.columns:
            # Underdog
            df[f'{conceded_base}_underdog'] = np.where(
                is_home_underdog, df[home_col],
                np.where(is_away_underdog, df[away_col], np.nan)
            )

            # Favorite
            df[f'{conceded_base}_fav'] = np.where(
                is_home_favorite, df[home_col],
                np.where(is_away_favorite, df[away_col], np.nan)
            )

            # Difference (fav - underdog)
            df[f'{conceded_base}_diff'] = (
                df[f'{conceded_base}_fav'] - df[f'{conceded_base}_underdog']
            )
    print("      ✓ Rolling Stats features created.")

    # ===================================================================
    # === 7. Momentum Features ===
    # ===================================================================

    print("   7. Creating Momentum features...")
    df['Momentum_underdog'] = np.where(is_home_underdog, df['MomentumHome'],
                                np.where(is_away_underdog, df['MomentumAway'], np.nan))
    df['Momentum_fav'] = np.where(is_home_favorite, df['MomentumHome'],
                            np.where(is_away_favorite, df['MomentumAway'], np.nan))
    df['Momentum_diff'] = df['Momentum_fav'] - df['Momentum_underdog']
    print("      ✓ Momentum features created.")

    # ===================================================================
    # === CLEANUP: Drop all temporary columns ===
    # ===================================================================
    
    print("   🧹 Cleaning up temporary columns...")
    temp_cols = ['_match_id', 'team_pair', 'prev_home_pts', 'prev_away_pts', 
                 'prev_home_wins', 'prev_away_wins', 'prev_home']
    df = df.drop(columns=[c for c in temp_cols if c in df.columns], errors='ignore')
    
    print(f"\n✅ Feature engineering completed! Final shape: {df.shape}")
    return df


def extract_features(df, feature_set='set2'):
    """
    Extract feature column names based on feature set selection.
    
    Args:
        df: DataFrame
        feature_set: Name of feature set ('raw', 'set2', etc.)
        
    Returns:
        List of feature column names
    """
    from .config import RAW_FEATURES, PREPARED_FEATURES_SET2
    
    if feature_set == 'raw':
        return RAW_FEATURES
    elif feature_set == 'set2':
        # Filter only existing columns
        return [col for col in PREPARED_FEATURES_SET2 if col in df.columns]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")