# Cinderella Story Project

**Predicting Football Upsets with Advanced Feature Engineering and Machine Learning**

A production-ready machine learning pipeline that identifies "Cinderella" wins in football matches—rare upsets where a significant underdog (with <30% win probability) defeats a heavily favored opponent.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Project Structure](#-project-structure)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Data](#-data)
- [Methodology](#-methodology)
- [Notebooks](#-notebooks)
- [Summary: The Notebook Journey](#summary-the-notebook-journey)

---

## 🎯 Overview

This project demonstrates the transformative power of **data preparation and feature engineering** in machine learning. By comparing a baseline model trained on raw data against a sophisticated pipeline with 30+ engineered features, we quantify the impact of proper data science methodology on predictive performance.

### The Challenge

Predicting football upsets is inherently difficult because:
- **Rare Event**: Cinderella wins occur in only ~5.8% of matches (severe class imbalance)
- **Low Signal**: Raw match statistics show minimal differences between upsets and normal outcomes
- **Complex Patterns**: Upsets depend on subtle interactions between team strength, form, fatigue, and tactical factors

### The Solution

A two-stage modeling approach:
1. **RAW Model**: Baseline using only pre-existing features (Elo ratings, recent form)
2. **PREPARED Model**: Advanced pipeline with engineered features capturing momentum, fatigue, rolling statistics, and head-to-head history

---

## 🎪 Problem Statement

### Definition: What is a "Cinderella" Win?

A match qualifies as a Cinderella win when:
1. The **underdog** (team with <50% win probability) wins
2. The underdog's pre-match win probability was **<30%**
3. Win probability calculated using: `P(Home Win) = 1 / (1 + 10^(-(HomeElo + 100 - AwayElo) / 400))`

### Business Value

- **Sports Analytics**: Identify conditions that enable upsets for tactical planning
- **Betting Models**: Detect mispriced matches where underdogs have hidden advantages
- **Fan Engagement**: Predict "upset potential" to highlight exciting fixtures
- **Team Performance**: Understand vulnerabilities in strong teams that lead to shock defeats

---

## 📁 Project Structure

```
.
PROJECT_DPV/
├── data/
│   └── Matches.csv
│   └── Matches_Prepared.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── pipelines.py
│   └── modeling.py
├── 01_EDA_Raw.ipynb
├── 02_Data_Preparation_and_Analysis.ipynb
├── 03_EDA_w_Prepared_Data.ipynb
├── main.py
├── requirements.txt
└── README.md
```

- **data/**: Contains raw and processed data
- **src/**: Source code for data loading, feature engineering, model training, and evaluation
- **requirements.txt**: Python package dependencies
- **README.md**: Project documentation

---

## 🔑 Key Features

- **Comprehensive Data Processing**: Handles raw data ingestion, cleaning, and transformation
- **Advanced Feature Engineering**: 100+ engineered features capturing complex match dynamics
- **Robust Modeling**: Comparison of baseline and advanced models to quantify impact of feature engineering
- **Detailed Results Analysis**: In-depth evaluation of model performance and feature importance
- **Modular Code Structure**: Well-organized codebase with clear separation of concerns

---

## 🚀 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ngvthoang/PROJECT_DPV.git
    cd PROJECT_DPV
    ```
2. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 📊 Data

### Source & Scope
- **Dataset**: Historical football match data (2000-2024) from [Github](https://github.com/xgabora/Club-Football-Match-Data-2000-2025?fbclid=IwY2xjawOciaRleHRuA2FlbQIxMABicmlkETFxREJZRFc0SnRwYWwySkZsc3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHu3dcntxgOQgmhRAQqQ12f1ooeTmpFBOzlEXmoKzg4HTL1FYzxOZpHpFHpku_aem_6uJbnoQViegJB2Mha_nMBw)
- **Coverage**: Top 5 European leagues (EPL, La Liga, Bundesliga, Serie A, Ligue 1)
- **Size**: 43,668 matches after filtering and cleaning
- **Class Distribution**: 94.2% Normal (41,132) | 5.8% Cinderella (2,536)

### Key Features

#### Pre-Match Features (Predictive)
| Category | Features | Description |
|----------|----------|-------------|
| **Team Strength** | HomeElo, AwayElo | Elo ratings (1200-2000 range) |
| **Recent Form** | Form3/5 Home/Away | Points in last 3/5 matches |
| **Match Context** | Division, MatchDate, Teams | League and participants |

#### Match Outcome Features (Target Creation)
| Feature | Description |
|---------|-------------|
| FTHome, FTAway | Final score |
| FTResult | Match result (H/D/A) |

#### In-Match Statistics (Feature Engineering)
Performance metrics used to create rolling averages: Shots, Shots on Target, Corners, Fouls, Yellow/Red Cards

### Target Variable Definition

A match is classified as **Cinderella** (`is_cinderella = 1`) when:

1. **Underdog wins**: The team with lower pre-match probability wins
2. **Low probability threshold**: Underdog's win probability < 30%

**Probability Calculation:**
```python
Elo_diff = (HomeElo + 100) - AwayElo  # +100 for home advantage
P(HomeWin) = 1 / (1 + 10^(-Elo_diff / 400))
```
---

## 📚 Methodology

1. **Data Ingestion**: Load raw data from CSV files
2. **Data Cleaning**: Handle missing values, incorrect data types, and duplicates
3. **Feature Engineering**: Create new features capturing team form, momentum, fatigue, and head-to-head statistics
4. **Data Splitting**: Split the data into training, validation, and test sets
5. **Model Training**: Train baseline and advanced models using the training set
6. **Model Evaluation**: Evaluate model performance on the validation and test sets
7. **Results Analysis**: Analyze model predictions, feature importance, and error cases

---

## 📓 Notebooks

### 1. `01_EDA_Raw.ipynb` - Exploratory Analysis on Raw Data

**Purpose**: Baseline analysis to understand raw data limitations and establish the need for feature engineering.

**Key Sections**:
- **Data Quality Assessment**: Missing values (75% in MatchTime, ~10% in match stats), class imbalance (5.8% Cinderella rate)
- **Baseline Patterns**: Elo ratings show signal, but raw form/shots features lack predictive power
- **Hypothesis Testing**: 
  - Elo distributions differ between Normal vs Cinderella matches
  - Raw form (Form3/5) and match stats (shots, corners) show minimal univariate differences
- **Visualization**: Distribution plots, box plots, and statistical tests (Mann-Whitney U)
- **SHAP Analysis on RAW Model**: 
  - HomeElo and AwayElo dominate importance (~70-80% of signal)
  - Form features show minimal SHAP impact due to lack of context
  - Identifies "competitive underdog zone" (HomeElo 1600-1700, AwayElo 1900+)

**Key Insight**: Raw data reveals strength differentials but hides momentum, fatigue, and tactical signals—proving the necessity of advanced feature engineering.

---

### 2. `02_Data_Preparation_and_Analysis.ipynb` - Feature Engineering Pipeline

**Purpose**: Transform raw data into predictive features through systematic engineering and train the production model.

**Major Steps**:
1. **Data Cleaning** (Part 1): Remove invalid scores, duplicates, and handle missing values
2. **Target Creation** (Part 2): Define `is_cinderella` using probability threshold (<30% underdog win probability)
3. **Feature Engineering** (Part 3): Create 30+ features across 5 categories:
   - Rest Days & Congestion (fatigue metrics)
   - Elo Changes (1-month and 2-month trajectories)
   - Rolling Stats (3-match and 5-match windows for goals, shots, corners, fouls, cards)
   - Head-to-Head (H2H) history (last 3 and 5 encounters)
   - Momentum (short-term vs long-term form trends)
4. **Feature Selection** (Part 3.4): Test 6 feature sets, select best performer (Home/Away with manual selection - 24 features)
5. **Model Training** (Part 5): LightGBM with class balancing and threshold optimization
6. **Model Interpretation** (Part 6): SHAP analysis reveals top drivers:
   - `Elo_diff` (0.245), `MomentumAway` (0.187), `EloChange1Home` (0.143)
   - `Corners3_diff` (0.128), `GA3_diff` (0.119) validate tactical dominance hypothesis
7. **Impact Analysis**: Compare RAW vs PREPARED models showing 59% F1-score improvement

**Output**: `Matches_Prepared.csv` with 120 engineered columns ready for production deployment.

---

### 3. `03_EDA_w_Prepared_Data.ipynb` - Validation & Hypothesis Testing

**Purpose**: Visually validate engineered features and test specific hypotheses about upset drivers.

**Hypothesis Tests**:
1. **Momentum Signal** (H1): 
   - Box plot shows Cinderella matches have lower `Momentum_diff` (median -0.067 vs 0.000)
   - Underdogs "in form" significantly increase upset probability
   
2. **Fatigue Signal** (H2): 
   - ECDF plots reveal `RestDays_diff` and `Congestion_diff` distributions are **identical** for both outcomes
   - Fatigue alone is not a standalone predictor (may contribute via interactions)
   
3. **Defensive Fragility** (H3): 
   - Box plot shows `GA5_fav` medians are equal (6.0 vs 6.0)
   - Simple univariate analysis misses the signal—SHAP later reveals interaction effects
   
4. **Tactical Dominance** (H4): 
   - Median shots are identical, but **outlier analysis** reveals "reckless attacker" effect
   - Extremely high shot volumes from underdogs often correlate with losses, not wins
   - Successful Cinderella teams show balanced, efficient attacking (not volume-based)

5. **Elo × Momentum Interaction** (H6):
   - Scatter plot reveals the "upset zone": Large `Elo_diff` + Low `Momentum_diff` (underdog momentum advantage)
   - Proves upsets require **combination of factors**, not isolated signals

**Visualization Techniques**: Box plots, ECDF curves, scatter plots with color-coded outcomes, and annotated medians for clear communication.

**Key Takeaway**: Engineered features transform invisible patterns (momentum, tactical trends) into observable signals. Simple features fail individually but combine powerfully in ML models—validating the feature engineering strategy.

---

## Summary: The Notebook Journey

| Notebook | Focus | Key Output |
|----------|-------|------------|
| **01_EDA_Raw** | Problem identification | Raw data limitations, SHAP baseline |
| **02_Data_Preparation** | Solution implementation | Engineered features, trained model |
| **03_EDA_Prepared** | Solution validation | Hypothesis confirmation, interaction discovery |

**Narrative Arc**: The notebooks tell a complete story—from discovering signal gaps in raw data (01), to engineering solutions (02), to validating those solutions work (03). Each notebook builds on the previous, creating a cohesive data science workflow.

---