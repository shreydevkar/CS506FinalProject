from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def train_linear_regression(X, y):
    """
    Train baseline Linear Regression model.
    
    Why Linear Regression?
    - Simple, interpretable baseline: shows linear relationships between features and volatility
    - Fast to train and predict: good for real-time trading systems
    - Provides feature importance via coefficients (e.g., "rolling_vol_5 has coefficient 0.8")
    - Weakness: assumes volatility follows linear patterns (reality is non-linear with vol spikes)
    
    Use case: Good for understanding feature relationships; poor for capturing complex patterns.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_random_forest(X, y):
    """
    Train Random Forest Regressor: ensemble of decision trees.
    
    Why Random Forest?
    - Captures non-linear relationships (volatility clustering, regime changes)
    - Handles feature interactions (e.g., "high vol + high volume" behaves differently than either alone)
    - Robust to outliers (trees split around extremes)
    - Feature importance ranking: identifies which features matter most
    
    Hyperparameters:
    - n_estimators=100: 100 trees; more = better fit but slower; 100 is standard
    - random_state=42: reproducibility (same results every run)
    - min_samples_leaf=1: minimum samples to split a leaf (1 = allow deep trees for complex patterns)
    - max_features='sqrt': consider sqrt(6)≈2.45 features per split (prevents overfitting)
    
    Weakness: Can overfit on small datasets; slower predictions than LR.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=1, max_features='sqrt')
    model.fit(X, y)
    return model


def train_xgboost(X, y):
    """
    Train XGBoost: gradient boosting (sequential weak learners → strong learner).
    
    Why XGBoost?
    - State-of-the-art for regression: wins Kaggle competitions
    - Builds trees sequentially, each correcting previous errors (smarter than Random Forest)
    - Handles non-linear patterns better: especially good for volatility spikes
    - Fast training on large datasets
    
    Hyperparameters:
    - n_estimators=100: 100 boosting rounds; each tree adds to prediction
    - max_depth=5: shallow trees (prevents overfitting on small volatility dataset)
    - learning_rate=0.1: shrinkage factor; lower = more conservative updates, generalizes better
    
    Ensemble Logic:
    Round 1: Train tree on raw data
    Round 2: Train tree on residuals (errors) from round 1
    Round 3: Train tree on residuals from round 2
    ...continues 100 times, combining all predictions
    
    Expected performance: Best overall, but may overfit without regularization.
    """
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X, y)
    return model 