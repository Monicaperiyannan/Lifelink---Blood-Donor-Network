"""
LifeLink — Donor Recommendation Model Trainer
Trains a Gradient Boosting regressor to score donors
based on blood group match, district, eligibility, etc.
Saves model to models/recommendation_model.pkl
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# ── Paths ──────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'datasets', 'recommendation_data.csv')
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
MODEL_PATH   = os.path.join(MODEL_DIR, 'recommendation_model.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """Load and validate the recommendation dataset."""
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        print("   Run ml/generate_dataset.py first!")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Dataset loaded: {len(df)} rows")
    print(f"   Score range: {df['recommendation_score'].min():.3f} – "
          f"{df['recommendation_score'].max():.3f}")
    return df


def preprocess(df):
    """Prepare features and target."""
    feature_cols = [
        'blood_group_match',
        'same_district',
        'age',
        'weight',
        'days_since_donation',
        'is_eligible',
        'emergency_level',
        'availability'
    ]

    X = df[feature_cols]
    y = df['recommendation_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_cols


def train(X_train, y_train):
    """Train a Gradient Boosting Regressor."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.85,
            random_state=42
        ))
    ])

    print("\n🔄 Training Gradient Boosting Regressor...")
    pipeline.fit(X_train, y_train)

    # Cross-validation R² score
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring='r2')
    print(f"✅ Cross-val R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return pipeline


def evaluate(pipeline, X_test, y_test):
    """Evaluate the trained model."""
    y_pred = pipeline.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, 1.0)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n📈 Test RMSE : {rmse:.4f}")
    print(f"📈 Test MAE  : {mae:.4f}")
    print(f"📈 Test R²   : {r2:.4f}")

    # Score distribution
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    hist, _ = np.histogram(y_pred, bins=bins)
    print(f"\n📊 Predicted Score Distribution:")
    labels = ['0–30%','30–50%','50–70%','70–90%','90–100%']
    for label, count in zip(labels, hist):
        print(f"   {label}: {count}")

    return rmse, r2


def save_model(pipeline, feature_cols):
    """Save the trained pipeline and metadata."""
    model_data = {
        'pipeline':     pipeline,
        'feature_cols': feature_cols,
        'model_type':   'GradientBoostingRegressor',
        'version':      '1.0',
        'score_range':  (0.0, 1.0)
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")


def main():
    print("=" * 55)
    print("  LifeLink — Recommendation Model Trainer")
    print("=" * 55)

    df                                          = load_data()
    X_train, X_test, y_train, y_test, feat_cols = preprocess(df)
    pipeline                                    = train(X_train, y_train)
    rmse, r2                                    = evaluate(pipeline, X_test, y_test)
    save_model(pipeline, feat_cols)

    print("\n" + "=" * 55)
    print(f"  ✅ Training Complete!")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    print("=" * 55)


if __name__ == '__main__':
    main()