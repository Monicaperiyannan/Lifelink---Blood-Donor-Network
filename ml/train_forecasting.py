"""
LifeLink — Blood Demand Forecasting Model Trainer
Trains a Random Forest regressor to predict future
blood demand by district and blood group.
Saves model to models/forecasting_model.pkl
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# ── Paths ──────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'datasets', 'forecasting_data.csv')
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
MODEL_PATH   = os.path.join(MODEL_DIR, 'forecasting_model.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """Load and validate the forecasting dataset."""
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        print("   Run ml/generate_dataset.py first!")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Dataset loaded: {len(df)} rows")
    print(f"   Demand range: {df['demand'].min()} – {df['demand'].max()}")
    print(f"   Blood groups: {df['blood_group'].nunique()}")
    print(f"   Districts   : {df['district'].nunique()}")
    return df


def preprocess(df):
    """
    Encode categorical features and prepare train/test split.
    Returns encoded data + encoders for later use in prediction.
    """
    df = df.copy()

    # Label encode blood_group and district
    bg_encoder = LabelEncoder()
    dist_encoder = LabelEncoder()

    df['blood_group_enc'] = bg_encoder.fit_transform(df['blood_group'])
    df['district_enc']    = dist_encoder.fit_transform(df['district'])

    # Add seasonal feature (quarter)
    df['quarter'] = df['month'].apply(lambda m:
        1 if m in [1,2,3] else
        2 if m in [4,5,6] else
        3 if m in [7,8,9] else 4
    )

    # Add is_summer flag (high demand months)
    df['is_summer'] = df['month'].apply(lambda m: 1 if m in [4,5,6] else 0)

    feature_cols = [
        'month', 'quarter', 'is_summer',
        'blood_group_enc', 'district_enc',
        'registrations', 'emergency_requests'
    ]

    X = df[feature_cols]
    y = df['demand']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"   Features: {feature_cols}")

    return (X_train, X_test, y_train, y_test,
            feature_cols, bg_encoder, dist_encoder)


def train(X_train, y_train):
    """Train a Random Forest Regressor for demand forecasting."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("\n🔄 Training Random Forest Regressor...")
    pipeline.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring='r2')
    print(f"✅ Cross-val R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return pipeline


def evaluate(pipeline, X_test, y_test):
    """Evaluate the demand forecasting model."""
    y_pred = pipeline.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)  # No negative demand

    mse   = mean_squared_error(y_test, y_pred)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(y_test, y_pred)
    r2    = r2_score(y_test, y_pred)

    print(f"\n📈 Test RMSE : {rmse:.2f} units")
    print(f"📈 Test MAE  : {mae:.2f} units")
    print(f"📈 Test R²   : {r2:.4f}")

    # Sample predictions
    sample_idx = np.random.choice(len(X_test), 5, replace=False)
    X_sample   = X_test.iloc[sample_idx]
    y_actual   = y_test.iloc[sample_idx].values
    y_sample   = np.clip(pipeline.predict(X_sample), 0, None)

    print(f"\n📊 Sample Predictions:")
    print(f"   {'Actual':>8} | {'Predicted':>10} | {'Error':>8}")
    print(f"   {'-'*32}")
    for act, pred in zip(y_actual, y_sample):
        print(f"   {act:>8} | {pred:>10.1f} | {abs(act-pred):>8.1f}")

    return rmse, r2


def save_model(pipeline, feature_cols, bg_encoder, dist_encoder):
    """Save the trained model, encoders, and metadata."""
    model_data = {
        'pipeline':      pipeline,
        'feature_cols':  feature_cols,
        'bg_encoder':    bg_encoder,
        'dist_encoder':  dist_encoder,
        'blood_groups':  list(bg_encoder.classes_),
        'districts':     list(dist_encoder.classes_),
        'model_type':    'RandomForestRegressor',
        'version':       '1.0'
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")


def main():
    print("=" * 55)
    print("  LifeLink — Forecasting Model Trainer")
    print("=" * 55)

    df = load_data()
    (X_train, X_test, y_train, y_test,
     feat_cols, bg_enc, dist_enc) = preprocess(df)
    pipeline                       = train(X_train, y_train)
    rmse, r2                       = evaluate(pipeline, X_test, y_test)
    save_model(pipeline, feat_cols, bg_enc, dist_enc)

    print("\n" + "=" * 55)
    print(f"  ✅ Training Complete!")
    print(f"  RMSE : {rmse:.2f} units")
    print(f"  R²   : {r2:.4f}")
    print("=" * 55)


if __name__ == '__main__':
    main()