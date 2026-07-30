"""
LifeLink — Eligibility Model Trainer
Trains a Random Forest classifier for donor eligibility prediction.
Saves model to models/eligibility_model.pkl
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
from sklearn.pipeline import Pipeline

# ── Paths ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'datasets', 'eligibility_data.csv')
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
MODEL_PATH   = os.path.join(MODEL_DIR, 'eligibility_model.pkl')
SCALER_PATH  = os.path.join(MODEL_DIR, 'eligibility_scaler.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """Load and validate the eligibility dataset."""
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        print("   Run ml/generate_dataset.py first!")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Dataset loaded: {len(df)} rows, {df.shape[1]} columns")
    print(f"   Eligible: {df['eligible'].sum()} | "
          f"Not Eligible: {(df['eligible']==0).sum()}")
    return df


def preprocess(df):
    """Split features and target, then train/test split."""
    feature_cols = [
        'age', 'weight', 'has_donated_before',
        'days_since_donation', 'hemoglobin',
        'systolic_bp', 'diastolic_bp', 'pulse'
    ]
    X = df[feature_cols]
    y = df['eligible']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_cols


def train(X_train, y_train):
    """Train a Random Forest classifier with scaling pipeline."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("\n🔄 Training Random Forest...")
    pipeline.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                cv=5, scoring='accuracy')
    print(f"✅ Cross-val Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return pipeline


def evaluate(pipeline, X_test, y_test):
    """Evaluate the trained model."""
    y_pred      = pipeline.predict(X_test)
    y_prob      = pipeline.predict_proba(X_test)[:, 1]

    acc         = accuracy_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_prob)

    print(f"\n📈 Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"📈 ROC AUC Score : {auc:.4f}")
    print(f"\n📋 Classification Report:\n")
    print(classification_report(y_test, y_pred,
          target_names=['Not Eligible', 'Eligible']))

    print("📋 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"   FN={cm[1,0]}  TP={cm[1,1]}")

    return acc, auc


def save_model(pipeline, feature_cols):
    """Save the trained pipeline and metadata."""
    model_data = {
        'pipeline':     pipeline,
        'feature_cols': feature_cols,
        'model_type':   'RandomForestClassifier',
        'version':      '1.0'
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")


def main():
    print("=" * 55)
    print("  LifeLink — Eligibility Model Trainer")
    print("=" * 55)

    df                              = load_data()
    X_train, X_test, y_train, y_test, feature_cols = preprocess(df)
    pipeline                        = train(X_train, y_train)
    acc, auc                        = evaluate(pipeline, X_test, y_test)
    save_model(pipeline, feature_cols)

    print("\n" + "=" * 55)
    print(f"  ✅ Training Complete!")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  AUC      : {auc:.4f}")
    print("=" * 55)


if __name__ == '__main__':
    main()