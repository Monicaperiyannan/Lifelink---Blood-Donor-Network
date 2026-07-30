"""
LifeLink — ML Predictor
Central module for all ML predictions.
Loaded once at app startup, used across routes.
"""

import os
import numpy as np
import joblib
from datetime import datetime, timedelta

# ── Paths ──────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR     = os.path.join(BASE_DIR, 'models')

ELIGIBILITY_MODEL_PATH   = os.path.join(MODELS_DIR, 'eligibility_model.pkl')
RECOMMENDATION_MODEL_PATH = os.path.join(MODELS_DIR, 'recommendation_model.pkl')
FORECASTING_MODEL_PATH   = os.path.join(MODELS_DIR, 'forecasting_model.pkl')


# ══════════════════════════════════════════════════════════
# MODEL LOADERS (lazy loading with caching)
# ══════════════════════════════════════════════════════════

_eligibility_model   = None
_recommendation_model = None
_forecasting_model   = None


def _load_eligibility_model():
    global _eligibility_model
    if _eligibility_model is None:
        if not os.path.exists(ELIGIBILITY_MODEL_PATH):
            return None
        _eligibility_model = joblib.load(ELIGIBILITY_MODEL_PATH)
    return _eligibility_model


def _load_recommendation_model():
    global _recommendation_model
    if _recommendation_model is None:
        if not os.path.exists(RECOMMENDATION_MODEL_PATH):
            return None
        _recommendation_model = joblib.load(RECOMMENDATION_MODEL_PATH)
    return _recommendation_model


def _load_forecasting_model():
    global _forecasting_model
    if _forecasting_model is None:
        if not os.path.exists(FORECASTING_MODEL_PATH):
            return None
        _forecasting_model = joblib.load(FORECASTING_MODEL_PATH)
    return _forecasting_model


# ══════════════════════════════════════════════════════════
# 1. ELIGIBILITY PREDICTION
# ══════════════════════════════════════════════════════════

def predict_eligibility(age, weight, last_donated_date=None,
                        hemoglobin=14.0, systolic_bp=120,
                        diastolic_bp=80, pulse=72):
    """
    Predict donor eligibility using the trained ML model.

    Parameters:
        age              : int
        weight           : float (kg)
        last_donated_date: str 'YYYY-MM-DD' or None
        hemoglobin       : float (g/dL) — default 14.0
        systolic_bp      : int (mmHg)   — default 120
        diastolic_bp     : int (mmHg)   — default 80
        pulse            : int (bpm)    — default 72

    Returns:
        dict with keys:
            is_eligible     : bool
            confidence      : float (0–100)
            probability_eligible   : float (0–100)
            probability_ineligible : float (0–100)
            reasons         : list[str]
            model_used      : str
    """

    # ── Calculate days since donation ─────────────────────
    has_donated_before  = 0
    days_since_donation = 0

    if last_donated_date:
        try:
            last_date = datetime.strptime(last_donated_date, '%Y-%m-%d').date()
            today     = datetime.now().date()
            if last_date <= today:
                has_donated_before  = 1
                days_since_donation = (today - last_date).days
        except ValueError:
            pass

    # ── Try ML model first ─────────────────────────────────
    model_data = _load_eligibility_model()

    if model_data:
        try:
            pipeline     = model_data['pipeline']
            feature_cols = model_data['feature_cols']

            features = np.array([[
                age, weight, has_donated_before,
                days_since_donation, hemoglobin,
                systolic_bp, diastolic_bp, pulse
            ]])

            import pandas as pd
            X = pd.DataFrame(features, columns=feature_cols)

            prediction  = pipeline.predict(X)[0]
            probabilities = pipeline.predict_proba(X)[0]

            prob_not_eligible = round(probabilities[0] * 100, 1)
            prob_eligible     = round(probabilities[1] * 100, 1)
            is_eligible       = bool(prediction == 1)
            confidence        = max(prob_eligible, prob_not_eligible)

            reasons = _generate_eligibility_reasons(
                age, weight, has_donated_before,
                days_since_donation, is_eligible
            )

            return {
                'is_eligible':            is_eligible,
                'confidence':             confidence,
                'probability_eligible':   prob_eligible,
                'probability_ineligible': prob_not_eligible,
                'reasons':                reasons,
                'model_used':             'RandomForest ML Model',
                'days_since_donation':    days_since_donation
                    if has_donated_before else None
            }

        except Exception as e:
            print(f"⚠️ ML model error: {e}. Falling back to rule-based.")

    # ── Fallback: rule-based prediction ───────────────────
    return _rule_based_eligibility(
        age, weight, has_donated_before, days_since_donation
    )


def _generate_eligibility_reasons(age, weight, has_donated_before,
                                   days_since_donation, is_eligible):
    """Generate human-readable reasons for the eligibility result."""
    reasons = []

    if is_eligible:
        if 18 <= age <= 65:
            reasons.append(f"✔ Age {age} is within eligible range (18–65)")
        if weight >= 50:
            reasons.append(f"✔ Weight {weight} kg meets minimum (50 kg)")
        if has_donated_before and days_since_donation >= 90:
            reasons.append(
                f"✔ {days_since_donation} days since last donation (≥90 days)")
        elif not has_donated_before:
            reasons.append("✔ First-time donor — no gap restriction")
    else:
        if age < 18:
            reasons.append(f"✘ Age {age} is below minimum (18 years)")
        elif age > 65:
            reasons.append(f"✘ Age {age} exceeds maximum (65 years)")
        if weight < 50:
            reasons.append(f"✘ Weight {weight} kg is below minimum (50 kg)")
        if has_donated_before and days_since_donation < 90:
            remaining = 90 - days_since_donation
            reasons.append(
                f"✘ Only {days_since_donation} days since last donation. "
                f"Wait {remaining} more days.")

    return reasons


def _rule_based_eligibility(age, weight, has_donated_before,
                             days_since_donation):
    """Fallback rule-based eligibility check (no ML model needed)."""
    reasons     = []
    is_eligible = True

    if not (18 <= age <= 65):
        is_eligible = False
        reasons.append(
            f"✘ Age {age} is outside eligible range (18–65)")
    else:
        reasons.append(f"✔ Age {age} is within eligible range (18–65)")

    if weight < 50:
        is_eligible = False
        reasons.append(f"✘ Weight {weight} kg is below minimum (50 kg)")
    else:
        reasons.append(f"✔ Weight {weight} kg meets minimum (50 kg)")

    if has_donated_before:
        if days_since_donation < 90:
            is_eligible = False
            remaining   = 90 - days_since_donation
            reasons.append(
                f"✘ Only {days_since_donation} days since last donation. "
                f"Wait {remaining} more days.")
        else:
            reasons.append(
                f"✔ {days_since_donation} days since last donation (≥90 days)")
    else:
        reasons.append("✔ First-time donor — no gap restriction")

    confidence = 92.0 if is_eligible else 88.0

    return {
        'is_eligible':            is_eligible,
        'confidence':             confidence,
        'probability_eligible':   confidence if is_eligible else 100 - confidence,
        'probability_ineligible': 100 - confidence if is_eligible else confidence,
        'reasons':                reasons,
        'model_used':             'Rule-Based System',
        'days_since_donation':    days_since_donation
            if has_donated_before else None
    }


# ══════════════════════════════════════════════════════════
# 2. DONOR RECOMMENDATION
# ══════════════════════════════════════════════════════════

def compute_recommendation_score(donor, target_blood_group,
                                  target_district, emergency_level=2):
    """
    Compute recommendation score for a single donor.

    Parameters:
        donor              : sqlite3.Row or dict
        target_blood_group : str (e.g. 'O+')
        target_district    : str (e.g. 'Chennai')
        emergency_level    : int 1–3

    Returns:
        float score 0.0–1.0
    """
    model_data = _load_recommendation_model()

    # ── Feature engineering ────────────────────────────────
    blood_group_match = 1 if donor['blood_group'] == target_blood_group else 0
    same_district     = 1 if (
        donor['city'].lower() == target_district.lower()) else 0

    age    = donor['age']
    weight = donor['weight']

    # Days since last donation
    days_since = 0
    if donor['last_donated']:
        try:
            last_date  = datetime.strptime(donor['last_donated'], '%Y-%m-%d').date()
            days_since = (datetime.now().date() - last_date).days
        except ValueError:
            days_since = 0

    # Rough eligibility check
    is_eligible = int(
        18 <= age <= 65 and
        weight >= 45 and
        (donor['last_donated'] is None or days_since >= 90)
    )

    availability = int(donor.get('is_available', 1))

    if model_data:
        try:
            pipeline     = model_data['pipeline']
            feature_cols = model_data['feature_cols']

            features = np.array([[
                blood_group_match, same_district, age, weight,
                days_since, is_eligible, emergency_level, availability
            ]])

            import pandas as pd
            X     = pd.DataFrame(features, columns=feature_cols)
            score = float(np.clip(pipeline.predict(X)[0], 0.0, 1.0))
            return score

        except Exception as e:
            print(f"⚠️ Recommendation model error: {e}")

    # ── Fallback weighted scoring ──────────────────────────
    score = (
        blood_group_match * 0.35 +
        same_district     * 0.20 +
        is_eligible       * 0.20 +
        availability      * 0.10 +
        (emergency_level / 3.0) * 0.10 +
        (min(days_since, 365) / 365.0) * 0.05
    )
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def get_ai_recommendations(donors, target_blood_group,
                            target_district, emergency_level=2, top_n=5):
    """
    Score and rank donors, returning top N AI recommendations.

    Parameters:
        donors             : list of sqlite3.Row objects
        target_blood_group : str
        target_district    : str
        emergency_level    : int 1–3
        top_n              : int (default 5)

    Returns:
        list of dicts with donor info + AI scores + reasons
    """
    if not donors:
        return []

    scored = []

    for donor in donors:
        score = compute_recommendation_score(
            donor, target_blood_group,
            target_district, emergency_level
        )

        # Generate recommendation reasons
        reasons = _generate_recommendation_reasons(
            donor, target_blood_group,
            target_district, score
        )

        # Confidence: based on score and number of matching criteria
        confidence = round(min(score * 110, 99.0), 1)

        scored.append({
            'donor':             donor,
            'score':             score,
            'score_pct':         round(score * 100, 1),
            'confidence':        confidence,
            'reasons':           reasons,
            'recommendation_level': _get_recommendation_level(score)
        })

    # Sort by score descending
    scored.sort(key=lambda x: x['score'], reverse=True)

    # Return top N
    return scored[:top_n]


def _generate_recommendation_reasons(donor, target_bg,
                                       target_district, score):
    """Generate human-readable recommendation reasons."""
    reasons = []

    if donor['blood_group'] == target_bg:
        reasons.append("✔ Exact Blood Group Match")
    else:
        reasons.append("~ Different Blood Group")

    if donor['city'].lower() == target_district.lower():
        reasons.append("✔ Same District")
    else:
        reasons.append(f"~ Located in {donor['city']}")

    if 18 <= donor['age'] <= 65 and donor['weight'] >= 45:
        reasons.append("✔ Meets Basic Eligibility")

    if donor['last_donated']:
        try:
            last_date  = datetime.strptime(
                donor['last_donated'], '%Y-%m-%d').date()
            days_since = (datetime.now().date() - last_date).days
            if days_since >= 180:
                reasons.append(
                    f"✔ Long Time Since Last Donation ({days_since} days)")
            elif days_since >= 90:
                reasons.append(
                    f"✔ Eligible Gap ({days_since} days since last donation)")
        except ValueError:
            pass
    else:
        reasons.append("✔ Never Donated — Fully Ready")

    if donor.get('is_available', 1):
        reasons.append("✔ Currently Available")

    return reasons


def _get_recommendation_level(score):
    """Return recommendation level label based on score."""
    if score >= 0.85:
        return 'Highly Recommended'
    elif score >= 0.65:
        return 'Recommended'
    elif score >= 0.45:
        return 'Possible Match'
    else:
        return 'Low Match'


# ══════════════════════════════════════════════════════════
# 3. BLOOD DEMAND FORECASTING
# ══════════════════════════════════════════════════════════

def forecast_blood_demand(blood_groups=None, districts=None,
                           month=None, registrations=50,
                           emergency_requests=5):
    """
    Forecast blood demand for next month.

    Parameters:
        blood_groups       : list[str] — blood groups to forecast
        districts          : list[str] — districts to forecast
        month              : int (1–12) — target month, default = next month
        registrations      : int — expected registrations
        emergency_requests : int — expected emergency requests

    Returns:
        dict with:
            blood_group_demand : dict {blood_group: demand}
            district_demand    : dict {district: demand}
            total_demand       : int
            forecast_month     : str
            model_used         : str
    """

    default_blood_groups = ['A+','A-','B+','B-','AB+','AB-','O+','O-']
    default_districts = [
        "Chennai","Coimbatore","Madurai","Salem","Trichy",
        "Vellore","Erode","Tirunelveli","Thoothukudi","Namakkal"
    ]

    blood_groups = blood_groups or default_blood_groups
    districts    = districts    or default_districts

    if month is None:
        next_month = datetime.now().replace(day=1) + timedelta(days=32)
        month      = next_month.month

    model_data = _load_forecasting_model()

    if model_data:
        try:
            return _ml_forecast(
                model_data, blood_groups, districts,
                month, registrations, emergency_requests
            )
        except Exception as e:
            print(f"⚠️ Forecasting model error: {e}")

    # Fallback
    return _rule_based_forecast(blood_groups, districts, month,
                                 registrations, emergency_requests)


def _ml_forecast(model_data, blood_groups, districts,
                  month, registrations, emergency_requests):
    """Use trained ML model to forecast demand."""
    import pandas as pd

    pipeline    = model_data['pipeline']
    bg_encoder  = model_data['bg_encoder']
    dist_encoder = model_data['dist_encoder']

    quarter   = (1 if month in [1,2,3] else
                 2 if month in [4,5,6] else
                 3 if month in [7,8,9] else 4)
    is_summer = 1 if month in [4,5,6] else 0

    blood_group_demand = {}
    district_demand    = {}

    # Forecast per blood group
    for bg in blood_groups:
        if bg in bg_encoder.classes_:
            bg_enc   = bg_encoder.transform([bg])[0]
            # Use first available district for bg-level forecast
            dist_enc = 0
            X = pd.DataFrame([[
                month, quarter, is_summer,
                bg_enc, dist_enc, registrations, emergency_requests
            ]], columns=model_data['feature_cols'])
            pred = max(int(pipeline.predict(X)[0]), 1)
            blood_group_demand[bg] = pred
        else:
            blood_group_demand[bg] = _default_bg_demand(bg, month)

    # Forecast per district (using most common blood group O+)
    for dist in districts:
        if dist in dist_encoder.classes_:
            dist_enc = dist_encoder.transform([dist])[0]
            bg_enc   = (bg_encoder.transform(['O+'])[0]
                        if 'O+' in bg_encoder.classes_ else 0)
            X = pd.DataFrame([[
                month, quarter, is_summer,
                bg_enc, dist_enc, registrations, emergency_requests
            ]], columns=model_data['feature_cols'])
            pred = max(int(pipeline.predict(X)[0]), 1)
            district_demand[dist] = pred
        else:
            district_demand[dist] = max(registrations // 2, 5)

    total_demand    = sum(blood_group_demand.values())
    forecast_month  = datetime(2024, month, 1).strftime('%B %Y')

    return {
        'blood_group_demand': blood_group_demand,
        'district_demand':    district_demand,
        'total_demand':       total_demand,
        'forecast_month':     forecast_month,
        'model_used':         'RandomForest ML Model'
    }


def _rule_based_forecast(blood_groups, districts, month,
                          registrations, emergency_requests):
    """Fallback rule-based forecast."""
    seasonal = 1.3 if month in [4, 5, 6] else 1.0

    bg_weights = {
        'O+':1.5,'A+':1.3,'B+':1.2,'AB+':0.8,
        'O-':1.1,'A-':0.9,'B-':0.8,'AB-':0.6
    }

    blood_group_demand = {
        bg: max(int((registrations * 0.6 + emergency_requests * 2)
                    * seasonal * bg_weights.get(bg, 1.0)), 1)
        for bg in blood_groups
    }

    district_demand = {
        dist: max(int((registrations * 0.4 + emergency_requests)
                      * seasonal), 1)
        for dist in districts
    }

    forecast_month = datetime(2024, month, 1).strftime('%B %Y')

    return {
        'blood_group_demand': blood_group_demand,
        'district_demand':    district_demand,
        'total_demand':       sum(blood_group_demand.values()),
        'forecast_month':     forecast_month,
        'model_used':         'Rule-Based System'
    }


def _default_bg_demand(blood_group, month):
    """Default demand estimate per blood group."""
    base = {'O+':45,'A+':35,'B+':30,'AB+':15,
            'O-':20,'A-':15,'B-':12,'AB-':8}.get(blood_group, 20)
    seasonal = 1.3 if month in [4,5,6] else 1.0
    return max(int(base * seasonal), 1)


# ══════════════════════════════════════════════════════════
# UTILITY: Check which models are loaded
# ══════════════════════════════════════════════════════════

def get_model_status():
    """Return dict showing which models are available."""
    return {
        'eligibility':    os.path.exists(ELIGIBILITY_MODEL_PATH),
        'recommendation': os.path.exists(RECOMMENDATION_MODEL_PATH),
        'forecasting':    os.path.exists(FORECASTING_MODEL_PATH),
    }