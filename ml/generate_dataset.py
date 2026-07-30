"""
LifeLink — ML Dataset Generator
Generates synthetic donor eligibility dataset for training.
"""

import pandas as pd
import numpy as np
import os

# Seed for reproducibility
np.random.seed(42)

N = 5000  # Number of samples

def generate_eligibility_dataset(n=N):
    """
    Generate synthetic donor eligibility dataset.

    Features:
        age           : 16 - 70
        weight        : 35 - 100 kg
        days_since_donation : 0 - 365 (0 = never donated)
        has_donated_before  : 0 / 1
        hemoglobin    : 8.0 - 18.0 g/dL
        systolic_bp   : 80 - 160 mmHg
        diastolic_bp  : 50 - 100 mmHg
        pulse         : 50 - 110 bpm

    Target:
        eligible : 0 (Not Eligible) / 1 (Eligible)
    """

    age                  = np.random.randint(16, 71, n)
    weight               = np.round(np.random.uniform(35, 100, n), 1)
    has_donated_before   = np.random.randint(0, 2, n)
    days_since_donation  = np.where(
        has_donated_before == 1,
        np.random.randint(1, 366, n),
        0
    )
    hemoglobin           = np.round(np.random.uniform(8.0, 18.0, n), 1)
    systolic_bp          = np.random.randint(80, 161, n)
    diastolic_bp         = np.random.randint(50, 101, n)
    pulse                = np.random.randint(50, 111, n)

    # ── Rule-based eligibility label ──────────────────────
    eligible = np.ones(n, dtype=int)

    # Age: must be 18–65
    eligible = np.where((age < 18) | (age > 65), 0, eligible)

    # Weight: must be >= 50 kg
    eligible = np.where(weight < 50, 0, eligible)

    # Donation gap: if donated before, must be >= 90 days
    eligible = np.where(
        (has_donated_before == 1) & (days_since_donation < 90),
        0, eligible
    )

    # Hemoglobin: women typically need >= 12.5, men >= 13.0
    # Using 12.0 as a general lower bound for synthetic data
    eligible = np.where(hemoglobin < 12.0, 0, eligible)

    # Blood pressure: systolic 90–140, diastolic 50–90
    eligible = np.where(
        (systolic_bp < 90) | (systolic_bp > 140), 0, eligible)
    eligible = np.where(
        (diastolic_bp < 50) | (diastolic_bp > 90), 0, eligible)

    # Pulse: 60–100 bpm
    eligible = np.where(
        (pulse < 60) | (pulse > 100), 0, eligible)

    # Add slight noise (5%) to simulate real-world edge cases
    noise_mask = np.random.random(n) < 0.05
    eligible   = np.where(noise_mask, 1 - eligible, eligible)

    df = pd.DataFrame({
        'age':                 age,
        'weight':              weight,
        'has_donated_before':  has_donated_before,
        'days_since_donation': days_since_donation,
        'hemoglobin':          hemoglobin,
        'systolic_bp':         systolic_bp,
        'diastolic_bp':        diastolic_bp,
        'pulse':               pulse,
        'eligible':            eligible
    })

    return df


def generate_recommendation_dataset(n=N):
    """
    Generate synthetic donor recommendation dataset.

    Features:
        blood_group_match  : 0 / 1
        same_district      : 0 / 1
        age                : 18–65
        weight             : 45–100
        days_since_donation: 0–365
        is_eligible        : 0 / 1
        emergency_level    : 1 (low) – 3 (high)
        availability       : 0 / 1

    Target:
        recommendation_score : 0.0 – 1.0
    """

    blood_group_match   = np.random.randint(0, 2, n)
    same_district       = np.random.randint(0, 2, n)
    age                 = np.random.randint(18, 66, n)
    weight              = np.round(np.random.uniform(45, 100, n), 1)
    days_since_donation = np.random.randint(0, 366, n)
    is_eligible         = np.random.randint(0, 2, n)
    emergency_level     = np.random.randint(1, 4, n)
    availability        = np.random.randint(0, 2, n)

    # Score based on weighted rules
    score = (
        blood_group_match  * 0.35 +
        same_district      * 0.20 +
        is_eligible        * 0.20 +
        availability       * 0.10 +
        (emergency_level / 3.0) * 0.10 +
        (np.clip(days_since_donation, 0, 365) / 365.0) * 0.05
    )

    # Add noise
    score = np.clip(score + np.random.normal(0, 0.05, n), 0.0, 1.0)
    score = np.round(score, 4)

    df = pd.DataFrame({
        'blood_group_match':   blood_group_match,
        'same_district':       same_district,
        'age':                 age,
        'weight':              weight,
        'days_since_donation': days_since_donation,
        'is_eligible':         is_eligible,
        'emergency_level':     emergency_level,
        'availability':        availability,
        'recommendation_score': score
    })

    return df


def generate_forecasting_dataset():
    """
    Generate synthetic monthly blood demand dataset.

    Features:
        month             : 1–12
        blood_group       : A+, A-, B+, B-, AB+, AB-, O+, O-
        district          : Tamil Nadu districts
        registrations     : donor registrations that month
        emergency_requests: emergency requests that month

    Target:
        demand : predicted units needed
    """

    districts = [
        "Chennai","Coimbatore","Madurai","Salem","Trichy",
        "Vellore","Erode","Tirunelveli","Thoothukudi","Namakkal",
        "Dharmapuri","Krishnagiri","Dindigul","Thanjavur","Karur"
    ]
    blood_groups = ['A+','A-','B+','B-','AB+','AB-','O+','O-']
    months       = list(range(1, 13))
    years        = [2022, 2023, 2024]

    records = []
    for year in years:
        for month in months:
            for district in districts:
                for bg in blood_groups:
                    regs = np.random.randint(5, 80)
                    emg  = np.random.randint(0, 15)

                    # Seasonal factor: higher demand in summer (Apr-Jun)
                    seasonal = 1.3 if month in [4, 5, 6] else 1.0

                    # Blood group weights (O+ most common)
                    bg_weight = {
                        'O+':1.5,'A+':1.3,'B+':1.2,'AB+':0.8,
                        'O-':1.1,'A-':0.9,'B-':0.8,'AB-':0.6
                    }.get(bg, 1.0)

                    demand = int(
                        (regs * 0.6 + emg * 2.0) * seasonal * bg_weight
                        + np.random.randint(-5, 10)
                    )
                    demand = max(demand, 1)

                    records.append({
                        'year':              year,
                        'month':             month,
                        'district':          district,
                        'blood_group':       bg,
                        'registrations':     regs,
                        'emergency_requests':emg,
                        'demand':            demand
                    })

    return pd.DataFrame(records)


if __name__ == '__main__':
    # Create datasets folder
    os.makedirs('datasets', exist_ok=True)

    print("Generating eligibility dataset...")
    elig_df = generate_eligibility_dataset()
    elig_df.to_csv('datasets/eligibility_data.csv', index=False)
    print(f"✅ Eligibility dataset: {len(elig_df)} rows → datasets/eligibility_data.csv")
    print(f"   Eligible: {elig_df['eligible'].sum()} | "
          f"Not Eligible: {(elig_df['eligible']==0).sum()}")

    print("\nGenerating recommendation dataset...")
    rec_df = generate_recommendation_dataset()
    rec_df.to_csv('datasets/recommendation_data.csv', index=False)
    print(f"✅ Recommendation dataset: {len(rec_df)} rows → datasets/recommendation_data.csv")

    print("\nGenerating forecasting dataset...")
    fore_df = generate_forecasting_dataset()
    fore_df.to_csv('datasets/forecasting_data.csv', index=False)
    print(f"✅ Forecasting dataset: {len(fore_df)} rows → datasets/forecasting_data.csv")

    print("\n✅ All datasets generated successfully!")