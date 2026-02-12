"""
Sample Healthcare Dataset Generator
Generates realistic patient data for testing the analytics platform
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_dataset(n_patients: int = 1000) -> pd.DataFrame:
    """Generate a sample healthcare dataset with realistic patient data"""
    np.random.seed(42)
    random.seed(42)
    
    # Patient demographics
    patient_ids = [f"P{str(i).zfill(5)}" for i in range(1, n_patients + 1)]
    ages = np.random.normal(55, 18, n_patients).clip(18, 95).astype(int)
    genders = np.random.choice(['Male', 'Female'], n_patients, p=[0.48, 0.52])
    
    # Ethnic backgrounds
    ethnicities = np.random.choice(
        ['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'],
        n_patients,
        p=[0.55, 0.18, 0.15, 0.08, 0.04]
    )
    
    # Insurance types
    insurance_types = np.random.choice(
        ['Private', 'Medicare', 'Medicaid', 'Self-Pay', 'Other'],
        n_patients,
        p=[0.40, 0.30, 0.15, 0.10, 0.05]
    )
    
    # Primary diagnoses
    diagnoses = np.random.choice(
        ['Diabetes', 'Heart Disease', 'COPD', 'Pneumonia', 'Heart Failure', 
         'Hypertension', 'Kidney Disease', 'Stroke', 'Cancer', 'Arthritis'],
        n_patients,
        p=[0.15, 0.12, 0.10, 0.08, 0.12, 0.18, 0.08, 0.05, 0.07, 0.05]
    )
    
    # Comorbidities count (0-5)
    comorbidities = np.random.poisson(1.5, n_patients).clip(0, 5)
    
    # Length of stay (days)
    los = np.random.exponential(5, n_patients).clip(1, 60).astype(int)
    
    # Number of previous admissions (0-10)
    prev_admissions = np.random.poisson(1.2, n_patients).clip(0, 10)
    
    # Number of medications
    num_medications = np.random.poisson(4, n_patients).clip(0, 15)
    
    # Lab values
    hemoglobin = np.random.normal(12.5, 2, n_patients).clip(6, 18).round(1)
    creatinine = np.random.lognormal(0, 0.5, n_patients).clip(0.5, 8).round(2)
    glucose = np.random.normal(120, 40, n_patients).clip(60, 400).astype(int)
    blood_pressure_systolic = np.random.normal(130, 20, n_patients).clip(80, 200).astype(int)
    blood_pressure_diastolic = np.random.normal(80, 12, n_patients).clip(50, 120).astype(int)
    
    # BMI
    bmi = np.random.normal(28, 6, n_patients).clip(15, 50).round(1)
    
    # Smoking status
    smoking_status = np.random.choice(
        ['Never', 'Former', 'Current'],
        n_patients,
        p=[0.45, 0.35, 0.20]
    )
    
    # Emergency admission
    emergency_admission = np.random.choice([0, 1], n_patients, p=[0.6, 0.4])
    
    # ICU stay
    icu_stay = np.random.choice([0, 1], n_patients, p=[0.85, 0.15])
    
    # Generate readmission risk based on factors
    # Higher risk for: older age, more comorbidities, longer LOS, more prev admissions
    risk_score = (
        (ages / 100) * 0.15 +
        (comorbidities / 5) * 0.25 +
        (los / 60) * 0.15 +
        (prev_admissions / 10) * 0.20 +
        (num_medications / 15) * 0.10 +
        (creatinine / 8) * 0.10 +
        emergency_admission * 0.15 +
        icu_stay * 0.20
    )
    risk_score = (risk_score / risk_score.max()).clip(0, 1)
    
    # 30-day readmission (binary outcome based on risk score with randomness)
    readmission_prob = risk_score * 0.7 + np.random.uniform(0, 0.3, n_patients)
    readmitted_30_days = (readmission_prob > 0.5).astype(int)
    
    # Complication risk
    complication_occurred = np.random.choice([0, 1], n_patients, p=[0.75, 0.25])
    
    # Admission dates (last 2 years)
    base_date = datetime.now() - timedelta(days=730)
    admission_dates = [base_date + timedelta(days=int(np.random.uniform(0, 730))) for _ in range(n_patients)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'age': ages,
        'gender': genders,
        'ethnicity': ethnicities,
        'insurance_type': insurance_types,
        'primary_diagnosis': diagnoses,
        'comorbidities_count': comorbidities,
        'length_of_stay': los,
        'previous_admissions': prev_admissions,
        'num_medications': num_medications,
        'hemoglobin': hemoglobin,
        'creatinine': creatinine,
        'glucose': glucose,
        'blood_pressure_systolic': blood_pressure_systolic,
        'blood_pressure_diastolic': blood_pressure_diastolic,
        'bmi': bmi,
        'smoking_status': smoking_status,
        'emergency_admission': emergency_admission,
        'icu_stay': icu_stay,
        'risk_score': risk_score.round(3),
        'readmitted_30_days': readmitted_30_days,
        'complication_occurred': complication_occurred,
        'admission_date': admission_dates
    })
    
    return df


def get_sample_data_description() -> str:
    """Return description of the sample dataset columns"""
    return """
    ### Sample Dataset Columns:
    
    **Demographics:**
    - `patient_id`: Unique patient identifier
    - `age`: Patient age (18-95)
    - `gender`: Male/Female
    - `ethnicity`: Patient ethnic background
    - `insurance_type`: Type of insurance coverage
    
    **Clinical Information:**
    - `primary_diagnosis`: Main diagnosis for admission
    - `comorbidities_count`: Number of comorbid conditions (0-5)
    - `length_of_stay`: Hospital stay duration in days
    - `previous_admissions`: Number of prior hospitalizations
    - `num_medications`: Number of medications prescribed
    
    **Lab Values:**
    - `hemoglobin`: Blood hemoglobin level (g/dL)
    - `creatinine`: Kidney function marker (mg/dL)
    - `glucose`: Blood glucose level (mg/dL)
    - `blood_pressure_systolic/diastolic`: Blood pressure readings
    - `bmi`: Body Mass Index
    
    **Risk Factors:**
    - `smoking_status`: Never/Former/Current smoker
    - `emergency_admission`: Was this an emergency? (0/1)
    - `icu_stay`: Required ICU care? (0/1)
    
    **Outcomes:**
    - `risk_score`: Calculated risk score (0-1)
    - `readmitted_30_days`: Readmitted within 30 days? (0/1)
    - `complication_occurred`: Had complications? (0/1)
    - `admission_date`: Date of hospital admission
    """
