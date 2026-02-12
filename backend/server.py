from fastapi import FastAPI, APIRouter, Response, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import json
import io

# Import healthcare modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from modules.sample_data import generate_sample_dataset
from modules.data_validation import DataValidator, ValidationSeverity
from modules.feature_engineering import FeatureEngineer
from modules.model_manager import ModelManager, RiskSegmenter

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="Healthcare Risk Analytics API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# In-memory storage for session data
session_data: Dict[str, Any] = {}
model_manager = ModelManager()
feature_engineer = FeatureEngineer()

# Individual Patient Assessment Model
class PatientAssessment(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Patient age")
    gender: str = Field(..., description="Male or Female")
    comorbidities_count: int = Field(default=0, ge=0, le=10)
    length_of_stay: int = Field(default=3, ge=1, le=100)
    previous_admissions: int = Field(default=0, ge=0, le=20)
    num_medications: int = Field(default=2, ge=0, le=30)
    hemoglobin: float = Field(default=12.5, ge=5, le=20)
    creatinine: float = Field(default=1.0, ge=0.1, le=15)
    glucose: int = Field(default=100, ge=40, le=500)
    blood_pressure_systolic: int = Field(default=120, ge=70, le=250)
    blood_pressure_diastolic: int = Field(default=80, ge=40, le=150)
    bmi: float = Field(default=25, ge=10, le=60)
    smoking_status: str = Field(default="Never", description="Never, Former, or Current")
    emergency_admission: int = Field(default=0, ge=0, le=1)
    icu_stay: int = Field(default=0, ge=0, le=1)

# Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class DatasetInfo(BaseModel):
    rows: int
    columns: int
    column_names: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    missing_percentage: float

class ValidationResult(BaseModel):
    severity: str
    column: Optional[str]
    message: str
    suggestion: Optional[str]

class TrainingRequest(BaseModel):
    target_column: str
    feature_columns: List[str]
    test_size: float = 0.2

class PredictionResult(BaseModel):
    patient_id: str
    risk_probability: float
    risk_category: str
    predicted_outcome: int

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Healthcare Risk Analytics API", "version": "1.0.0"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    return status_checks

@api_router.post("/generate-sample")
async def generate_sample(n_patients: int = 1000):
    """Generate sample healthcare dataset"""
    try:
        df = generate_sample_dataset(n_patients)
        session_data['dataset'] = df
        session_data['validated'] = False
        
        return {
            "success": True,
            "message": f"Generated {n_patients} patient records",
            "info": {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file for analysis"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        session_data['dataset'] = df
        session_data['validated'] = False
        
        return {
            "success": True,
            "message": f"Uploaded {file.filename}",
            "info": {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.get("/dataset-info")
async def get_dataset_info():
    """Get information about the current dataset"""
    if 'dataset' not in session_data:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    
    df = session_data['dataset']
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "missing_percentage": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        "validated": session_data.get('validated', False),
        "model_trained": session_data.get('model_trained', False)
    }

@api_router.post("/validate")
async def validate_dataset():
    """Validate the current dataset"""
    if 'dataset' not in session_data:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    
    df = session_data['dataset']
    validator = DataValidator(df)
    messages = validator.validate()
    
    session_data['validator'] = validator
    session_data['validated'] = True
    
    return {
        "success": True,
        "schema": validator.get_schema_summary().to_dict(orient='records'),
        "target_candidates": validator.target_candidates,
        "recommended_target": validator.get_recommended_target(),
        "messages": [
            {
                "severity": msg.severity.value,
                "column": msg.column,
                "message": msg.message,
                "suggestion": msg.suggestion
            }
            for msg in messages
        ]
    }

@api_router.post("/train")
async def train_models(request: TrainingRequest):
    """Train ML models on the dataset"""
    if 'dataset' not in session_data:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    
    df = session_data['dataset']
    
    try:
        # Feature engineering
        fe = FeatureEngineer()
        df_engineered = fe.fit_transform(
            df[request.feature_columns + [request.target_column]].copy(),
            target_column=request.target_column,
            create_derived=True
        )
        
        feature_cols = fe.get_numeric_features(df_engineered, exclude_columns=[request.target_column])
        
        # Train models
        mm = ModelManager()
        results = mm.train_pipeline(
            df_engineered,
            target_column=request.target_column,
            feature_columns=feature_cols,
            test_size=request.test_size
        )
        
        # Store in session
        session_data['feature_engineer'] = fe
        session_data['model_manager'] = mm
        session_data['model_trained'] = True
        session_data['target_column'] = request.target_column
        session_data['original_features'] = request.feature_columns
        
        return {
            "success": True,
            "best_model": results['best_model'],
            "results": {
                model_name: {
                    "accuracy": metrics['accuracy'],
                    "precision": metrics['precision'],
                    "recall": metrics['recall'],
                    "f1_score": metrics['f1_score'],
                    "roc_auc": metrics['roc_auc']
                }
                for model_name, metrics in results['results'].items()
            },
            "feature_importance": mm.get_feature_importance().head(15).to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/predict")
async def get_predictions(model: str = "best"):
    """Generate predictions for all patients"""
    if 'model_manager' not in session_data or not session_data.get('model_trained'):
        raise HTTPException(status_code=400, detail="Models not trained")
    
    df = session_data['dataset']
    mm = session_data['model_manager']
    fe = session_data['feature_engineer']
    target = session_data['target_column']
    original_features = session_data.get('original_features', [])
    
    try:
        # Re-apply feature engineering with original feature columns
        fe_new = FeatureEngineer()
        cols_to_use = original_features + [target] if target in df.columns else original_features
        df_pred = fe_new.fit_transform(
            df[cols_to_use].copy(),
            target_column=target if target in df.columns else None,
            create_derived=True
        )
        
        # Ensure we have all required features
        missing_features = [f for f in mm.feature_columns if f not in df_pred.columns]
        for f in missing_features:
            df_pred[f] = 0
        
        predictions, probabilities = mm.predict(df_pred, model)
        
        # Risk segmentation
        segmenter = RiskSegmenter()
        categories, stats = segmenter.segment(probabilities)
        
        return {
            "success": True,
            "total_patients": len(df),
            "risk_segments": stats,
            "predictions": [
                {
                    "index": i,
                    "risk_probability": round(float(prob), 4),
                    "risk_category": cat,
                    "predicted_outcome": int(pred)
                }
                for i, (prob, cat, pred) in enumerate(zip(probabilities, categories, predictions))
            ][:100]  # Limit to first 100 for API response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/export-predictions")
async def export_predictions(model: str = "best"):
    """Export predictions as CSV"""
    if 'model_manager' not in session_data or not session_data.get('model_trained'):
        raise HTTPException(status_code=400, detail="Models not trained")
    
    df = session_data['dataset']
    mm = session_data['model_manager']
    target = session_data['target_column']
    original_features = session_data.get('original_features', [])
    
    try:
        # Re-apply feature engineering
        fe_new = FeatureEngineer()
        cols_to_use = original_features + [target] if target in df.columns else original_features
        df_pred = fe_new.fit_transform(
            df[cols_to_use].copy(),
            target_column=target if target in df.columns else None,
            create_derived=True
        )
        
        # Ensure we have all required features
        missing_features = [f for f in mm.feature_columns if f not in df_pred.columns]
        for f in missing_features:
            df_pred[f] = 0
        
        predictions, probabilities = mm.predict(df_pred, model)
        
        segmenter = RiskSegmenter()
        categories, _ = segmenter.segment(probabilities)
        
        # Create export dataframe
        export_df = df.copy()
        export_df['Risk_Probability'] = probabilities.round(4)
        export_df['Risk_Category'] = categories
        export_df['Predicted_Outcome'] = predictions
        
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        
        return StreamingResponse(
            io.BytesIO(csv_buffer.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/feature-importance")
async def get_feature_importance(model: str = "random_forest"):
    """Get feature importance from trained model"""
    if 'model_manager' not in session_data or not session_data.get('model_trained'):
        raise HTTPException(status_code=400, detail="Models not trained")
    
    mm = session_data['model_manager']
    importance_df = mm.get_feature_importance(model)
    
    return {
        "success": True,
        "features": importance_df.to_dict(orient='records')
    }

@api_router.post("/assess-patient")
async def assess_individual_patient(patient: PatientAssessment):
    """Assess risk for an individual patient"""
    if 'model_manager' not in session_data or not session_data.get('model_trained'):
        raise HTTPException(status_code=400, detail="Models not trained. Please train models first.")
    
    mm = session_data['model_manager']
    original_features = session_data.get('original_features', [])
    target = session_data.get('target_column', 'readmitted_30_days')
    
    try:
        # Create DataFrame from patient data
        patient_dict = patient.model_dump()
        patient_df = pd.DataFrame([patient_dict])
        
        # Add placeholder for target column
        patient_df[target] = 0
        
        # Feature engineering
        fe_new = FeatureEngineer()
        cols_to_use = [c for c in original_features if c in patient_df.columns] + [target]
        
        # Add missing original features with defaults
        for col in original_features:
            if col not in patient_df.columns:
                patient_df[col] = 0
        
        patient_engineered = fe_new.fit_transform(
            patient_df[original_features + [target]].copy(),
            target_column=target,
            create_derived=True
        )
        
        # Ensure all model features exist
        for f in mm.feature_columns:
            if f not in patient_engineered.columns:
                patient_engineered[f] = 0
        
        # Get prediction
        predictions, probabilities = mm.predict(patient_engineered, 'best')
        risk_prob = float(probabilities[0])
        
        # Determine risk category
        if risk_prob < 0.3:
            risk_category = "Low Risk"
            risk_color = "#22c55e"
        elif risk_prob < 0.7:
            risk_category = "Medium Risk"
            risk_color = "#f59e0b"
        else:
            risk_category = "High Risk"
            risk_color = "#ef4444"
        
        # Get feature contributions (approximate using feature importance)
        importance_df = mm.get_feature_importance('best')
        top_factors = []
        
        # Identify key risk factors for this patient
        risk_explanations = {
            'age': f"Age {patient.age}" + (" (elderly)" if patient.age >= 65 else ""),
            'comorbidities_count': f"{patient.comorbidities_count} comorbidities" if patient.comorbidities_count > 0 else None,
            'previous_admissions': f"{patient.previous_admissions} previous admissions" if patient.previous_admissions > 0 else None,
            'emergency_admission': "Emergency admission" if patient.emergency_admission == 1 else None,
            'icu_stay': "Required ICU care" if patient.icu_stay == 1 else None,
            'length_of_stay': f"{patient.length_of_stay} day hospital stay" if patient.length_of_stay > 5 else None,
            'creatinine': f"Elevated creatinine ({patient.creatinine})" if patient.creatinine > 1.5 else None,
            'glucose': f"High glucose ({patient.glucose})" if patient.glucose > 126 else None,
            'bmi': f"BMI {patient.bmi}" + (" (obese)" if patient.bmi >= 30 else ""),
            'smoking_status': f"Smoking: {patient.smoking_status}" if patient.smoking_status != "Never" else None,
        }
        
        for _, row in importance_df.head(10).iterrows():
            feature = row['Feature']
            # Map engineered features back to original
            base_feature = feature.split('_')[0] if '_' in feature else feature
            if base_feature in risk_explanations and risk_explanations[base_feature]:
                top_factors.append({
                    'factor': risk_explanations[base_feature],
                    'importance': round(row['Importance %'], 1)
                })
        
        # Remove duplicates and limit
        seen = set()
        unique_factors = []
        for f in top_factors:
            if f['factor'] not in seen:
                seen.add(f['factor'])
                unique_factors.append(f)
        
        return {
            "success": True,
            "risk_probability": round(risk_prob, 4),
            "risk_percentage": round(risk_prob * 100, 1),
            "risk_category": risk_category,
            "risk_color": risk_color,
            "predicted_readmission": bool(predictions[0]),
            "risk_factors": unique_factors[:5],
            "recommendation": get_risk_recommendation(risk_category, patient)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_risk_recommendation(risk_category: str, patient: PatientAssessment) -> str:
    """Generate personalized recommendations based on risk"""
    if risk_category == "High Risk":
        recs = ["Schedule follow-up within 7 days", "Consider care coordination program"]
        if patient.comorbidities_count >= 3:
            recs.append("Multi-disciplinary care team review recommended")
        if patient.previous_admissions >= 2:
            recs.append("Evaluate for readmission prevention program")
        return " | ".join(recs)
    elif risk_category == "Medium Risk":
        recs = ["Schedule follow-up within 14 days", "Ensure medication reconciliation"]
        if patient.emergency_admission:
            recs.append("Review discharge planning")
        return " | ".join(recs)
    else:
        return "Standard follow-up care | Patient education on warning signs"

@api_router.get("/model-metrics")
async def get_model_metrics():
    """Get detailed model metrics including ROC curve and confusion matrix data"""
    if 'model_manager' not in session_data or not session_data.get('model_trained'):
        raise HTTPException(status_code=400, detail="Models not trained")
    
    mm = session_data['model_manager']
    metadata = mm.metadata
    
    # Get stored training results if available
    results = {}
    for model_name in ['logistic_regression', 'random_forest', 'gradient_boosting']:
        if model_name in metadata.get('model_metrics', {}):
            results[model_name] = metadata['model_metrics'][model_name]
    
    return {
        "success": True,
        "best_model": metadata.get('best_model'),
        "metrics": results,
        "training_date": metadata.get('training_date'),
        "n_samples": metadata.get('n_samples'),
        "n_features": metadata.get('n_features')
    }

@api_router.get("/dashboard-stats")
async def get_dashboard_stats():
    """Get comprehensive dashboard statistics"""
    if 'dataset' not in session_data:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    
    df = session_data['dataset']
    
    stats = {
        "total_patients": len(df),
        "total_columns": len(df.columns),
        "validated": session_data.get('validated', False),
        "model_trained": session_data.get('model_trained', False)
    }
    
    # Calculate additional stats if columns exist
    if 'age' in df.columns:
        stats['avg_age'] = round(df['age'].mean(), 1)
        stats['age_distribution'] = df['age'].describe().to_dict()
    
    if 'readmitted_30_days' in df.columns:
        stats['readmission_rate'] = round(df['readmitted_30_days'].mean() * 100, 1)
        stats['readmitted_count'] = int(df['readmitted_30_days'].sum())
    
    if 'risk_score' in df.columns:
        stats['avg_risk_score'] = round(df['risk_score'].mean(), 3)
        stats['high_risk_count'] = int((df['risk_score'] >= 0.7).sum())
        stats['medium_risk_count'] = int(((df['risk_score'] >= 0.3) & (df['risk_score'] < 0.7)).sum())
        stats['low_risk_count'] = int((df['risk_score'] < 0.3).sum())
    
    if 'gender' in df.columns:
        stats['gender_distribution'] = df['gender'].value_counts().to_dict()
    
    if 'primary_diagnosis' in df.columns:
        stats['top_diagnoses'] = df['primary_diagnosis'].value_counts().head(5).to_dict()
    
    if 'comorbidities_count' in df.columns:
        stats['avg_comorbidities'] = round(df['comorbidities_count'].mean(), 1)
    
    if 'length_of_stay' in df.columns:
        stats['avg_los'] = round(df['length_of_stay'].mean(), 1)
    
    if 'emergency_admission' in df.columns:
        stats['emergency_rate'] = round(df['emergency_admission'].mean() * 100, 1)
    
    if 'icu_stay' in df.columns:
        stats['icu_rate'] = round(df['icu_stay'].mean() * 100, 1)
    
    return stats

@api_router.get("/chart-data/{chart_type}")
async def get_chart_data(chart_type: str, filter_gender: Optional[str] = None, filter_age_min: Optional[int] = None, filter_age_max: Optional[int] = None):
    """Get data for interactive charts with optional filters"""
    if 'dataset' not in session_data:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    
    df = session_data['dataset'].copy()
    
    # Apply filters
    if filter_gender and 'gender' in df.columns:
        df = df[df['gender'] == filter_gender]
    if filter_age_min and 'age' in df.columns:
        df = df[df['age'] >= filter_age_min]
    if filter_age_max and 'age' in df.columns:
        df = df[df['age'] <= filter_age_max]
    
    if chart_type == "age_distribution":
        if 'age' not in df.columns:
            return {"success": False, "error": "Age column not found"}
        bins = list(range(0, 101, 10))
        hist, edges = np.histogram(df['age'].dropna(), bins=bins)
        return {
            "success": True,
            "data": {
                "labels": [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges)-1)],
                "values": hist.tolist()
            }
        }
    
    elif chart_type == "risk_by_diagnosis":
        if 'primary_diagnosis' not in df.columns or 'readmitted_30_days' not in df.columns:
            return {"success": False, "error": "Required columns not found"}
        risk_by_diag = df.groupby('primary_diagnosis')['readmitted_30_days'].agg(['mean', 'count']).reset_index()
        risk_by_diag.columns = ['diagnosis', 'risk_rate', 'count']
        risk_by_diag['risk_rate'] = (risk_by_diag['risk_rate'] * 100).round(1)
        risk_by_diag = risk_by_diag.sort_values('risk_rate', ascending=False)
        return {
            "success": True,
            "data": risk_by_diag.to_dict(orient='records')
        }
    
    elif chart_type == "risk_by_age_group":
        if 'age' not in df.columns or 'risk_score' not in df.columns:
            return {"success": False, "error": "Required columns not found"}
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 75, 100], labels=['18-30', '31-45', '46-60', '61-75', '75+'])
        risk_by_age = df.groupby('age_group')['risk_score'].agg(['mean', 'count']).reset_index()
        risk_by_age.columns = ['age_group', 'avg_risk', 'count']
        risk_by_age['avg_risk'] = risk_by_age['avg_risk'].round(3)
        return {
            "success": True,
            "data": risk_by_age.to_dict(orient='records')
        }
    
    elif chart_type == "gender_risk":
        if 'gender' not in df.columns or 'readmitted_30_days' not in df.columns:
            return {"success": False, "error": "Required columns not found"}
        gender_risk = df.groupby('gender').agg({
            'readmitted_30_days': 'mean',
            'risk_score': 'mean' if 'risk_score' in df.columns else 'count'
        }).reset_index()
        gender_risk['readmitted_30_days'] = (gender_risk['readmitted_30_days'] * 100).round(1)
        if 'risk_score' in df.columns:
            gender_risk['risk_score'] = gender_risk['risk_score'].round(3)
        return {
            "success": True,
            "data": gender_risk.to_dict(orient='records')
        }
    
    elif chart_type == "comorbidity_impact":
        if 'comorbidities_count' not in df.columns or 'readmitted_30_days' not in df.columns:
            return {"success": False, "error": "Required columns not found"}
        comorbid_risk = df.groupby('comorbidities_count')['readmitted_30_days'].agg(['mean', 'count']).reset_index()
        comorbid_risk.columns = ['comorbidities', 'risk_rate', 'count']
        comorbid_risk['risk_rate'] = (comorbid_risk['risk_rate'] * 100).round(1)
        return {
            "success": True,
            "data": comorbid_risk.to_dict(orient='records')
        }
    
    else:
        return {"success": False, "error": f"Unknown chart type: {chart_type}"}

@api_router.get("/executive-summary")
async def get_executive_summary():
    """Get executive business insights for leadership presentation"""
    if 'dataset' not in session_data:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    
    df = session_data['dataset']
    model_trained = session_data.get('model_trained', False)
    
    # Base statistics
    total_patients = len(df)
    
    # Calculate key business metrics
    insights = {
        "total_patients": total_patients,
        "analysis_date": datetime.now().strftime("%B %d, %Y"),
        "model_status": "Operational" if model_trained else "Not Deployed"
    }
    
    # Readmission analysis
    if 'readmitted_30_days' in df.columns:
        current_readmission_rate = df['readmitted_30_days'].mean()
        readmitted_count = int(df['readmitted_30_days'].sum())
        
        insights['current_readmission_rate'] = round(current_readmission_rate * 100, 1)
        insights['readmitted_patients'] = readmitted_count
        
        # Estimate potential reduction (industry benchmark: 15-25% reduction with predictive models)
        potential_reduction_rate = 0.20  # 20% reduction potential
        preventable_readmissions = int(readmitted_count * potential_reduction_rate)
        insights['preventable_readmissions'] = preventable_readmissions
        insights['potential_reduction_percent'] = 20
        
        # Cost estimation (average readmission cost: $15,000-$25,000)
        avg_readmission_cost = 17500
        potential_savings = preventable_readmissions * avg_readmission_cost
        insights['estimated_cost_savings'] = potential_savings
        insights['cost_per_readmission'] = avg_readmission_cost
    
    # Risk stratification
    if 'risk_score' in df.columns:
        high_risk = (df['risk_score'] >= 0.7).sum()
        medium_risk = ((df['risk_score'] >= 0.3) & (df['risk_score'] < 0.7)).sum()
        low_risk = (df['risk_score'] < 0.3).sum()
        
        insights['risk_distribution'] = {
            'high_risk': {'count': int(high_risk), 'percentage': round(high_risk/total_patients*100, 1)},
            'medium_risk': {'count': int(medium_risk), 'percentage': round(medium_risk/total_patients*100, 1)},
            'low_risk': {'count': int(low_risk), 'percentage': round(low_risk/total_patients*100, 1)}
        }
        
        # High-risk patient intervention potential
        insights['intervention_target'] = int(high_risk)
        insights['intervention_percentage'] = round(high_risk/total_patients*100, 1)
    
    # Top risk factors from model
    if model_trained and 'model_manager' in session_data:
        mm = session_data['model_manager']
        importance_df = mm.get_feature_importance('best')
        if not importance_df.empty:
            insights['top_risk_factors'] = importance_df.head(5)['Feature'].tolist()
    
    # Generate business recommendations
    insights['recommendations'] = generate_business_recommendations(insights)
    
    return insights

def generate_business_recommendations(insights: dict) -> List[dict]:
    """Generate actionable business recommendations"""
    recommendations = []
    
    if insights.get('intervention_percentage', 0) > 10:
        recommendations.append({
            'priority': 'High',
            'category': 'Care Management',
            'title': 'Implement High-Risk Patient Program',
            'description': f"Target {insights.get('intervention_target', 0)} high-risk patients with enhanced care coordination to reduce readmissions.",
            'expected_impact': 'Reduce readmission rate by 15-20%'
        })
    
    if insights.get('potential_savings', 0) > 100000:
        recommendations.append({
            'priority': 'High',
            'category': 'Financial',
            'title': 'Launch Readmission Prevention Initiative',
            'description': f"Potential annual savings of ${insights.get('estimated_cost_savings', 0):,} through proactive intervention.",
            'expected_impact': f"Prevent {insights.get('preventable_readmissions', 0)} readmissions annually"
        })
    
    recommendations.append({
        'priority': 'Medium',
        'category': 'Operations',
        'title': 'Integrate Risk Scores into Discharge Workflow',
        'description': 'Display patient risk scores in EMR system during discharge planning.',
        'expected_impact': 'Improve care team decision-making'
    })
    
    recommendations.append({
        'priority': 'Medium',
        'category': 'Quality',
        'title': 'Establish Weekly High-Risk Patient Review',
        'description': 'Schedule care team meetings to review high-risk patients before discharge.',
        'expected_impact': 'Ensure appropriate follow-up care'
    })
    
    return recommendations

@api_router.post("/scenario-simulation")
async def simulate_scenario(patient: PatientAssessment, variations: Optional[List[dict]] = None):
    """Simulate risk changes based on parameter variations"""
    if 'model_manager' not in session_data or not session_data.get('model_trained'):
        raise HTTPException(status_code=400, detail="Models not trained")
    
    mm = session_data['model_manager']
    original_features = session_data.get('original_features', [])
    target = session_data.get('target_column', 'readmitted_30_days')
    
    try:
        # Get base prediction
        patient_dict = patient.model_dump()
        base_result = await calculate_patient_risk(patient_dict, mm, original_features, target)
        
        results = {
            'base_scenario': {
                'risk_probability': base_result['risk_probability'],
                'risk_category': base_result['risk_category'],
                'parameters': patient_dict
            },
            'variations': []
        }
        
        # Pre-defined scenario variations
        scenario_variations = [
            {'name': 'Age +10 years', 'field': 'age', 'delta': 10},
            {'name': 'Age -10 years', 'field': 'age', 'delta': -10},
            {'name': '+2 Comorbidities', 'field': 'comorbidities_count', 'delta': 2},
            {'name': '-1 Comorbidity', 'field': 'comorbidities_count', 'delta': -1},
            {'name': 'Emergency Admission', 'field': 'emergency_admission', 'value': 1},
            {'name': 'Non-Emergency', 'field': 'emergency_admission', 'value': 0},
            {'name': 'ICU Required', 'field': 'icu_stay', 'value': 1},
            {'name': 'No ICU', 'field': 'icu_stay', 'value': 0},
            {'name': 'High Creatinine (2.5)', 'field': 'creatinine', 'value': 2.5},
            {'name': 'Normal Creatinine (1.0)', 'field': 'creatinine', 'value': 1.0},
        ]
        
        for variation in scenario_variations:
            varied_patient = patient_dict.copy()
            
            if 'delta' in variation:
                current_val = varied_patient.get(variation['field'], 0)
                new_val = max(0, current_val + variation['delta'])
                # Apply bounds
                if variation['field'] == 'age':
                    new_val = max(18, min(100, new_val))
                elif variation['field'] == 'comorbidities_count':
                    new_val = max(0, min(10, new_val))
                varied_patient[variation['field']] = new_val
            else:
                varied_patient[variation['field']] = variation['value']
            
            varied_result = await calculate_patient_risk(varied_patient, mm, original_features, target)
            
            risk_change = varied_result['risk_probability'] - base_result['risk_probability']
            
            results['variations'].append({
                'scenario': variation['name'],
                'risk_probability': varied_result['risk_probability'],
                'risk_category': varied_result['risk_category'],
                'risk_change': round(risk_change * 100, 1),
                'change_direction': 'increase' if risk_change > 0 else 'decrease' if risk_change < 0 else 'no_change'
            })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def calculate_patient_risk(patient_dict: dict, mm: ModelManager, original_features: list, target: str) -> dict:
    """Helper function to calculate patient risk"""
    patient_df = pd.DataFrame([patient_dict])
    patient_df[target] = 0
    
    for col in original_features:
        if col not in patient_df.columns:
            patient_df[col] = 0
    
    fe_new = FeatureEngineer()
    patient_engineered = fe_new.fit_transform(
        patient_df[original_features + [target]].copy(),
        target_column=target,
        create_derived=True
    )
    
    for f in mm.feature_columns:
        if f not in patient_engineered.columns:
            patient_engineered[f] = 0
    
    predictions, probabilities = mm.predict(patient_engineered, 'best')
    risk_prob = float(probabilities[0])
    
    if risk_prob < 0.3:
        risk_category = "Low Risk"
    elif risk_prob < 0.7:
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"
    
    return {
        'risk_probability': round(risk_prob, 4),
        'risk_category': risk_category
    }

@api_router.get("/care-recommendations/{risk_level}")
async def get_care_recommendations(risk_level: str):
    """Get detailed care recommendations based on risk level"""
    recommendations = {
        'high': {
            'level': 'High Risk',
            'color': '#ef4444',
            'summary': 'Immediate intervention required. Patient has significant risk of 30-day readmission.',
            'actions': [
                {
                    'category': 'Follow-up Care',
                    'icon': 'calendar',
                    'items': [
                        'Schedule follow-up appointment within 7 days of discharge',
                        'Arrange home health visit within 48 hours',
                        'Enroll in transitional care management program'
                    ]
                },
                {
                    'category': 'Medication Management',
                    'icon': 'pill',
                    'items': [
                        'Complete medication reconciliation before discharge',
                        'Schedule pharmacist consultation',
                        'Ensure patient has medication supply for 30 days',
                        'Set up medication reminders or pill organizer'
                    ]
                },
                {
                    'category': 'Care Coordination',
                    'icon': 'users',
                    'items': [
                        'Assign dedicated care coordinator',
                        'Notify primary care physician of discharge',
                        'Schedule multidisciplinary team review',
                        'Document clear escalation pathway'
                    ]
                },
                {
                    'category': 'Patient Education',
                    'icon': 'book',
                    'items': [
                        'Provide disease-specific education materials',
                        'Review warning signs requiring immediate care',
                        'Ensure patient understands medication regimen',
                        'Provide 24/7 nurse hotline number'
                    ]
                }
            ],
            'monitoring': [
                'Daily phone check-ins for first week',
                'Vital signs monitoring if applicable',
                'Weight tracking for heart failure patients',
                'Blood glucose monitoring for diabetic patients'
            ]
        },
        'medium': {
            'level': 'Medium Risk',
            'color': '#f59e0b',
            'summary': 'Enhanced monitoring recommended. Patient has moderate risk factors requiring attention.',
            'actions': [
                {
                    'category': 'Follow-up Care',
                    'icon': 'calendar',
                    'items': [
                        'Schedule follow-up appointment within 14 days',
                        'Consider telehealth check-in at day 7',
                        'Provide clear instructions for appointment scheduling'
                    ]
                },
                {
                    'category': 'Medication Management',
                    'icon': 'pill',
                    'items': [
                        'Review all medications at discharge',
                        'Provide medication list to patient and family',
                        'Discuss potential side effects to watch for'
                    ]
                },
                {
                    'category': 'Patient Education',
                    'icon': 'book',
                    'items': [
                        'Review discharge instructions thoroughly',
                        'Provide written materials on condition management',
                        'Discuss lifestyle modifications'
                    ]
                }
            ],
            'monitoring': [
                'Phone check-in at day 3 and day 10',
                'Self-monitoring guidelines provided',
                'Clear criteria for when to seek care'
            ]
        },
        'low': {
            'level': 'Low Risk',
            'color': '#10b981',
            'summary': 'Standard care pathway appropriate. Patient has low probability of readmission.',
            'actions': [
                {
                    'category': 'Follow-up Care',
                    'icon': 'calendar',
                    'items': [
                        'Schedule routine follow-up within 30 days',
                        'Provide contact information for questions'
                    ]
                },
                {
                    'category': 'Patient Education',
                    'icon': 'book',
                    'items': [
                        'Provide standard discharge instructions',
                        'Review warning signs requiring medical attention'
                    ]
                }
            ],
            'monitoring': [
                'Standard post-discharge protocols',
                'Patient-initiated contact as needed'
            ]
        }
    }
    
    risk_key = risk_level.lower().replace(' ', '').replace('-', '').replace('_', '')
    if risk_key in ['high', 'highrisk']:
        return recommendations['high']
    elif risk_key in ['medium', 'mediumrisk', 'moderate']:
        return recommendations['medium']
    elif risk_key in ['low', 'lowrisk']:
        return recommendations['low']
    else:
        raise HTTPException(status_code=400, detail=f"Invalid risk level: {risk_level}")

@api_router.get("/generate-readme")
async def generate_readme():
    """Generate GitHub README documentation"""
    model_trained = session_data.get('model_trained', False)
    
    # Get model metrics if available
    model_metrics = ""
    if model_trained and 'model_manager' in session_data:
        mm = session_data['model_manager']
        metadata = mm.metadata
        best_model = metadata.get('best_model', 'N/A')
        metrics = metadata.get('model_metrics', {})
        
        if metrics:
            model_metrics = "| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |\n"
            model_metrics += "|-------|----------|-----------|--------|----------|----------|\n"
            for name, m in metrics.items():
                model_metrics += f"| {name.replace('_', ' ').title()} | {m['accuracy']:.1%} | {m['precision']:.1%} | {m['recall']:.1%} | {m['f1_score']:.1%} | {m['roc_auc']:.1%} |\n"
    
    readme_content = f'''# Healthcare Risk Analytics Platform

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![React](https://img.shields.io/badge/React-18.x-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A production-ready machine learning platform for predicting patient readmission risk and enabling proactive healthcare interventions.

## 🎯 Problem Statement

Hospital readmissions within 30 days are a significant quality and cost concern in healthcare:
- **15-20%** of patients are readmitted within 30 days
- Average readmission costs **$15,000-$25,000** per patient
- Early identification enables **preventive interventions**

This platform uses machine learning to identify high-risk patients before discharge, enabling targeted care coordination.

## ✨ Key Features

### Patient Risk Assessment
- **Individual Assessment Form**: Enter patient details for instant risk prediction
- **Risk Gauge Visualization**: Clear visual representation of risk score (0-100%)
- **Key Risk Factors**: Understand which factors contribute most to risk
- **Care Recommendations**: Actionable recommendations based on risk level

### Executive Dashboard
- **KPI Cards**: Total patients, high-risk count, readmission rate, cost impact
- **Interactive Charts**: Risk by diagnosis, age group distribution, trends
- **Business Insights**: Estimated savings and intervention recommendations

### Machine Learning Models
- **3 ML Models**: Logistic Regression, Random Forest, Gradient Boosting
- **Automatic Comparison**: Side-by-side model performance metrics
- **Feature Importance**: Understand which features drive predictions

### Scenario Simulation
- **What-If Analysis**: See how changing patient parameters affects risk
- **Dynamic Recalculation**: Real-time risk score updates

## 📊 Dataset Description

The platform works with healthcare datasets containing:

| Feature Category | Variables |
|------------------|-----------|
| Demographics | Age, Gender, Ethnicity, Insurance Type |
| Clinical | Primary Diagnosis, Comorbidities, Medications |
| Vitals | Blood Pressure, BMI, Hemoglobin, Creatinine, Glucose |
| Admission | Length of Stay, Emergency Admission, ICU Stay |
| Historical | Previous Admissions, Prior Readmissions |
| Target | 30-Day Readmission (Binary: Yes/No) |

## 🤖 Machine Learning Approach

### Feature Engineering
- **Derived Features**: Age groups, BMI categories, hypertension flags
- **Interaction Features**: Age-comorbidity score, utilization score
- **Encoding**: Label encoding for categorical variables
- **Scaling**: StandardScaler for numerical features

### Model Training
- **Train/Test Split**: 80/20 stratified split
- **Class Balancing**: Weighted classes for imbalanced data
- **Cross-Validation**: 5-fold CV for robust evaluation

### Model Performance
{model_metrics if model_metrics else "Train models to see performance metrics."}

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
Node.js 16+
MongoDB
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/healthcare-risk-analytics.git
cd healthcare-risk-analytics

# Backend setup
cd backend
pip install -r requirements.txt

# Frontend setup
cd ../frontend
yarn install
```

### Running the Application
```bash
# Start backend (port 8001)
cd backend
uvicorn server:app --host 0.0.0.0 --port 8001 --reload

# Start frontend (port 3000)
cd frontend
yarn start
```

## 📸 Screenshots

### Dashboard
*KPI cards showing patient statistics and risk distribution*

### Patient Assessment
*Individual patient risk assessment with gauge visualization*

### Model Performance
*ML model comparison with feature importance*

### Risk Predictions
*Batch predictions with risk segmentation*

## 📁 Project Structure

```
├── backend/
│   ├── server.py              # FastAPI application
│   ├── modules/
│   │   ├── data_validation.py # Schema detection
│   │   ├── feature_engineering.py # Feature creation
│   │   ├── model_manager.py   # ML training
│   │   └── sample_data.py     # Data generation
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js            # Main React component
│   │   └── App.css           # Styling
│   └── package.json
└── README.md
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate-sample` | POST | Generate sample dataset |
| `/api/validate` | POST | Validate and analyze schema |
| `/api/train` | POST | Train ML models |
| `/api/predict` | GET | Batch predictions |
| `/api/assess-patient` | POST | Individual assessment |
| `/api/dashboard-stats` | GET | Dashboard KPIs |
| `/api/executive-summary` | GET | Business insights |
| `/api/scenario-simulation` | POST | What-if analysis |
| `/api/export-predictions` | GET | Download CSV |

## 💼 Business Value

- **Reduce Readmissions**: Target high-risk patients with interventions
- **Cost Savings**: Estimated 20% reduction in preventable readmissions
- **Quality Improvement**: Better discharge planning and follow-up
- **Care Coordination**: Actionable recommendations for care teams

## 🛠️ Technologies

- **Backend**: Python, FastAPI, scikit-learn, pandas, numpy
- **Frontend**: React, CSS3, Plotly
- **Database**: MongoDB
- **ML Models**: Logistic Regression, Random Forest, Gradient Boosting

## 📄 License

MIT License - feel free to use this project for learning and portfolio purposes.

## 👤 Author

Built with ❤️ for healthcare innovation

---

*This project demonstrates production-ready ML application development for healthcare use cases.*
'''
    
    return {
        "success": True,
        "content": readme_content,
        "filename": "README.md"
    }

@api_router.get("/demo-steps")
async def get_demo_steps():
    """Get guided demo walkthrough steps"""
    current_state = {
        'has_data': 'dataset' in session_data,
        'validated': session_data.get('validated', False),
        'model_trained': session_data.get('model_trained', False),
        'has_predictions': 'predictions' in session_data
    }
    
    steps = [
        {
            'step': 1,
            'title': 'Generate Sample Data',
            'description': 'Create 1,000 realistic patient records for demonstration',
            'action': 'generate-sample',
            'completed': current_state['has_data'],
            'icon': 'database'
        },
        {
            'step': 2,
            'title': 'Validate Dataset',
            'description': 'Analyze schema and detect target column for prediction',
            'action': 'validate',
            'completed': current_state['validated'],
            'icon': 'check-circle',
            'requires': 'generate-sample'
        },
        {
            'step': 3,
            'title': 'Train ML Models',
            'description': 'Train Logistic Regression, Random Forest, and Gradient Boosting',
            'action': 'train',
            'completed': current_state['model_trained'],
            'icon': 'cpu',
            'requires': 'validate'
        },
        {
            'step': 4,
            'title': 'View Dashboard',
            'description': 'Explore KPIs, charts, and executive insights',
            'action': 'dashboard',
            'completed': current_state['has_data'],
            'icon': 'bar-chart',
            'requires': 'generate-sample'
        },
        {
            'step': 5,
            'title': 'Assess Patient',
            'description': 'Enter patient details and see instant risk prediction',
            'action': 'assessment',
            'completed': False,
            'icon': 'user-plus',
            'requires': 'train'
        },
        {
            'step': 6,
            'title': 'Generate Predictions',
            'description': 'Batch predict risk for all patients with segmentation',
            'action': 'predictions',
            'completed': current_state.get('has_predictions', False),
            'icon': 'target',
            'requires': 'train'
        },
        {
            'step': 7,
            'title': 'Export Results',
            'description': 'Download predictions as CSV for further analysis',
            'action': 'export',
            'completed': False,
            'icon': 'download',
            'requires': 'predictions'
        }
    ]
    
    return {
        'success': True,
        'current_state': current_state,
        'steps': steps,
        'next_step': next((s for s in steps if not s['completed']), steps[-1])
    }

# Include the router
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
