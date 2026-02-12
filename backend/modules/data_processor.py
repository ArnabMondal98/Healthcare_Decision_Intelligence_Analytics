"""
Data Processing Module
Handles data loading, cleaning, and preprocessing for analysis
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import io


def load_csv_data(uploaded_file) -> pd.DataFrame:
    """Load CSV data from uploaded file"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {str(e)}")


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive summary statistics of the dataset"""
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    return summary


def clean_data(df: pd.DataFrame, 
               handle_missing: str = 'drop',
               fill_numeric: str = 'mean',
               fill_categorical: str = 'mode') -> pd.DataFrame:
    """
    Clean the dataset by handling missing values
    
    Args:
        df: Input DataFrame
        handle_missing: 'drop' to remove rows, 'fill' to impute values
        fill_numeric: Method for numeric columns ('mean', 'median', 'zero')
        fill_categorical: Method for categorical columns ('mode', 'unknown')
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if handle_missing == 'drop':
        df_clean = df_clean.dropna()
    elif handle_missing == 'fill':
        # Handle numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                if fill_numeric == 'mean':
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif fill_numeric == 'median':
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif fill_numeric == 'zero':
                    df_clean[col].fillna(0, inplace=True)
        
        # Handle categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                if fill_categorical == 'mode':
                    mode_val = df_clean[col].mode()
                    if len(mode_val) > 0:
                        df_clean[col].fillna(mode_val[0], inplace=True)
                elif fill_categorical == 'unknown':
                    df_clean[col].fillna('Unknown', inplace=True)
    
    return df_clean


def prepare_features_for_ml(df: pd.DataFrame, 
                            target_column: str,
                            feature_columns: Optional[List[str]] = None,
                            scale_features: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], Any]:
    """
    Prepare features for machine learning
    
    Args:
        df: Input DataFrame
        target_column: Name of the target variable column
        feature_columns: List of feature columns to use (None = all numeric)
        scale_features: Whether to standardize features
    
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        scaler: Fitted scaler (or None if not scaling)
    """
    df_prep = df.copy()
    
    # Select feature columns
    if feature_columns is None:
        feature_columns = df_prep.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in feature_columns:
            feature_columns.remove(target_column)
    
    # Encode categorical features if present
    label_encoders = {}
    for col in feature_columns:
        if df_prep[col].dtype == 'object':
            le = LabelEncoder()
            df_prep[col] = le.fit_transform(df_prep[col].astype(str))
            label_encoders[col] = le
    
    X = df_prep[feature_columns].values
    y = df_prep[target_column].values
    
    # Scale features
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y, feature_columns, scaler


def calculate_demographic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate demographic statistics from the dataset"""
    stats = {}
    
    # Age statistics
    if 'age' in df.columns:
        stats['age'] = {
            'mean': df['age'].mean(),
            'median': df['age'].median(),
            'std': df['age'].std(),
            'min': df['age'].min(),
            'max': df['age'].max(),
            'distribution': df['age'].value_counts(bins=10).to_dict()
        }
    
    # Gender distribution
    if 'gender' in df.columns:
        stats['gender'] = df['gender'].value_counts().to_dict()
    
    # Ethnicity distribution
    if 'ethnicity' in df.columns:
        stats['ethnicity'] = df['ethnicity'].value_counts().to_dict()
    
    # Insurance distribution
    if 'insurance_type' in df.columns:
        stats['insurance'] = df['insurance_type'].value_counts().to_dict()
    
    return stats


def calculate_disease_prevalence(df: pd.DataFrame, diagnosis_column: str = 'primary_diagnosis') -> pd.DataFrame:
    """Calculate disease prevalence statistics"""
    if diagnosis_column not in df.columns:
        return pd.DataFrame()
    
    prevalence = df[diagnosis_column].value_counts().reset_index()
    prevalence.columns = ['Diagnosis', 'Count']
    prevalence['Percentage'] = (prevalence['Count'] / len(df) * 100).round(2)
    
    return prevalence


def calculate_risk_factors(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze risk factors in the dataset"""
    risk_analysis = {}
    
    # Readmission rates by various factors
    if 'readmitted_30_days' in df.columns:
        overall_rate = df['readmitted_30_days'].mean() * 100
        risk_analysis['overall_readmission_rate'] = round(overall_rate, 2)
        
        # By diagnosis
        if 'primary_diagnosis' in df.columns:
            risk_analysis['readmission_by_diagnosis'] = df.groupby('primary_diagnosis')['readmitted_30_days'].mean().mul(100).round(2).to_dict()
        
        # By age group
        if 'age' in df.columns:
            df_temp = df.copy()
            df_temp['age_group'] = pd.cut(df_temp['age'], bins=[0, 30, 50, 65, 80, 100], labels=['18-30', '31-50', '51-65', '66-80', '80+'])
            risk_analysis['readmission_by_age_group'] = df_temp.groupby('age_group')['readmitted_30_days'].mean().mul(100).round(2).to_dict()
    
    # Complication rates
    if 'complication_occurred' in df.columns:
        risk_analysis['overall_complication_rate'] = round(df['complication_occurred'].mean() * 100, 2)
    
    return risk_analysis


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for numeric columns"""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()


def export_to_csv(df: pd.DataFrame) -> bytes:
    """Export DataFrame to CSV bytes for download"""
    return df.to_csv(index=False).encode('utf-8')
