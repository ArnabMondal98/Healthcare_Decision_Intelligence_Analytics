"""
Feature Engineering Module
Automatic feature engineering for healthcare datasets
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FeatureInfo:
    name: str
    original_columns: List[str]
    feature_type: str  # 'original', 'encoded', 'derived', 'interaction'
    description: str


class FeatureEngineer:
    """Automatic feature engineering for healthcare data"""
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_info: List[FeatureInfo] = []
        self.original_columns: List[str] = []
        self.processed_columns: List[str] = []
        self._is_fitted = False
    
    def fit_transform(self, 
                      df: pd.DataFrame, 
                      target_column: Optional[str] = None,
                      create_derived: bool = True) -> pd.DataFrame:
        """
        Fit and transform the dataset with automatic feature engineering
        
        Args:
            df: Input DataFrame
            target_column: Target column to exclude from features
            create_derived: Whether to create derived health indicators
            
        Returns:
            Transformed DataFrame with engineered features
        """
        self.original_columns = df.columns.tolist()
        df_processed = df.copy()
        
        # 1. Handle missing values
        df_processed = self._handle_missing_values(df_processed, target_column)
        
        # 2. Encode categorical variables
        df_processed = self._encode_categoricals(df_processed, target_column)
        
        # 3. Create derived features if enabled
        if create_derived:
            df_processed = self._create_derived_features(df_processed, target_column)
        
        # 4. Create interaction features
        df_processed = self._create_interaction_features(df_processed, target_column)
        
        self.processed_columns = [c for c in df_processed.columns if c != target_column]
        self._is_fitted = True
        
        return df_processed
    
    def transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Transform new data using fitted encoders"""
        if not self._is_fitted:
            raise ValueError("FeatureEngineer must be fitted first")
        
        df_processed = df.copy()
        
        # Apply same transformations
        df_processed = self._handle_missing_values(df_processed, target_column)
        df_processed = self._apply_encoders(df_processed, target_column)
        df_processed = self._create_derived_features(df_processed, target_column)
        df_processed = self._create_interaction_features(df_processed, target_column)
        
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
        """Handle missing values intelligently"""
        for col in df.columns:
            if col == target_column:
                continue
                
            if df[col].isnull().sum() == 0:
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Use median for numeric columns (robust to outliers)
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                # Use mode for categorical columns
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna('Unknown')
        
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col == target_column:
                continue
            
            # Use Label Encoding for ordinal/binary, One-hot for nominal with few categories
            n_unique = df[col].nunique()
            
            if n_unique <= 2:
                # Binary encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                
                self.feature_info.append(FeatureInfo(
                    name=col,
                    original_columns=[col],
                    feature_type='encoded',
                    description=f'Label encoded from {n_unique} categories'
                ))
            elif n_unique <= 10:
                # One-hot encoding for manageable cardinality
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                
                self.feature_info.append(FeatureInfo(
                    name=col,
                    original_columns=[col],
                    feature_type='encoded',
                    description=f'Label encoded from {n_unique} categories'
                ))
            else:
                # High cardinality - use label encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                
                self.feature_info.append(FeatureInfo(
                    name=col,
                    original_columns=[col],
                    feature_type='encoded',
                    description=f'Label encoded (high cardinality: {n_unique})'
                ))
        
        return df
    
    def _apply_encoders(self, df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
        """Apply fitted encoders to new data"""
        for col, le in self.label_encoders.items():
            if col in df.columns and col != target_column:
                # Handle unseen categories
                df[col] = df[col].apply(
                    lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                )
        return df
    
    def _create_derived_features(self, df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
        """Create derived health indicators"""
        columns_lower = {c.lower(): c for c in df.columns}
        
        # Age groups
        if 'age' in columns_lower:
            age_col = columns_lower['age']
            if pd.api.types.is_numeric_dtype(df[age_col]):
                df['age_group'] = pd.cut(
                    df[age_col], 
                    bins=[0, 30, 45, 60, 75, 100],
                    labels=[0, 1, 2, 3, 4]
                ).astype(float).fillna(2)
                
                df['is_elderly'] = (df[age_col] >= 65).astype(int)
                
                self.feature_info.append(FeatureInfo(
                    name='age_group',
                    original_columns=[age_col],
                    feature_type='derived',
                    description='Age grouped into 5 categories'
                ))
                self.feature_info.append(FeatureInfo(
                    name='is_elderly',
                    original_columns=[age_col],
                    feature_type='derived',
                    description='Binary indicator for age >= 65'
                ))
        
        # BMI categories
        if 'bmi' in columns_lower:
            bmi_col = columns_lower['bmi']
            if pd.api.types.is_numeric_dtype(df[bmi_col]):
                df['bmi_category'] = pd.cut(
                    df[bmi_col],
                    bins=[0, 18.5, 25, 30, 100],
                    labels=[0, 1, 2, 3]  # underweight, normal, overweight, obese
                ).astype(float).fillna(1)
                
                df['is_obese'] = (df[bmi_col] >= 30).astype(int)
                
                self.feature_info.append(FeatureInfo(
                    name='bmi_category',
                    original_columns=[bmi_col],
                    feature_type='derived',
                    description='BMI category (0=underweight, 1=normal, 2=overweight, 3=obese)'
                ))
        
        # Blood pressure category
        bp_sys_cols = [c for c in columns_lower if 'systolic' in c or 'bp_sys' in c or 'blood_pressure_s' in c]
        if bp_sys_cols:
            bp_col = columns_lower.get(bp_sys_cols[0], bp_sys_cols[0])
            if bp_col in df.columns and pd.api.types.is_numeric_dtype(df[bp_col]):
                df['hypertension'] = (df[bp_col] >= 140).astype(int)
                
                self.feature_info.append(FeatureInfo(
                    name='hypertension',
                    original_columns=[bp_col],
                    feature_type='derived',
                    description='Hypertension indicator (systolic BP >= 140)'
                ))
        
        # Glucose category
        if 'glucose' in columns_lower:
            glucose_col = columns_lower['glucose']
            if pd.api.types.is_numeric_dtype(df[glucose_col]):
                df['high_glucose'] = (df[glucose_col] >= 126).astype(int)
                
                self.feature_info.append(FeatureInfo(
                    name='high_glucose',
                    original_columns=[glucose_col],
                    feature_type='derived',
                    description='High glucose indicator (>= 126 mg/dL)'
                ))
        
        # Creatinine category (kidney function)
        if 'creatinine' in columns_lower:
            creat_col = columns_lower['creatinine']
            if pd.api.types.is_numeric_dtype(df[creat_col]):
                df['kidney_risk'] = (df[creat_col] >= 1.5).astype(int)
                
                self.feature_info.append(FeatureInfo(
                    name='kidney_risk',
                    original_columns=[creat_col],
                    feature_type='derived',
                    description='Kidney risk indicator (creatinine >= 1.5)'
                ))
        
        # Length of stay categories
        los_cols = [c for c in columns_lower if 'length' in c or 'los' in c or 'stay' in c]
        if los_cols:
            los_col = columns_lower.get(los_cols[0], los_cols[0])
            if los_col in df.columns and pd.api.types.is_numeric_dtype(df[los_col]):
                df['extended_stay'] = (df[los_col] >= 7).astype(int)
                
                self.feature_info.append(FeatureInfo(
                    name='extended_stay',
                    original_columns=[los_col],
                    feature_type='derived',
                    description='Extended hospital stay (>= 7 days)'
                ))
        
        # Comorbidity severity
        comorbid_cols = [c for c in columns_lower if 'comorbid' in c]
        if comorbid_cols:
            comorbid_col = columns_lower.get(comorbid_cols[0], comorbid_cols[0])
            if comorbid_col in df.columns and pd.api.types.is_numeric_dtype(df[comorbid_col]):
                df['high_comorbidity'] = (df[comorbid_col] >= 3).astype(int)
                
                self.feature_info.append(FeatureInfo(
                    name='high_comorbidity',
                    original_columns=[comorbid_col],
                    feature_type='derived',
                    description='High comorbidity count (>= 3)'
                ))
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
        """Create interaction features between important variables"""
        columns_lower = {c.lower(): c for c in df.columns}
        
        # Age x Comorbidity interaction
        if 'age' in columns_lower and 'comorbidities_count' in columns_lower:
            age_col = columns_lower['age']
            comorbid_col = columns_lower['comorbidities_count']
            if pd.api.types.is_numeric_dtype(df[age_col]) and pd.api.types.is_numeric_dtype(df[comorbid_col]):
                # Normalize before interaction
                age_norm = (df[age_col] - df[age_col].min()) / (df[age_col].max() - df[age_col].min() + 1e-10)
                comorbid_norm = (df[comorbid_col] - df[comorbid_col].min()) / (df[comorbid_col].max() - df[comorbid_col].min() + 1e-10)
                df['age_comorbidity_score'] = (age_norm * comorbid_norm).round(4)
                
                self.feature_info.append(FeatureInfo(
                    name='age_comorbidity_score',
                    original_columns=[age_col, comorbid_col],
                    feature_type='interaction',
                    description='Interaction between age and comorbidity count'
                ))
        
        # Emergency x ICU interaction
        if 'emergency_admission' in columns_lower and 'icu_stay' in columns_lower:
            em_col = columns_lower['emergency_admission']
            icu_col = columns_lower['icu_stay']
            if pd.api.types.is_numeric_dtype(df[em_col]) and pd.api.types.is_numeric_dtype(df[icu_col]):
                df['emergency_icu'] = (df[em_col] * df[icu_col]).astype(int)
                
                self.feature_info.append(FeatureInfo(
                    name='emergency_icu',
                    original_columns=[em_col, icu_col],
                    feature_type='interaction',
                    description='Emergency admission with ICU stay'
                ))
        
        # Previous admissions x LOS
        prev_cols = [c for c in columns_lower if 'previous' in c or 'prior' in c]
        los_cols = [c for c in columns_lower if 'length' in c or 'los' in c or 'stay' in c]
        if prev_cols and los_cols:
            prev_col = columns_lower.get(prev_cols[0], prev_cols[0])
            los_col = columns_lower.get(los_cols[0], los_cols[0])
            if prev_col in df.columns and los_col in df.columns:
                if pd.api.types.is_numeric_dtype(df[prev_col]) and pd.api.types.is_numeric_dtype(df[los_col]):
                    # Create utilization score
                    prev_norm = (df[prev_col] - df[prev_col].min()) / (df[prev_col].max() - df[prev_col].min() + 1e-10)
                    los_norm = (df[los_col] - df[los_col].min()) / (df[los_col].max() - df[los_col].min() + 1e-10)
                    df['utilization_score'] = (prev_norm + los_norm).round(4)
                    
                    self.feature_info.append(FeatureInfo(
                        name='utilization_score',
                        original_columns=[prev_col, los_col],
                        feature_type='interaction',
                        description='Healthcare utilization score'
                    ))
        
        return df
    
    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary of all engineered features"""
        if not self.feature_info:
            return pd.DataFrame()
        
        data = []
        for info in self.feature_info:
            data.append({
                'Feature': info.name,
                'Type': info.feature_type,
                'Source Columns': ', '.join(info.original_columns),
                'Description': info.description
            })
        
        return pd.DataFrame(data)
    
    def scale_features(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, StandardScaler]:
        """Scale numeric features for ML"""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(df[feature_columns])
        return X_scaled, self.scaler
    
    def get_numeric_features(self, df: pd.DataFrame, exclude_columns: List[str] = None) -> List[str]:
        """Get list of numeric feature columns"""
        exclude = exclude_columns or []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude]
