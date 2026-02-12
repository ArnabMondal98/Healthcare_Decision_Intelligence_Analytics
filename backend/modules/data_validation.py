"""
Data Validation Module
Automatically detects dataset schema, identifies target columns, and validates data quality
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ColumnType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    ID = "identifier"
    UNKNOWN = "unknown"


class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationMessage:
    severity: ValidationSeverity
    column: Optional[str]
    message: str
    suggestion: Optional[str] = None


@dataclass
class ColumnSchema:
    name: str
    dtype: str
    column_type: ColumnType
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    is_target_candidate: bool
    sample_values: List[Any]


class DataValidator:
    """Comprehensive data validation for healthcare datasets"""
    
    # Common healthcare target columns
    TARGET_KEYWORDS = [
        'readmit', 'readmission', 'readmitted', 'outcome', 'target',
        'mortality', 'death', 'expired', 'complication', 'risk',
        'label', 'class', 'result', 'event', 'flag'
    ]
    
    # Common ID column patterns
    ID_KEYWORDS = ['id', 'key', 'code', 'number', 'no', 'num', 'index']
    
    # Expected healthcare columns
    EXPECTED_COLUMNS = {
        'demographics': ['age', 'gender', 'sex', 'ethnicity', 'race'],
        'clinical': ['diagnosis', 'comorbidity', 'medication', 'procedure'],
        'vitals': ['blood_pressure', 'heart_rate', 'temperature', 'bmi', 'weight', 'height'],
        'labs': ['hemoglobin', 'creatinine', 'glucose', 'cholesterol', 'hba1c'],
        'admission': ['admission', 'discharge', 'length_of_stay', 'los', 'icu', 'emergency']
    }
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.schema: Dict[str, ColumnSchema] = {}
        self.messages: List[ValidationMessage] = []
        self.target_candidates: List[str] = []
        self._analyze_schema()
    
    def _analyze_schema(self):
        """Analyze the schema of each column"""
        for col in self.df.columns:
            col_type = self._detect_column_type(col)
            is_target = self._is_target_candidate(col)
            
            schema = ColumnSchema(
                name=col,
                dtype=str(self.df[col].dtype),
                column_type=col_type,
                missing_count=int(self.df[col].isnull().sum()),
                missing_percentage=round(self.df[col].isnull().sum() / len(self.df) * 100, 2),
                unique_count=int(self.df[col].nunique()),
                unique_percentage=round(self.df[col].nunique() / len(self.df) * 100, 2),
                is_target_candidate=is_target,
                sample_values=self.df[col].dropna().head(5).tolist()
            )
            
            self.schema[col] = schema
            
            if is_target:
                self.target_candidates.append(col)
    
    def _detect_column_type(self, col: str) -> ColumnType:
        """Detect the semantic type of a column"""
        series = self.df[col]
        col_lower = col.lower()
        
        # Check for ID columns
        if any(kw in col_lower for kw in self.ID_KEYWORDS):
            if series.nunique() == len(series):
                return ColumnType.ID
        
        # Check dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnType.DATETIME
        
        if pd.api.types.is_bool_dtype(series):
            return ColumnType.BOOLEAN
        
        if pd.api.types.is_numeric_dtype(series):
            # Binary could be boolean
            unique_vals = series.dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                return ColumnType.BOOLEAN
            return ColumnType.NUMERIC
        
        if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            unique_ratio = series.nunique() / len(series)
            # High cardinality text vs low cardinality categorical
            if unique_ratio > 0.5:
                return ColumnType.TEXT
            return ColumnType.CATEGORICAL
        
        return ColumnType.UNKNOWN
    
    def _is_target_candidate(self, col: str) -> bool:
        """Check if column could be a prediction target"""
        col_lower = col.lower()
        
        # Check keywords
        if any(kw in col_lower for kw in self.TARGET_KEYWORDS):
            return True
        
        # Check if binary
        series = self.df[col]
        if pd.api.types.is_numeric_dtype(series):
            unique_vals = series.dropna().unique()
            if len(unique_vals) == 2:
                return True
        
        return False
    
    def validate(self) -> List[ValidationMessage]:
        """Run all validation checks"""
        self.messages = []
        
        self._check_row_count()
        self._check_missing_values()
        self._check_column_types()
        self._check_target_columns()
        self._check_expected_columns()
        self._check_data_quality()
        
        return self.messages
    
    def _check_row_count(self):
        """Check if dataset has sufficient rows"""
        row_count = len(self.df)
        
        if row_count < 50:
            self.messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                column=None,
                message=f"Dataset has only {row_count} rows. Minimum 50 recommended for ML.",
                suggestion="Upload a larger dataset for meaningful predictions."
            ))
        elif row_count < 200:
            self.messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                column=None,
                message=f"Dataset has {row_count} rows. Consider more data for better model performance.",
                suggestion="500+ rows recommended for robust predictions."
            ))
        else:
            self.messages.append(ValidationMessage(
                severity=ValidationSeverity.INFO,
                column=None,
                message=f"Dataset has {row_count} rows. Sufficient for training."
            ))
    
    def _check_missing_values(self):
        """Check for missing values"""
        for col, schema in self.schema.items():
            if schema.missing_percentage > 50:
                self.messages.append(ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    column=col,
                    message=f"Column '{col}' has {schema.missing_percentage}% missing values.",
                    suggestion="Consider dropping this column or imputing values."
                ))
            elif schema.missing_percentage > 20:
                self.messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    column=col,
                    message=f"Column '{col}' has {schema.missing_percentage}% missing values.",
                    suggestion="Missing values will be imputed during preprocessing."
                ))
    
    def _check_column_types(self):
        """Check for problematic column types"""
        for col, schema in self.schema.items():
            if schema.column_type == ColumnType.TEXT:
                self.messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    column=col,
                    message=f"Column '{col}' appears to be free text with high cardinality.",
                    suggestion="This column may be excluded from ML features."
                ))
            elif schema.column_type == ColumnType.ID:
                self.messages.append(ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    column=col,
                    message=f"Column '{col}' detected as identifier.",
                    suggestion="Will be excluded from ML features."
                ))
    
    def _check_target_columns(self):
        """Check for valid target columns"""
        if not self.target_candidates:
            self.messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                column=None,
                message="No obvious target column detected for prediction.",
                suggestion="Please manually select a target column (binary outcome preferred)."
            ))
        else:
            self.messages.append(ValidationMessage(
                severity=ValidationSeverity.INFO,
                column=None,
                message=f"Detected {len(self.target_candidates)} potential target column(s): {', '.join(self.target_candidates)}"
            ))
    
    def _check_expected_columns(self):
        """Check for expected healthcare columns"""
        found_categories = []
        for category, keywords in self.EXPECTED_COLUMNS.items():
            for col in self.df.columns:
                col_lower = col.lower()
                if any(kw in col_lower for kw in keywords):
                    found_categories.append(category)
                    break
        
        missing_categories = set(self.EXPECTED_COLUMNS.keys()) - set(found_categories)
        if missing_categories:
            self.messages.append(ValidationMessage(
                severity=ValidationSeverity.INFO,
                column=None,
                message=f"Optional column categories not found: {', '.join(missing_categories)}",
                suggestion="These columns could enhance prediction accuracy."
            ))
    
    def _check_data_quality(self):
        """Check for data quality issues"""
        # Check for constant columns
        for col, schema in self.schema.items():
            if schema.unique_count == 1:
                self.messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    column=col,
                    message=f"Column '{col}' has only one unique value.",
                    suggestion="This column provides no predictive value."
                ))
        
        # Check for highly imbalanced targets
        for target in self.target_candidates:
            if self.schema[target].column_type in [ColumnType.BOOLEAN, ColumnType.CATEGORICAL]:
                value_counts = self.df[target].value_counts(normalize=True)
                if len(value_counts) >= 2:
                    imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[1]
                    if imbalance_ratio > 10:
                        self.messages.append(ValidationMessage(
                            severity=ValidationSeverity.WARNING,
                            column=target,
                            message=f"Target '{target}' is highly imbalanced (ratio: {imbalance_ratio:.1f}:1).",
                            suggestion="Class balancing will be applied during training."
                        ))
    
    def get_schema_summary(self) -> pd.DataFrame:
        """Get schema summary as DataFrame"""
        data = []
        for col, schema in self.schema.items():
            data.append({
                'Column': schema.name,
                'Type': schema.column_type.value,
                'Data Type': schema.dtype,
                'Missing %': schema.missing_percentage,
                'Unique': schema.unique_count,
                'Target Candidate': '✓' if schema.is_target_candidate else ''
            })
        return pd.DataFrame(data)
    
    def get_recommended_target(self) -> Optional[str]:
        """Get the most likely target column"""
        # Priority: readmission > outcome > complication > other binary
        priority_keywords = ['readmit', 'outcome', 'complication', 'mortality']
        
        for keyword in priority_keywords:
            for candidate in self.target_candidates:
                if keyword in candidate.lower():
                    return candidate
        
        # Return first binary candidate
        for candidate in self.target_candidates:
            if self.schema[candidate].column_type == ColumnType.BOOLEAN:
                return candidate
        
        return self.target_candidates[0] if self.target_candidates else None
    
    def get_feature_columns(self, exclude_target: str = None) -> List[str]:
        """Get recommended feature columns"""
        features = []
        excluded_types = [ColumnType.ID, ColumnType.TEXT, ColumnType.DATETIME]
        
        for col, schema in self.schema.items():
            if col == exclude_target:
                continue
            if schema.column_type in excluded_types:
                continue
            if schema.missing_percentage > 50:
                continue
            if schema.unique_count == 1:
                continue
            features.append(col)
        
        return features
