"""
Machine Learning Models Module
Implements Logistic Regression and Random Forest for risk prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class HealthcareRiskPredictor:
    """Class for training and evaluating healthcare risk prediction models"""
    
    def __init__(self):
        self.logistic_model = None
        self.random_forest_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
    def prepare_data(self, 
                     df: pd.DataFrame, 
                     target_column: str,
                     feature_columns: List[str],
                     test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            feature_columns: List of feature column names
            test_size: Proportion of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        df_prep = df.copy()
        self.feature_names = feature_columns.copy()
        
        # Encode categorical features
        for col in feature_columns:
            if df_prep[col].dtype == 'object':
                le = LabelEncoder()
                df_prep[col] = le.fit_transform(df_prep[col].astype(str))
                self.label_encoders[col] = le
        
        X = df_prep[feature_columns].values
        y = df_prep[target_column].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, 
                                  X_train: np.ndarray, 
                                  y_train: np.ndarray,
                                  **kwargs) -> LogisticRegression:
        """Train Logistic Regression model"""
        self.logistic_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            **kwargs
        )
        self.logistic_model.fit(X_train, y_train)
        return self.logistic_model
    
    def train_random_forest(self, 
                            X_train: np.ndarray, 
                            y_train: np.ndarray,
                            n_estimators: int = 100,
                            max_depth: Optional[int] = 10,
                            **kwargs) -> RandomForestClassifier:
        """Train Random Forest model"""
        self.random_forest_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            **kwargs
        )
        self.random_forest_model.fit(X_train, y_train)
        return self.random_forest_model
    
    def evaluate_model(self, 
                       model, 
                       X_test: np.ndarray, 
                       y_test: np.ndarray,
                       model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluate a trained model
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1_score': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'roc_auc': round(roc_auc_score(y_test, y_pred_proba), 4),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # ROC curve data
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
        
        # Precision-Recall curve data
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        metrics['pr_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist()
        }
        
        return metrics
    
    def get_feature_importance(self, model_type: str = 'random_forest') -> pd.DataFrame:
        """Get feature importance from trained model"""
        if model_type == 'random_forest' and self.random_forest_model is not None:
            importances = self.random_forest_model.feature_importances_
        elif model_type == 'logistic' and self.logistic_model is not None:
            importances = np.abs(self.logistic_model.coef_[0])
        else:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def cross_validate(self, 
                       model, 
                       X: np.ndarray, 
                       y: np.ndarray, 
                       cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        return {
            'mean_cv_score': round(scores.mean(), 4),
            'std_cv_score': round(scores.std(), 4),
            'cv_scores': scores.tolist()
        }
    
    def predict_risk(self, 
                     df: pd.DataFrame, 
                     model_type: str = 'random_forest') -> np.ndarray:
        """
        Predict risk scores for new data
        
        Args:
            df: DataFrame with patient data
            model_type: 'logistic' or 'random_forest'
            
        Returns:
            Array of risk probabilities
        """
        df_prep = df.copy()
        
        # Encode categorical features
        for col in self.feature_names:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories
                df_prep[col] = df_prep[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        X = df_prep[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        if model_type == 'random_forest' and self.random_forest_model is not None:
            return self.random_forest_model.predict_proba(X_scaled)[:, 1]
        elif model_type == 'logistic' and self.logistic_model is not None:
            return self.logistic_model.predict_proba(X_scaled)[:, 1]
        else:
            raise ValueError(f"Model {model_type} not trained")
    
    def compare_models(self, 
                       X_train: np.ndarray, 
                       X_test: np.ndarray, 
                       y_train: np.ndarray, 
                       y_test: np.ndarray) -> pd.DataFrame:
        """
        Train and compare both models
        
        Returns:
            DataFrame with comparison metrics
        """
        # Train both models
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.is_trained = True
        
        # Evaluate both models
        lr_metrics = self.evaluate_model(self.logistic_model, X_test, y_test, "Logistic Regression")
        rf_metrics = self.evaluate_model(self.random_forest_model, X_test, y_test, "Random Forest")
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            'Logistic Regression': [
                lr_metrics['accuracy'],
                lr_metrics['precision'],
                lr_metrics['recall'],
                lr_metrics['f1_score'],
                lr_metrics['roc_auc']
            ],
            'Random Forest': [
                rf_metrics['accuracy'],
                rf_metrics['precision'],
                rf_metrics['recall'],
                rf_metrics['f1_score'],
                rf_metrics['roc_auc']
            ]
        })
        
        return comparison, lr_metrics, rf_metrics


def get_default_feature_columns() -> List[str]:
    """Return default feature columns for prediction"""
    return [
        'age', 'comorbidities_count', 'length_of_stay', 'previous_admissions',
        'num_medications', 'hemoglobin', 'creatinine', 'glucose',
        'blood_pressure_systolic', 'blood_pressure_diastolic', 'bmi',
        'emergency_admission', 'icu_stay'
    ]


def get_available_targets() -> List[str]:
    """Return available target columns for prediction"""
    return ['readmitted_30_days', 'complication_occurred', 'risk_score']
