"""
Model Manager Module
Handles model persistence, training pipeline, and prediction caching
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler
import pickle
import hashlib
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ModelManager:
    """Manages ML models with persistence and caching"""
    
    MODEL_DIR = Path('/app/backend/models')
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.metadata: Dict[str, Any] = {}
        self.is_trained = False
        self.feature_columns: List[str] = []
        self.target_column: str = ''
        self._ensure_model_dir()
    
    def _ensure_model_dir(self):
        """Ensure model directory exists"""
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    def _get_data_hash(self, df: pd.DataFrame, target: str) -> str:
        """Generate hash of dataset for caching"""
        data_str = f"{df.shape}_{target}_{sorted(df.columns.tolist())}"
        return hashlib.md5(data_str.encode()).hexdigest()[:12]
    
    def train_pipeline(self,
                       df: pd.DataFrame,
                       target_column: str,
                       feature_columns: List[str],
                       test_size: float = 0.2,
                       random_state: int = 42) -> Dict[str, Any]:
        """
        Complete training pipeline with automatic model selection
        
        Returns:
            Dictionary with training results and metrics
        """
        self.target_column = target_column
        self.feature_columns = feature_columns
        
        # Prepare data
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Handle any remaining NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        self.scalers['main'] = StandardScaler()
        X_scaled = self.scalers['main'].fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train models
        results = {}
        
        # 1. Logistic Regression
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced',
            solver='lbfgs'
        )
        lr_model.fit(X_train, y_train)
        self.models['logistic_regression'] = lr_model
        results['logistic_regression'] = self._evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
        
        # 2. Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        results['random_forest'] = self._evaluate_model(rf_model, X_test, y_test, 'Random Forest')
        
        # 3. Gradient Boosting (optional advanced model)
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=random_state,
            learning_rate=0.1
        )
        gb_model.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb_model
        results['gradient_boosting'] = self._evaluate_model(gb_model, X_test, y_test, 'Gradient Boosting')
        
        # Determine best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        
        # Store metadata
        self.metadata = {
            'training_date': datetime.now().isoformat(),
            'n_samples': len(df),
            'n_features': len(feature_columns),
            'feature_columns': feature_columns,
            'target_column': target_column,
            'test_size': test_size,
            'best_model': best_model_name,
            'model_metrics': {k: {
                'accuracy': v['accuracy'],
                'precision': v['precision'],
                'recall': v['recall'],
                'f1_score': v['f1_score'],
                'roc_auc': v['roc_auc']
            } for k, v in results.items()}
        }
        
        self.is_trained = True
        
        return {
            'results': results,
            'best_model': best_model_name,
            'metadata': self.metadata
        }
    
    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Evaluate a trained model"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        return {
            'model_name': model_name,
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1_score': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'roc_auc': round(roc_auc_score(y_test, y_pred_proba), 4),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
        }
    
    def predict(self, 
                df: pd.DataFrame, 
                model_name: str = 'best') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using trained model
        
        Returns:
            Tuple of (predicted_classes, predicted_probabilities)
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_pipeline first.")
        
        if model_name == 'best':
            model_name = self.metadata.get('best_model', 'random_forest')
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Prepare features
        X = df[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scalers['main'].transform(X)
        
        # Predict
        model = self.models[model_name]
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def get_feature_importance(self, model_name: str = 'best') -> pd.DataFrame:
        """Get feature importance from trained model"""
        if not self.is_trained:
            return pd.DataFrame()
        
        if model_name == 'best':
            model_name = self.metadata.get('best_model', 'random_forest')
        
        model = self.models.get(model_name)
        if model is None:
            return pd.DataFrame()
        
        # Get importances based on model type
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Add percentage
        importance_df['Importance %'] = (importance_df['Importance'] / importance_df['Importance'].sum() * 100).round(2)
        
        return importance_df
    
    def save_models(self, filename: str = 'healthcare_models.pkl') -> str:
        """Save trained models to disk"""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        filepath = self.MODEL_DIR / filename
        
        save_data = {
            'models': self.models,
            'scalers': self.scalers,
            'metadata': self.metadata,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        return str(filepath)
    
    def load_models(self, filename: str = 'healthcare_models.pkl') -> bool:
        """Load trained models from disk"""
        filepath = self.MODEL_DIR / filename
        
        if not filepath.exists():
            return False
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.models = save_data['models']
        self.scalers = save_data['scalers']
        self.metadata = save_data['metadata']
        self.feature_columns = save_data['feature_columns']
        self.target_column = save_data['target_column']
        self.is_trained = True
        
        return True
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison table of all trained models"""
        if not self.metadata or 'model_metrics' not in self.metadata:
            return pd.DataFrame()
        
        metrics = self.metadata['model_metrics']
        data = []
        
        for model_name, model_metrics in metrics.items():
            row = {'Model': model_name.replace('_', ' ').title()}
            row.update(model_metrics)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Highlight best model
        best_model = self.metadata.get('best_model', '').replace('_', ' ').title()
        
        return df


class RiskSegmenter:
    """Segment patients into risk categories"""
    
    def __init__(self, low_threshold: float = 0.3, high_threshold: float = 0.7):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def segment(self, probabilities: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Segment patients into Low, Medium, High risk categories
        
        Returns:
            Tuple of (risk_categories, segment_stats)
        """
        categories = np.empty(len(probabilities), dtype=object)
        
        categories[probabilities < self.low_threshold] = 'Low Risk'
        categories[(probabilities >= self.low_threshold) & (probabilities < self.high_threshold)] = 'Medium Risk'
        categories[probabilities >= self.high_threshold] = 'High Risk'
        
        # Calculate statistics
        total = len(categories)
        stats = {
            'Low Risk': {
                'count': int((categories == 'Low Risk').sum()),
                'percentage': round((categories == 'Low Risk').sum() / total * 100, 1),
                'avg_probability': round(probabilities[categories == 'Low Risk'].mean(), 3) if (categories == 'Low Risk').any() else 0
            },
            'Medium Risk': {
                'count': int((categories == 'Medium Risk').sum()),
                'percentage': round((categories == 'Medium Risk').sum() / total * 100, 1),
                'avg_probability': round(probabilities[categories == 'Medium Risk'].mean(), 3) if (categories == 'Medium Risk').any() else 0
            },
            'High Risk': {
                'count': int((categories == 'High Risk').sum()),
                'percentage': round((categories == 'High Risk').sum() / total * 100, 1),
                'avg_probability': round(probabilities[categories == 'High Risk'].mean(), 3) if (categories == 'High Risk').any() else 0
            }
        }
        
        return categories, stats
    
    def get_segment_summary(self, df: pd.DataFrame, categories: np.ndarray) -> pd.DataFrame:
        """Get summary statistics by risk segment"""
        df_with_risk = df.copy()
        df_with_risk['Risk_Category'] = categories
        
        # Calculate summary stats
        summary_data = []
        
        for category in ['Low Risk', 'Medium Risk', 'High Risk']:
            segment_df = df_with_risk[df_with_risk['Risk_Category'] == category]
            
            if len(segment_df) == 0:
                continue
            
            row = {
                'Risk Category': category,
                'Count': len(segment_df),
                'Percentage': f"{len(segment_df) / len(df) * 100:.1f}%"
            }
            
            # Add mean of numeric columns if available
            numeric_cols = segment_df.select_dtypes(include=[np.number]).columns[:5]
            for col in numeric_cols:
                if col != 'Risk_Category':
                    row[f'Avg {col}'] = round(segment_df[col].mean(), 2)
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
