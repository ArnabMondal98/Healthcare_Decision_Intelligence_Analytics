"""
Visualization Module
Creates various charts and plots for healthcare data analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def create_age_distribution(df: pd.DataFrame, column: str = 'age') -> go.Figure:
    """Create age distribution histogram with KDE"""
    if column not in df.columns:
        return None
    
    fig = px.histogram(
        df, x=column, nbins=30,
        title='Age Distribution of Patients',
        labels={column: 'Age (years)', 'count': 'Number of Patients'},
        color_discrete_sequence=['#6366f1']
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        font=dict(size=12)
    )
    return fig


def create_gender_pie_chart(df: pd.DataFrame, column: str = 'gender') -> go.Figure:
    """Create gender distribution pie chart"""
    if column not in df.columns:
        return None
    
    gender_counts = df[column].value_counts()
    fig = px.pie(
        values=gender_counts.values,
        names=gender_counts.index,
        title='Gender Distribution',
        color_discrete_sequence=['#6366f1', '#ec4899']
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template='plotly_white', title_x=0.5)
    return fig


def create_diagnosis_bar_chart(df: pd.DataFrame, column: str = 'primary_diagnosis') -> go.Figure:
    """Create diagnosis distribution bar chart"""
    if column not in df.columns:
        return None
    
    diagnosis_counts = df[column].value_counts().head(10)
    fig = px.bar(
        x=diagnosis_counts.index,
        y=diagnosis_counts.values,
        title='Top 10 Primary Diagnoses',
        labels={'x': 'Diagnosis', 'y': 'Number of Patients'},
        color=diagnosis_counts.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        xaxis_tickangle=-45,
        showlegend=False
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create correlation matrix heatmap"""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return None
    
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        title='Feature Correlation Matrix',
        color_continuous_scale='RdBu_r',
        aspect='auto',
        text_auto='.2f'
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        width=800,
        height=700
    )
    return fig


def create_readmission_by_diagnosis(df: pd.DataFrame) -> go.Figure:
    """Create readmission rate by diagnosis chart"""
    if 'primary_diagnosis' not in df.columns or 'readmitted_30_days' not in df.columns:
        return None
    
    readmission_rates = df.groupby('primary_diagnosis')['readmitted_30_days'].agg(['mean', 'count']).reset_index()
    readmission_rates.columns = ['Diagnosis', 'Readmission Rate', 'Patient Count']
    readmission_rates['Readmission Rate'] = readmission_rates['Readmission Rate'] * 100
    readmission_rates = readmission_rates.sort_values('Readmission Rate', ascending=True)
    
    fig = px.bar(
        readmission_rates,
        y='Diagnosis',
        x='Readmission Rate',
        orientation='h',
        title='30-Day Readmission Rate by Diagnosis',
        labels={'Readmission Rate': 'Readmission Rate (%)', 'Diagnosis': 'Primary Diagnosis'},
        color='Readmission Rate',
        color_continuous_scale='Reds'
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        height=500
    )
    return fig


def create_risk_score_distribution(df: pd.DataFrame) -> go.Figure:
    """Create risk score distribution with readmission overlay"""
    if 'risk_score' not in df.columns:
        return None
    
    fig = px.histogram(
        df, x='risk_score', nbins=50,
        color='readmitted_30_days' if 'readmitted_30_days' in df.columns else None,
        title='Risk Score Distribution',
        labels={'risk_score': 'Risk Score', 'count': 'Count'},
        barmode='overlay',
        color_discrete_map={0: '#22c55e', 1: '#ef4444'}
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        legend_title='Readmitted'
    )
    return fig


def create_los_boxplot(df: pd.DataFrame) -> go.Figure:
    """Create length of stay boxplot by diagnosis"""
    if 'length_of_stay' not in df.columns or 'primary_diagnosis' not in df.columns:
        return None
    
    fig = px.box(
        df, x='primary_diagnosis', y='length_of_stay',
        title='Length of Stay by Diagnosis',
        labels={'length_of_stay': 'Length of Stay (days)', 'primary_diagnosis': 'Diagnosis'},
        color='primary_diagnosis'
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        xaxis_tickangle=-45,
        showlegend=False
    )
    return fig


def create_age_risk_scatter(df: pd.DataFrame) -> go.Figure:
    """Create scatter plot of age vs risk score"""
    if 'age' not in df.columns or 'risk_score' not in df.columns:
        return None
    
    fig = px.scatter(
        df, x='age', y='risk_score',
        color='readmitted_30_days' if 'readmitted_30_days' in df.columns else None,
        title='Age vs Risk Score',
        labels={'age': 'Age (years)', 'risk_score': 'Risk Score'},
        color_discrete_map={0: '#22c55e', 1: '#ef4444'},
        opacity=0.6
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        legend_title='Readmitted'
    )
    return fig


def create_ethnicity_insurance_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create heatmap of ethnicity vs insurance type"""
    if 'ethnicity' not in df.columns or 'insurance_type' not in df.columns:
        return None
    
    cross_tab = pd.crosstab(df['ethnicity'], df['insurance_type'])
    
    fig = px.imshow(
        cross_tab,
        title='Patient Distribution: Ethnicity vs Insurance Type',
        labels=dict(x='Insurance Type', y='Ethnicity', color='Count'),
        color_continuous_scale='Blues',
        text_auto=True
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5
    )
    return fig


def create_vitals_boxplots(df: pd.DataFrame) -> go.Figure:
    """Create boxplots for vital signs comparison"""
    vital_columns = ['hemoglobin', 'creatinine', 'glucose', 'bmi']
    available_cols = [col for col in vital_columns if col in df.columns]
    
    if not available_cols:
        return None
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=available_cols)
    
    colors = ['#6366f1', '#ec4899', '#22c55e', '#f59e0b']
    for i, col in enumerate(available_cols):
        row, col_idx = (i // 2) + 1, (i % 2) + 1
        fig.add_trace(
            go.Box(y=df[col], name=col, marker_color=colors[i]),
            row=row, col=col_idx
        )
    
    fig.update_layout(
        title='Distribution of Vital Signs and Lab Values',
        template='plotly_white',
        title_x=0.5,
        showlegend=False,
        height=500
    )
    return fig


def create_roc_curve(metrics: Dict[str, Any], model_name: str = "Model") -> go.Figure:
    """Create ROC curve from model metrics"""
    if 'roc_curve' not in metrics:
        return None
    
    fpr = metrics['roc_curve']['fpr']
    tpr = metrics['roc_curve']['tpr']
    auc = metrics['roc_auc']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc:.3f})',
        line=dict(color='#6366f1', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white',
        title_x=0.5
    )
    return fig


def create_roc_comparison(lr_metrics: Dict, rf_metrics: Dict) -> go.Figure:
    """Create ROC curve comparison for both models"""
    fig = go.Figure()
    
    # Logistic Regression
    fig.add_trace(go.Scatter(
        x=lr_metrics['roc_curve']['fpr'],
        y=lr_metrics['roc_curve']['tpr'],
        mode='lines',
        name=f"Logistic Regression (AUC = {lr_metrics['roc_auc']:.3f})",
        line=dict(color='#6366f1', width=2)
    ))
    
    # Random Forest
    fig.add_trace(go.Scatter(
        x=rf_metrics['roc_curve']['fpr'],
        y=rf_metrics['roc_curve']['tpr'],
        mode='lines',
        name=f"Random Forest (AUC = {rf_metrics['roc_auc']:.3f})",
        line=dict(color='#ec4899', width=2)
    ))
    
    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white',
        title_x=0.5
    )
    return fig


def create_confusion_matrix_plot(cm: List[List[int]], model_name: str = "Model") -> go.Figure:
    """Create confusion matrix heatmap"""
    cm_array = np.array(cm)
    labels = ['Not Readmitted', 'Readmitted']
    
    fig = px.imshow(
        cm_array,
        title=f'Confusion Matrix - {model_name}',
        labels=dict(x='Predicted', y='Actual', color='Count'),
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        text_auto=True
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5
    )
    return fig


def create_feature_importance_chart(importance_df: pd.DataFrame, model_name: str = "Model") -> go.Figure:
    """Create feature importance bar chart"""
    if importance_df.empty:
        return None
    
    fig = px.bar(
        importance_df.head(15),
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top 15 Feature Importance - {model_name}',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig


def create_time_series_admissions(df: pd.DataFrame) -> go.Figure:
    """Create time series of admissions"""
    if 'admission_date' not in df.columns:
        return None
    
    df_temp = df.copy()
    df_temp['admission_date'] = pd.to_datetime(df_temp['admission_date'])
    daily_admissions = df_temp.groupby(df_temp['admission_date'].dt.to_period('M')).size().reset_index()
    daily_admissions.columns = ['Month', 'Admissions']
    daily_admissions['Month'] = daily_admissions['Month'].astype(str)
    
    fig = px.line(
        daily_admissions,
        x='Month',
        y='Admissions',
        title='Monthly Hospital Admissions',
        markers=True
    )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        xaxis_tickangle=-45
    )
    return fig


def create_smoking_analysis(df: pd.DataFrame) -> go.Figure:
    """Create smoking status analysis chart"""
    if 'smoking_status' not in df.columns:
        return None
    
    smoking_data = df.groupby('smoking_status').agg({
        'readmitted_30_days': 'mean' if 'readmitted_30_days' in df.columns else 'count',
        'risk_score': 'mean' if 'risk_score' in df.columns else 'count'
    }).reset_index()
    
    if 'readmitted_30_days' in df.columns:
        smoking_data['readmitted_30_days'] = smoking_data['readmitted_30_days'] * 100
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Readmission Rate', 'Average Risk Score'])
    
    fig.add_trace(
        go.Bar(x=smoking_data['smoking_status'], y=smoking_data['readmitted_30_days'],
               marker_color=['#22c55e', '#f59e0b', '#ef4444']),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=smoking_data['smoking_status'], y=smoking_data['risk_score'],
               marker_color=['#22c55e', '#f59e0b', '#ef4444']),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Risk Analysis by Smoking Status',
        template='plotly_white',
        title_x=0.5,
        showlegend=False
    )
    return fig


def create_dashboard_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate KPI metrics for dashboard"""
    kpis = {
        'total_patients': len(df),
        'avg_age': round(df['age'].mean(), 1) if 'age' in df.columns else 'N/A',
        'avg_los': round(df['length_of_stay'].mean(), 1) if 'length_of_stay' in df.columns else 'N/A',
        'readmission_rate': round(df['readmitted_30_days'].mean() * 100, 1) if 'readmitted_30_days' in df.columns else 'N/A',
        'complication_rate': round(df['complication_occurred'].mean() * 100, 1) if 'complication_occurred' in df.columns else 'N/A',
        'avg_risk_score': round(df['risk_score'].mean(), 3) if 'risk_score' in df.columns else 'N/A',
        'icu_rate': round(df['icu_stay'].mean() * 100, 1) if 'icu_stay' in df.columns else 'N/A',
        'emergency_rate': round(df['emergency_admission'].mean() * 100, 1) if 'emergency_admission' in df.columns else 'N/A'
    }
    return kpis
