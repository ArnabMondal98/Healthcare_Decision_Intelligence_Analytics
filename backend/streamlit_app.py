"""
Healthcare Risk Analytics Platform - Production Ready
A comprehensive Streamlit application for healthcare data analysis and risk prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom modules
from modules.sample_data import generate_sample_dataset, get_sample_data_description
from modules.data_validation import DataValidator, ValidationSeverity
from modules.feature_engineering import FeatureEngineer
from modules.model_manager import ModelManager, RiskSegmenter
from modules.visualizations import (
    create_age_distribution, create_gender_pie_chart, create_diagnosis_bar_chart,
    create_correlation_heatmap, create_readmission_by_diagnosis, create_risk_score_distribution,
    create_los_boxplot, create_age_risk_scatter, create_ethnicity_insurance_heatmap,
    create_vitals_boxplots, create_roc_comparison, create_confusion_matrix_plot,
    create_feature_importance_chart, create_time_series_admissions, 
    create_smoking_analysis, create_dashboard_kpis
)

# Page configuration
st.set_page_config(
    page_title="Healthcare Risk Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for production-ready styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --success: #22c55e;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border: #334155;
    }
    
    /* Main styling */
    .main {
        background-color: var(--bg-dark);
        font-family: 'Manrope', sans-serif;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, var(--bg-dark) 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border);
    }
    
    .main-header h1 {
        color: var(--text-primary);
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .main-header p {
        color: var(--text-secondary);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-dark) 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
        border-color: var(--primary);
    }
    
    .kpi-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0;
    }
    
    .kpi-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Risk segment cards */
    .risk-low { border-left: 4px solid var(--success); }
    .risk-medium { border-left: 4px solid var(--warning); }
    .risk-high { border-left: 4px solid var(--danger); }
    
    .risk-value-low { color: var(--success) !important; }
    .risk-value-medium { color: var(--warning) !important; }
    .risk-value-high { color: var(--danger) !important; }
    
    /* Section headers */
    .section-header {
        color: var(--text-primary);
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Validation messages */
    .validation-error {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid var(--danger);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: var(--danger);
    }
    
    .validation-warning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid var(--warning);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: var(--warning);
    }
    
    .validation-info {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid var(--primary);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: var(--primary);
    }
    
    /* Feature cards */
    .feature-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .feature-name {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .feature-type {
        color: var(--primary);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-family: 'Manrope', sans-serif;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-card);
        padding: 4px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 10px 20px;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary);
        color: white;
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-card);
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Success/Error/Warning boxes */
    .stSuccess {
        background-color: rgba(34, 197, 94, 0.1);
        border: 1px solid var(--success);
        border-radius: 8px;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid var(--danger);
        border-radius: 8px;
    }
    
    .stWarning {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid var(--warning);
        border-radius: 8px;
    }
    
    /* Model card */
    .model-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .model-card.best {
        border-color: var(--primary);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);
    }
    
    .model-name {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .model-metric {
        color: var(--primary);
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .model-label {
        color: var(--text-secondary);
        font-size: 0.75rem;
        text-transform: uppercase;
    }
    
    /* Progress bar */
    .progress-bar {
        background: var(--bg-dark);
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary) 0%, #8b5cf6 100%);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: var(--primary);
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'data': None,
        'data_validated': False,
        'validator': None,
        'feature_engineer': None,
        'model_manager': None,
        'risk_segmenter': None,
        'predictions': None,
        'risk_categories': None,
        'selected_target': None,
        'selected_features': [],
        'auto_trained': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Healthcare Risk Analytics Platform</h1>
        <p>Production-ready patient risk prediction and healthcare data analysis</p>
    </div>
    """, unsafe_allow_html=True)


def render_validation_messages(messages):
    """Render validation messages with appropriate styling"""
    for msg in messages:
        if msg.severity == ValidationSeverity.ERROR:
            st.markdown(f"""
            <div class="validation-error">
                <strong>⛔ Error:</strong> {msg.message}
                {f'<br><em>💡 {msg.suggestion}</em>' if msg.suggestion else ''}
            </div>
            """, unsafe_allow_html=True)
        elif msg.severity == ValidationSeverity.WARNING:
            st.markdown(f"""
            <div class="validation-warning">
                <strong>⚠️ Warning:</strong> {msg.message}
                {f'<br><em>💡 {msg.suggestion}</em>' if msg.suggestion else ''}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="validation-info">
                <strong>ℹ️ Info:</strong> {msg.message}
            </div>
            """, unsafe_allow_html=True)


def render_risk_segment_cards(stats):
    """Render risk segmentation cards"""
    cols = st.columns(3)
    
    segments = [
        ('Low Risk', 'risk-low', 'risk-value-low', '✅'),
        ('Medium Risk', 'risk-medium', 'risk-value-medium', '⚠️'),
        ('High Risk', 'risk-high', 'risk-value-high', '🚨')
    ]
    
    for col, (label, card_class, value_class, icon) in zip(cols, segments):
        with col:
            segment_stats = stats.get(label, {'count': 0, 'percentage': 0})
            st.markdown(f"""
            <div class="kpi-card {card_class}">
                <div class="kpi-icon">{icon}</div>
                <p class="kpi-value {value_class}">{segment_stats['count']}</p>
                <p class="kpi-label">{label} ({segment_stats['percentage']}%)</p>
            </div>
            """, unsafe_allow_html=True)


def render_kpi_row(kpis, items):
    """Render a row of KPI cards"""
    cols = st.columns(len(items))
    for col, (key, label, icon) in zip(cols, items):
        with col:
            value = kpis.get(key, 'N/A')
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <p class="kpi-value">{value}</p>
                <p class="kpi-label">{label}</p>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main application function"""
    init_session_state()
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Management")
        
        data_source = st.radio(
            "Select Data Source",
            ["📤 Upload CSV", "🎲 Sample Data"],
            index=1,
            label_visibility="collapsed"
        )
        
        if data_source == "📤 Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload healthcare dataset",
                type=['csv'],
                help="Upload a CSV file with patient data"
            )
            if uploaded_file is not None:
                try:
                    st.session_state.data = pd.read_csv(uploaded_file)
                    st.session_state.data_validated = False
                    st.session_state.auto_trained = False
                    st.success(f"✅ Loaded {len(st.session_state.data):,} records")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            sample_size = st.slider("Sample Size", 100, 5000, 1000, 100)
            if st.button("🎲 Generate Sample", use_container_width=True):
                with st.spinner("Generating..."):
                    st.session_state.data = generate_sample_dataset(sample_size)
                    st.session_state.data_validated = False
                    st.session_state.auto_trained = False
                    st.success(f"✅ Generated {sample_size:,} records")
        
        st.markdown("---")
        
        # Show data status
        if st.session_state.data is not None:
            st.markdown("### 📊 Data Status")
            st.markdown(f"**Rows:** {len(st.session_state.data):,}")
            st.markdown(f"**Columns:** {len(st.session_state.data.columns)}")
            st.markdown(f"**Validated:** {'✅' if st.session_state.data_validated else '❌'}")
            st.markdown(f"**Model Trained:** {'✅' if st.session_state.auto_trained else '❌'}")
    
    # Main content
    if st.session_state.data is None:
        st.info("👆 Please load data using the sidebar to get started")
        with st.expander("ℹ️ About Sample Dataset"):
            st.markdown(get_sample_data_description())
        return
    
    df = st.session_state.data
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Dataset Overview",
        "📈 Analytics & EDA",
        "🤖 Model Performance",
        "🎯 Risk Predictions"
    ])
    
    # TAB 1: Dataset Overview
    with tab1:
        render_dataset_overview(df)
    
    # TAB 2: Analytics & EDA
    with tab2:
        render_analytics_eda(df)
    
    # TAB 3: Model Performance
    with tab3:
        render_model_performance(df)
    
    # TAB 4: Risk Predictions
    with tab4:
        render_risk_predictions(df)


def render_dataset_overview(df: pd.DataFrame):
    """Render Dataset Overview tab"""
    st.markdown('<h2 class="section-header">📋 Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Numeric Cols", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        missing_pct = round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 1)
        st.metric("Missing %", f"{missing_pct}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Validate data
    if not st.session_state.data_validated:
        if st.button("🔍 Validate Dataset", use_container_width=True, type="primary"):
            with st.spinner("Validating dataset..."):
                validator = DataValidator(df)
                messages = validator.validate()
                st.session_state.validator = validator
                st.session_state.data_validated = True
                st.rerun()
    
    # Show validation results
    if st.session_state.data_validated and st.session_state.validator:
        validator = st.session_state.validator
        
        st.markdown("### 🔍 Validation Results")
        
        # Schema summary
        with st.expander("📊 Schema Analysis", expanded=True):
            schema_df = validator.get_schema_summary()
            st.dataframe(schema_df, use_container_width=True, hide_index=True)
        
        # Validation messages
        with st.expander("📝 Validation Messages", expanded=True):
            messages = validator.validate()
            render_validation_messages(messages)
        
        # Target column selection
        st.markdown("### 🎯 Target Column Selection")
        
        recommended_target = validator.get_recommended_target()
        target_options = validator.target_candidates if validator.target_candidates else df.columns.tolist()
        
        default_idx = target_options.index(recommended_target) if recommended_target in target_options else 0
        
        st.session_state.selected_target = st.selectbox(
            "Select target column for prediction",
            target_options,
            index=default_idx,
            help="This column will be predicted by the ML models"
        )
        
        if st.session_state.selected_target:
            # Show target distribution
            col1, col2 = st.columns(2)
            with col1:
                target_counts = df[st.session_state.selected_target].value_counts()
                fig = px.pie(
                    values=target_counts.values,
                    names=target_counts.index,
                    title=f'Target Distribution: {st.session_state.selected_target}',
                    color_discrete_sequence=['#22c55e', '#ef4444']
                )
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Target Statistics")
                st.markdown(f"**Unique Values:** {df[st.session_state.selected_target].nunique()}")
                for val, count in target_counts.items():
                    pct = count / len(df) * 100
                    st.markdown(f"- **{val}:** {count:,} ({pct:.1f}%)")
    
    # Data preview
    with st.expander("👀 Data Preview"):
        st.dataframe(df.head(100), use_container_width=True)
    
    # Export cleaned data
    st.markdown("### 💾 Export Data")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Dataset (CSV)",
        csv_data,
        f"healthcare_data_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
        use_container_width=True
    )


def render_analytics_eda(df: pd.DataFrame):
    """Render Analytics & EDA tab"""
    st.markdown('<h2 class="section-header">📈 Analytics & Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    # KPI Dashboard
    kpis = create_dashboard_kpis(df)
    
    kpi_items_1 = [
        ('total_patients', 'Total Patients', '👥'),
        ('avg_age', 'Avg Age (yrs)', '📅'),
        ('readmission_rate', 'Readmission %', '🔄'),
        ('avg_risk_score', 'Avg Risk Score', '⚠️')
    ]
    render_kpi_row(kpis, kpi_items_1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    kpi_items_2 = [
        ('avg_los', 'Avg LOS (days)', '🏨'),
        ('complication_rate', 'Complication %', '⚡'),
        ('icu_rate', 'ICU Rate %', '🚑'),
        ('emergency_rate', 'Emergency %', '🆘')
    ]
    render_kpi_row(kpis, kpi_items_2)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_age_distribution(df)
        if fig:
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_gender_pie_chart(df)
        if fig:
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_diagnosis_bar_chart(df)
        if fig:
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_readmission_by_diagnosis(df)
        if fig:
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.markdown("### 🔗 Feature Correlations")
    fig = create_correlation_heatmap(df)
    if fig:
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_los_boxplot(df)
        if fig:
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_smoking_analysis(df)
        if fig:
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)


def render_model_performance(df: pd.DataFrame):
    """Render Model Performance tab"""
    st.markdown('<h2 class="section-header">🤖 Machine Learning Model Performance</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_validated:
        st.warning("⚠️ Please validate the dataset in the 'Dataset Overview' tab first")
        return
    
    if not st.session_state.selected_target:
        st.warning("⚠️ Please select a target column in the 'Dataset Overview' tab first")
        return
    
    validator = st.session_state.validator
    target = st.session_state.selected_target
    
    # Feature Engineering Section
    st.markdown("### ⚙️ Automatic Feature Engineering")
    
    with st.expander("🔧 Feature Engineering Settings", expanded=not st.session_state.auto_trained):
        # Get available features
        available_features = validator.get_feature_columns(exclude_target=target)
        
        st.markdown("**Available Features:**")
        selected_features = st.multiselect(
            "Select features for training",
            available_features,
            default=available_features[:15] if len(available_features) > 15 else available_features,
            help="Features to use for model training"
        )
        
        st.session_state.selected_features = selected_features
        
        col1, col2 = st.columns(2)
        with col1:
            create_derived = st.checkbox("Create derived health indicators", value=True)
        with col2:
            test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    
    if len(st.session_state.selected_features) < 2:
        st.warning("Please select at least 2 features for training")
        return
    
    # Train Models Button
    if st.button("🚀 Train Models Automatically", use_container_width=True, type="primary"):
        with st.spinner("Engineering features and training models..."):
            try:
                # Feature engineering
                feature_engineer = FeatureEngineer()
                df_engineered = feature_engineer.fit_transform(
                    df[st.session_state.selected_features + [target]].copy(),
                    target_column=target,
                    create_derived=create_derived
                )
                
                # Get all numeric features after engineering
                feature_cols = feature_engineer.get_numeric_features(df_engineered, exclude_columns=[target])
                
                # Train models
                model_manager = ModelManager()
                results = model_manager.train_pipeline(
                    df_engineered,
                    target_column=target,
                    feature_columns=feature_cols,
                    test_size=test_size
                )
                
                # Store in session state
                st.session_state.feature_engineer = feature_engineer
                st.session_state.model_manager = model_manager
                st.session_state.auto_trained = True
                
                st.success("✅ Models trained successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Training error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Show results if trained
    if st.session_state.auto_trained and st.session_state.model_manager:
        model_manager = st.session_state.model_manager
        metadata = model_manager.metadata
        
        st.markdown("### 📊 Model Comparison")
        
        # Model comparison cards
        model_metrics = metadata.get('model_metrics', {})
        best_model = metadata.get('best_model', '')
        
        cols = st.columns(len(model_metrics))
        for col, (model_name, metrics) in zip(cols, model_metrics.items()):
            with col:
                is_best = model_name == best_model
                card_class = "model-card best" if is_best else "model-card"
                badge = "🏆 Best" if is_best else ""
                
                st.markdown(f"""
                <div class="{card_class}">
                    <div class="model-name">{model_name.replace('_', ' ').title()} {badge}</div>
                    <div class="model-metric">{metrics['roc_auc']:.3f}</div>
                    <div class="model-label">ROC AUC</div>
                    <hr style="border-color: #334155; margin: 0.5rem 0;">
                    <div style="font-size: 0.8rem; color: #94a3b8;">
                        Accuracy: {metrics['accuracy']:.3f}<br>
                        Precision: {metrics['precision']:.3f}<br>
                        Recall: {metrics['recall']:.3f}<br>
                        F1: {metrics['f1_score']:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Comparison table
        comparison_df = model_manager.get_model_comparison()
        if not comparison_df.empty:
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Feature Importance
        st.markdown("### 📊 Feature Importance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            importance_df = model_manager.get_feature_importance('random_forest')
            if not importance_df.empty:
                fig = px.bar(
                    importance_df.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Random Forest Feature Importance',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    template='plotly_dark',
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            importance_df = model_manager.get_feature_importance('logistic_regression')
            if not importance_df.empty:
                fig = px.bar(
                    importance_df.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Logistic Regression Coefficients',
                    color='Importance',
                    color_continuous_scale='Plasma'
                )
                fig.update_layout(
                    template='plotly_dark',
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Engineered features summary
        if st.session_state.feature_engineer:
            feature_summary = st.session_state.feature_engineer.get_feature_summary()
            if not feature_summary.empty:
                with st.expander("🔧 Engineered Features"):
                    st.dataframe(feature_summary, use_container_width=True, hide_index=True)
        
        # Save model option
        st.markdown("### 💾 Save Model")
        if st.button("💾 Save Trained Models", use_container_width=True):
            try:
                filepath = model_manager.save_models()
                st.success(f"✅ Models saved to: {filepath}")
            except Exception as e:
                st.error(f"Save error: {str(e)}")


def render_risk_predictions(df: pd.DataFrame):
    """Render Risk Predictions tab"""
    st.markdown('<h2 class="section-header">🎯 Patient Risk Predictions</h2>', unsafe_allow_html=True)
    
    if not st.session_state.auto_trained or not st.session_state.model_manager:
        st.warning("⚠️ Please train models in the 'Model Performance' tab first")
        return
    
    model_manager = st.session_state.model_manager
    feature_engineer = st.session_state.feature_engineer
    target = st.session_state.selected_target
    
    # Generate predictions
    st.markdown("### 🔮 Generate Risk Predictions")
    
    model_options = list(model_manager.models.keys()) + ['best']
    selected_model = st.selectbox(
        "Select model for predictions",
        model_options,
        index=len(model_options) - 1,  # Default to 'best'
        format_func=lambda x: f"{x.replace('_', ' ').title()} {'(Recommended)' if x == 'best' else ''}"
    )
    
    if st.button("🎯 Generate Predictions", use_container_width=True, type="primary"):
        with st.spinner("Generating predictions..."):
            try:
                # Prepare data with same feature engineering
                df_pred = feature_engineer.fit_transform(
                    df[st.session_state.selected_features + [target]].copy(),
                    target_column=target,
                    create_derived=True
                )
                
                # Generate predictions
                predictions, probabilities = model_manager.predict(df_pred, selected_model)
                
                # Risk segmentation
                risk_segmenter = RiskSegmenter(low_threshold=0.3, high_threshold=0.7)
                categories, stats = risk_segmenter.segment(probabilities)
                
                # Store results
                st.session_state.predictions = predictions
                st.session_state.probabilities = probabilities
                st.session_state.risk_categories = categories
                st.session_state.risk_stats = stats
                st.session_state.risk_segmenter = risk_segmenter
                
                st.success("✅ Predictions generated!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Show predictions if available
    if st.session_state.risk_categories is not None:
        stats = st.session_state.risk_stats
        categories = st.session_state.risk_categories
        probabilities = st.session_state.probabilities
        
        st.markdown("### 📊 Risk Segmentation Overview")
        render_risk_segment_cards(stats)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution pie chart
            fig = px.pie(
                values=[stats['Low Risk']['count'], stats['Medium Risk']['count'], stats['High Risk']['count']],
                names=['Low Risk', 'Medium Risk', 'High Risk'],
                title='Patient Risk Distribution',
                color_discrete_sequence=['#22c55e', '#f59e0b', '#ef4444']
            )
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk probability histogram
            fig = px.histogram(
                x=probabilities,
                nbins=50,
                title='Risk Probability Distribution',
                labels={'x': 'Risk Probability', 'y': 'Count'},
                color_discrete_sequence=['#6366f1']
            )
            fig.add_vline(x=0.3, line_dash="dash", line_color="#22c55e", annotation_text="Low/Medium")
            fig.add_vline(x=0.7, line_dash="dash", line_color="#ef4444", annotation_text="Medium/High")
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results
        st.markdown("### 📋 Prediction Results")
        
        # Create results dataframe
        results_df = df.copy()
        results_df['Risk_Probability'] = probabilities.round(4)
        results_df['Risk_Category'] = categories
        results_df['Predicted_Outcome'] = st.session_state.predictions
        
        # Sort by risk
        results_df = results_df.sort_values('Risk_Probability', ascending=False)
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_category = st.multiselect(
                "Filter by Risk Category",
                ['Low Risk', 'Medium Risk', 'High Risk'],
                default=['High Risk', 'Medium Risk']
            )
        with col2:
            top_n = st.slider("Show top N patients", 10, 500, 100)
        
        # Apply filters
        filtered_df = results_df[results_df['Risk_Category'].isin(filter_category)].head(top_n)
        
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Risk_Probability': st.column_config.ProgressColumn(
                    "Risk Probability",
                    format="%.3f",
                    min_value=0,
                    max_value=1
                ),
                'Risk_Category': st.column_config.TextColumn("Risk Category"),
            }
        )
        
        # Export predictions
        st.markdown("### 💾 Export Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_all = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download All Predictions (CSV)",
                csv_all,
                f"predictions_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # High risk patients only
            high_risk_df = results_df[results_df['Risk_Category'] == 'High Risk']
            csv_high = high_risk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download High Risk Only (CSV)",
                csv_high,
                f"predictions_high_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        # Risk segment summary table
        st.markdown("### 📊 Segment Statistics")
        
        if st.session_state.risk_segmenter:
            segment_summary = st.session_state.risk_segmenter.get_segment_summary(
                df, categories
            )
            if not segment_summary.empty:
                st.dataframe(segment_summary, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
