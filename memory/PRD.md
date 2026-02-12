# Healthcare Risk Analytics Platform - PRD

## Project Overview
A production-ready Healthcare Risk Analytics Platform built with React frontend and FastAPI backend, designed for interview demonstrations and real-world healthcare risk prediction.

## Original Problem Statement
Create a Healthcare Risk Analytics Platform with:
- Patient risk assessment and prediction
- Interactive dashboards with visualizations
- ML model training and comparison
- Export capabilities

## Latest Enhancement (Jan 2026)
Enhanced for interview demonstrations with:
- Individual Patient Risk Assessment Form with instant predictions
- Interactive risk gauge visualization
- Key risk factors panel with explanations
- Personalized care recommendations
- Beautiful healthcare-themed UI with sidebar navigation

## Architecture

### Backend (FastAPI) - `/app/backend/`
- `server.py` - Main API with 15+ endpoints
- `modules/data_validation.py` - Schema detection
- `modules/feature_engineering.py` - Auto feature creation
- `modules/model_manager.py` - ML training/prediction
- `modules/sample_data.py` - Dataset generator

### Frontend (React) - `/app/frontend/src/`
- `App.js` - Main component with 5 tabs
- `App.css` - Healthcare-themed dark UI

## Key Features Implemented

### 1. Patient Risk Assessment Form
- Input patient demographics, vitals, labs
- Instant risk prediction
- Risk gauge visualization (0-100%)
- Key risk factors with importance %
- Personalized recommendations

### 2. Interactive Dashboard
- 8 KPI cards (Total Patients, High Risk, Avg Risk Score, etc.)
- Risk by Diagnosis chart
- Risk Score by Age Group chart
- Gender distribution
- Top diagnoses list

### 3. ML Model Performance
- 3 models: Logistic Regression, Random Forest, Gradient Boosting
- Feature selection interface (15 chips)
- Feature importance visualization
- Model comparison cards with metrics

### 4. Risk Predictions
- Low/Medium/High segmentation
- Visual probability bars
- Patient predictions table
- CSV export functionality

### 5. Enhanced UI/UX
- Healthcare-themed color palette (teal, cyan)
- Collapsible sidebar navigation
- Animated KPI cards
- Status indicators (Data Ready, Model Trained)

## API Endpoints
- `POST /api/generate-sample` - Generate sample data
- `POST /api/validate` - Validate dataset
- `POST /api/train` - Train ML models
- `GET /api/predict` - Batch predictions
- `POST /api/assess-patient` - Individual patient assessment
- `GET /api/dashboard-stats` - KPI statistics
- `GET /api/chart-data/{type}` - Chart data
- `GET /api/feature-importance` - Feature rankings
- `GET /api/export-predictions` - CSV download

## Test Results
- Backend: 100% (13/13 tests passed)
- Frontend: 95% (25/26 components functional)

## Prioritized Backlog

### P1 (Next Sprint)
- [ ] Add ROC curve visualization with Plotly
- [ ] Add confusion matrix heatmap
- [ ] Implement model persistence/saving

### P2 (Future)
- [ ] User authentication
- [ ] Historical prediction tracking
- [ ] PDF report generation
- [ ] Email alerts for high-risk patients

## Demo Flow for Interviews
1. Show Dashboard with KPIs and charts
2. Generate sample data (1000 patients)
3. Validate and train models
4. Demo Patient Assessment - enter values, see risk gauge
5. Show Risk Predictions with segments
6. Export predictions CSV
