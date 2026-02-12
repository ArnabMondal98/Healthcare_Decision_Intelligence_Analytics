# 🏥 Healthcare Decision Intelligence Analytics Platform

An end-to-end Healthcare Risk Analytics Platform that enables automated patient risk prediction, segmentation, and interactive analytics using machine learning and healthcare data.
This platform allows analysts or hospitals to upload patient datasets, automatically generate insights, and predict readmission risk using ML models.

---

## 🚀 Project Overview

Hospital readmissions significantly increase healthcare costs and reduce patient outcomes. This platform provides:

- Automated healthcare data processing
- Patient risk prediction
- Risk segmentation
- Interactive dashboards
- Decision-support insights
- Exportable analytics reports

The system acts as a **Healthcare Decision Support Tool** for analytics teams.

---

## ✨ Key Features

### 📊 Automated Analytics
- Dataset upload & validation
- Automatic preprocessing
- Feature engineering
- Exploratory analytics

### 🤖 Machine Learning Pipeline
- Logistic Regression
- Random Forest
- Gradient Boosting
- Model comparison
- Feature importance analysis

### ⚠ Risk Prediction
- Predict patient readmission risk
- Risk probability scoring
- High / Medium / Low segmentation

### 📈 Interactive Dashboard
- KPI cards
- Risk distribution
- Diagnosis-based risk analysis
- Age-group risk patterns

### 🧑 Patient Risk Assessment
- Manual patient input form
- Instant risk prediction
- Personalized recommendations

### 📥 Export Capabilities
- Download predictions
- Download processed datasets

---

## 🏗 Architecture
Frontend (React)
↓
Backend API
↓
ML & Analytics Engine
↓
Streamlit Dashboard Interface

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- FastAPI backend
- React frontend
- Plotly / Matplotlib visualizations

---

## 📦 Project Structure
backend/
├── modules/
├── streamlit_app.py
├── server.py
└── requirements.txt
frontend/
├── src/
└── public/


---

## ▶ How to Run Locally
```bash
cd backend
pip install -r requirements.txt
streamlit run streamlit_app.py

🌍 Future Improvements
Real-time hospital integration
Predictive monitoring
PDF reporting
Model retraining automation

👤 Author
Arnab Mondal – Data Analyst
