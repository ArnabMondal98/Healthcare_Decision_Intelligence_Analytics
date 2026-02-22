🏥 Healthcare Decision Intelligence Analytics
AI-Powered Healthcare Risk Analytics & Decision Support Platform
<p align="center">










</p>

🚀 Live Application
👉 https://healthcaredecisionintelligenceanalytics.streamlit.app/

🧠 Project Vision

Healthcare organizations generate massive amounts of patient data, yet decision-making often remains reactive rather than predictive.

Healthcare Decision Intelligence Analytics transforms raw healthcare datasets into predictive intelligence, enabling:

Early risk detection

Data-driven clinical decisions

Automated analytics workflows

Explainable machine learning insights

This project demonstrates how Data Analytics + Machine Learning + Interactive Visualization can be combined into a real-world decision intelligence system.

🎯 Business Problem

Healthcare analysts and hospitals often struggle with:

Manual data analysis workflows

Lack of predictive risk insights

Complex ML implementation barriers

Non-interactive analytics tools

✅ Solution

A no-code analytics platform where users upload data and instantly receive:

Trained ML models

Risk predictions

Visual dashboards

Decision-ready insights

✨ Key Features
📂 Smart Data Ingestion

CSV dataset upload

Automatic schema validation

Sample dataset generation

Real-time dataset diagnostics

🤖 Automated Machine Learning Engine

Automatically trains multiple models:

Logistic Regression

Random Forest

Gradient Boosting

✔ Auto feature scaling
✔ Model comparison
✔ Best model selection (ROC-AUC based)

📊 Decision Intelligence Dashboard

Interactive analytics including:

Model performance metrics

Confusion matrix

ROC Curve analysis

Feature importance visualization

🧠 Patient Risk Segmentation

AI categorizes patients into:

Risk Level	Meaning
🟢 Low Risk	Stable patients
🟡 Medium Risk	Monitoring required
🔴 High Risk	Early intervention needed
📈 Interactive Visual Analytics

Dynamic charts

Real-time updates

Executive-friendly visuals

🏗️ System Architecture
Dataset Upload
      ↓
Data Validation
      ↓
Feature Engineering
      ↓
ML Training Pipeline
      ↓
Model Evaluation
      ↓
Risk Segmentation
      ↓
Interactive Dashboard
⚙️ Tech Stack
Layer	Technology
Language	Python
Framework	Streamlit
Data Processing	Pandas, NumPy
ML Models	Scikit-Learn
Visualization	Plotly, Matplotlib
Deployment	Streamlit Cloud
Model Storage	Temporary Cloud Cache (/tmp)
🧩 Project Structure
backend/
│
├── modules/
│   ├── data_validation.py
│   ├── data_processor.py
│   ├── feature_engineering.py
│   ├── model_manager.py
│   ├── ml_models.py
│   └── visualizations.py
│
└── run_streamlit.py
🚀 Application Workflow

1️⃣ Upload healthcare dataset
2️⃣ Automated validation & preprocessing
3️⃣ Feature scaling & preparation
4️⃣ Multi-model ML training
5️⃣ Best model auto-selection
6️⃣ Risk prediction generation
7️⃣ Interactive analytics dashboard

📊 Machine Learning Evaluation

Models are evaluated using:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

The system automatically selects the highest-performing model.

💡 Real-World Use Cases

Hospital patient risk prediction

Healthcare analytics dashboards

Clinical decision support

Insurance risk analysis

Data analyst portfolio demonstration

☁️ Deployment

Hosted using Streamlit Community Cloud (Free Tier).

⚠️ Note:
Models are stored in /tmp due to cloud filesystem permissions and retrain after restart.

💻 Local Setup
Clone Repository
git clone https://github.com/<your-username>/<repo>
cd Healthcare-Decision-Intelligence-Analytics
Install Dependencies
pip install -r requirements.txt
Run Application
streamlit run backend/run_streamlit.py
📸 Screenshots

Add screenshots here for maximum recruiter impact.

Recommended:

Dataset Upload Page

Model Training Results

Risk Segmentation Dashboard

Feature Importance Chart

🔮 Future Enhancements

✅ Model persistence (Cloud storage)

✅ SHAP explainability

✅ User authentication

✅ REST API integration

✅ Real-time healthcare streaming data

✅ Azure / AWS deployment

👨‍💻 Author

Arnab Mondal
Data Analyst | Power BI | Data Visualization | Data Engineering |

🔗 LinkedIn: https://www.linkedin.com/in/arnabmondal98/

💻 GitHub: https://github.com/ArnabMondal98

⭐ Support

If you found this project useful:

⭐ Star the repository
🍴 Fork the project
📢 Share feedback

🏆 Portfolio Impact

This project demonstrates:

✅ End-to-end ML pipeline
✅ Data engineering concepts
✅ Interactive analytics development
✅ Cloud deployment skills
✅ Production-style project architecture
