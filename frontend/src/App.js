import React, { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';

const API_URL = process.env.REACT_APP_BACKEND_URL || '';

// Healthcare-themed Icons
const Icons = {
  Hospital: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M3 21h18M9 8h1M9 12h1M9 16h1M14 8h1M14 12h1M14 16h1M5 21V5a2 2 0 012-2h10a2 2 0 012 2v16M12 7v4M10 9h4"/>
    </svg>
  ),
  Users: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2M9 11a4 4 0 100-8 4 4 0 000 8zM23 21v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75"/>
    </svg>
  ),
  AlertTriangle: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4M12 17h.01"/>
    </svg>
  ),
  Activity: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="22,12 18,12 15,21 9,3 6,12 2,12"/>
    </svg>
  ),
  TrendingUp: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="23,6 13.5,15.5 8.5,10.5 1,18"/>
      <polyline points="17,6 23,6 23,12"/>
    </svg>
  ),
  Heart: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M20.84 4.61a5.5 5.5 0 00-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 00-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 000-7.78z"/>
    </svg>
  ),
  UserPlus: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M16 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2M8.5 11a4 4 0 100-8 4 4 0 000 8zM20 8v6M23 11h-6"/>
    </svg>
  ),
  BarChart: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>
    </svg>
  ),
  PieChart: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21.21 15.89A10 10 0 118 2.83M22 12A10 10 0 0012 2v10z"/>
    </svg>
  ),
  Settings: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z"/>
    </svg>
  ),
  Download: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
    </svg>
  ),
  Upload: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
    </svg>
  ),
  Check: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="20,6 9,17 4,12"/>
    </svg>
  ),
  X: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
    </svg>
  ),
  ChevronRight: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="9,18 15,12 9,6"/>
    </svg>
  ),
  Database: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
    </svg>
  ),
  Brain: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 2a4 4 0 014 4c0 1.1-.9 2-2 2h-4c-1.1 0-2-.9-2-2a4 4 0 014-4zM9.5 22v-4.5M14.5 22v-4.5M8 10c-2.2 0-4 1.8-4 4s1.8 4 4 4M16 10c2.2 0 4 1.8 4 4s-1.8 4-4 4"/>
    </svg>
  ),
  Clipboard: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M16 4h2a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>
    </svg>
  ),
  Target: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>
    </svg>
  ),
  Stethoscope: () => (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M4.8 2.3A.3.3 0 105 2H4a2 2 0 00-2 2v5a6 6 0 006 6v0a6 6 0 006-6V4a2 2 0 00-2-2h-1a.2.2 0 10.3.3"/><path d="M8 15v1a6 6 0 006 6v0a6 6 0 006-6v-4"/><circle cx="20" cy="10" r="2"/>
    </svg>
  ),
};

// Risk Gauge Component
const RiskGauge = ({ value, size = 200 }) => {
  const percentage = value * 100;
  const rotation = (percentage / 100) * 180 - 90;
  
  const getColor = () => {
    if (percentage < 30) return '#22c55e';
    if (percentage < 70) return '#f59e0b';
    return '#ef4444';
  };
  
  return (
    <div className="risk-gauge" style={{ width: size, height: size / 2 + 40 }}>
      <svg viewBox="0 0 200 120" className="gauge-svg">
        {/* Background arc */}
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke="#1e293b"
          strokeWidth="16"
          strokeLinecap="round"
        />
        {/* Colored segments */}
        <path
          d="M 20 100 A 80 80 0 0 1 73 32"
          fill="none"
          stroke="#22c55e"
          strokeWidth="16"
          strokeLinecap="round"
          opacity="0.3"
        />
        <path
          d="M 73 32 A 80 80 0 0 1 127 32"
          fill="none"
          stroke="#f59e0b"
          strokeWidth="16"
          opacity="0.3"
        />
        <path
          d="M 127 32 A 80 80 0 0 1 180 100"
          fill="none"
          stroke="#ef4444"
          strokeWidth="16"
          strokeLinecap="round"
          opacity="0.3"
        />
        {/* Needle */}
        <g transform={`rotate(${rotation}, 100, 100)`}>
          <line x1="100" y1="100" x2="100" y2="35" stroke={getColor()} strokeWidth="4" strokeLinecap="round"/>
          <circle cx="100" cy="100" r="8" fill={getColor()}/>
        </g>
      </svg>
      <div className="gauge-value" style={{ color: getColor() }}>
        {percentage.toFixed(1)}%
      </div>
      <div className="gauge-label">Risk Score</div>
    </div>
  );
};

// KPI Card with animation
const KPICard = ({ icon, value, label, trend, color = "primary", delay = 0 }) => (
  <div 
    className={`kpi-card kpi-${color}`} 
    style={{ animationDelay: `${delay}ms` }}
    data-testid={`kpi-${label.toLowerCase().replace(/\s+/g, '-')}`}
  >
    <div className="kpi-icon-wrapper">{icon}</div>
    <div className="kpi-content">
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label}</div>
      {trend && (
        <div className={`kpi-trend ${trend > 0 ? 'up' : 'down'}`}>
          {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}%
        </div>
      )}
    </div>
  </div>
);

// Model Performance Card
const ModelCard = ({ name, metrics, isBest, onClick }) => (
  <div 
    className={`model-card ${isBest ? 'best' : ''}`} 
    onClick={onClick}
    data-testid={`model-${name.toLowerCase().replace(/\s+/g, '-')}`}
  >
    {isBest && <div className="best-badge">🏆 Best</div>}
    <div className="model-name">{name}</div>
    <div className="model-auc">{(metrics.roc_auc * 100).toFixed(1)}%</div>
    <div className="model-auc-label">ROC AUC</div>
    <div className="model-metrics-grid">
      <div className="metric-item">
        <span className="metric-value">{(metrics.accuracy * 100).toFixed(0)}%</span>
        <span className="metric-label">Accuracy</span>
      </div>
      <div className="metric-item">
        <span className="metric-value">{(metrics.precision * 100).toFixed(0)}%</span>
        <span className="metric-label">Precision</span>
      </div>
      <div className="metric-item">
        <span className="metric-value">{(metrics.recall * 100).toFixed(0)}%</span>
        <span className="metric-label">Recall</span>
      </div>
      <div className="metric-item">
        <span className="metric-value">{(metrics.f1_score * 100).toFixed(0)}%</span>
        <span className="metric-label">F1 Score</span>
      </div>
    </div>
  </div>
);

// Risk Segment Card
const RiskSegmentCard = ({ label, count, percentage, avgProb, colorClass }) => (
  <div className={`risk-segment-card ${colorClass}`} data-testid={`risk-${label.toLowerCase().replace(/\s+/g, '-')}`}>
    <div className="segment-header">
      <span className="segment-icon">
        {colorClass === 'low' ? '✓' : colorClass === 'medium' ? '⚠' : '⚡'}
      </span>
      <span className="segment-label">{label}</span>
    </div>
    <div className="segment-count">{count.toLocaleString()}</div>
    <div className="segment-percentage">{percentage}% of patients</div>
    {avgProb && <div className="segment-avg">Avg: {(avgProb * 100).toFixed(1)}%</div>}
  </div>
);

// Feature Bar Component
const FeatureBar = ({ feature, importance, maxImportance, index }) => {
  const width = (importance / maxImportance) * 100;
  return (
    <div className="feature-bar-item" style={{ animationDelay: `${index * 50}ms` }}>
      <div className="feature-info">
        <span className="feature-rank">#{index + 1}</span>
        <span className="feature-name">{feature.replace(/_/g, ' ')}</span>
      </div>
      <div className="feature-bar-container">
        <div className="feature-bar" style={{ width: `${width}%` }}>
          <span className="feature-value">{importance.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
};

// Patient Assessment Form
const PatientAssessmentForm = ({ onAssess, loading }) => {
  const [formData, setFormData] = useState({
    age: 55,
    gender: 'Male',
    comorbidities_count: 2,
    length_of_stay: 5,
    previous_admissions: 1,
    num_medications: 4,
    hemoglobin: 12.5,
    creatinine: 1.2,
    glucose: 120,
    blood_pressure_systolic: 130,
    blood_pressure_diastolic: 85,
    bmi: 28,
    smoking_status: 'Never',
    emergency_admission: 0,
    icu_stay: 0
  });

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onAssess(formData);
  };

  return (
    <form className="assessment-form" onSubmit={handleSubmit} data-testid="patient-assessment-form">
      <div className="form-grid">
        <div className="form-section">
          <h4><Icons.Users /> Demographics</h4>
          <div className="form-row">
            <label>
              Age
              <input
                type="number"
                value={formData.age}
                onChange={(e) => handleChange('age', parseInt(e.target.value))}
                min="18"
                max="100"
              />
            </label>
            <label>
              Gender
              <select value={formData.gender} onChange={(e) => handleChange('gender', e.target.value)}>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </label>
            <label>
              BMI
              <input
                type="number"
                step="0.1"
                value={formData.bmi}
                onChange={(e) => handleChange('bmi', parseFloat(e.target.value))}
                min="10"
                max="60"
              />
            </label>
          </div>
        </div>

        <div className="form-section">
          <h4><Icons.Activity /> Clinical Info</h4>
          <div className="form-row">
            <label>
              Comorbidities
              <input
                type="number"
                value={formData.comorbidities_count}
                onChange={(e) => handleChange('comorbidities_count', parseInt(e.target.value))}
                min="0"
                max="10"
              />
            </label>
            <label>
              Length of Stay
              <input
                type="number"
                value={formData.length_of_stay}
                onChange={(e) => handleChange('length_of_stay', parseInt(e.target.value))}
                min="1"
                max="100"
              />
            </label>
            <label>
              Previous Admissions
              <input
                type="number"
                value={formData.previous_admissions}
                onChange={(e) => handleChange('previous_admissions', parseInt(e.target.value))}
                min="0"
                max="20"
              />
            </label>
          </div>
        </div>

        <div className="form-section">
          <h4><Icons.Heart /> Vitals & Labs</h4>
          <div className="form-row">
            <label>
              Hemoglobin
              <input
                type="number"
                step="0.1"
                value={formData.hemoglobin}
                onChange={(e) => handleChange('hemoglobin', parseFloat(e.target.value))}
              />
            </label>
            <label>
              Creatinine
              <input
                type="number"
                step="0.1"
                value={formData.creatinine}
                onChange={(e) => handleChange('creatinine', parseFloat(e.target.value))}
              />
            </label>
            <label>
              Glucose
              <input
                type="number"
                value={formData.glucose}
                onChange={(e) => handleChange('glucose', parseInt(e.target.value))}
              />
            </label>
          </div>
          <div className="form-row">
            <label>
              BP Systolic
              <input
                type="number"
                value={formData.blood_pressure_systolic}
                onChange={(e) => handleChange('blood_pressure_systolic', parseInt(e.target.value))}
              />
            </label>
            <label>
              BP Diastolic
              <input
                type="number"
                value={formData.blood_pressure_diastolic}
                onChange={(e) => handleChange('blood_pressure_diastolic', parseInt(e.target.value))}
              />
            </label>
            <label>
              Medications
              <input
                type="number"
                value={formData.num_medications}
                onChange={(e) => handleChange('num_medications', parseInt(e.target.value))}
              />
            </label>
          </div>
        </div>

        <div className="form-section">
          <h4><Icons.AlertTriangle /> Risk Factors</h4>
          <div className="form-row">
            <label>
              Smoking Status
              <select value={formData.smoking_status} onChange={(e) => handleChange('smoking_status', e.target.value)}>
                <option value="Never">Never</option>
                <option value="Former">Former</option>
                <option value="Current">Current</option>
              </select>
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={formData.emergency_admission === 1}
                onChange={(e) => handleChange('emergency_admission', e.target.checked ? 1 : 0)}
              />
              <span>Emergency Admission</span>
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={formData.icu_stay === 1}
                onChange={(e) => handleChange('icu_stay', e.target.checked ? 1 : 0)}
              />
              <span>ICU Stay</span>
            </label>
          </div>
        </div>
      </div>

      <button type="submit" className="btn btn-primary btn-lg assess-btn" disabled={loading} data-testid="assess-patient-btn">
        {loading ? (
          <><span className="spinner"></span> Analyzing...</>
        ) : (
          <><Icons.Target /> Assess Patient Risk</>
        )}
      </button>
    </form>
  );
};

// Risk Assessment Result
const AssessmentResult = ({ result }) => {
  if (!result) return null;

  return (
    <div className="assessment-result" data-testid="assessment-result">
      <div className="result-header">
        <RiskGauge value={result.risk_probability} size={220} />
        <div className="result-summary">
          <div className={`risk-badge large ${result.risk_category.toLowerCase().replace(' ', '-')}`}>
            {result.risk_category}
          </div>
          <p className="result-prediction">
            Predicted 30-day readmission: <strong>{result.predicted_readmission ? 'Yes' : 'No'}</strong>
          </p>
        </div>
      </div>

      {result.risk_factors && result.risk_factors.length > 0 && (
        <div className="risk-factors-panel">
          <h4><Icons.AlertTriangle /> Key Risk Factors</h4>
          <div className="risk-factors-list">
            {result.risk_factors.map((factor, idx) => (
              <div key={idx} className="risk-factor-item">
                <span className="factor-bullet">•</span>
                <span className="factor-text">{factor.factor}</span>
                {factor.importance && (
                  <span className="factor-importance">{factor.importance}%</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {result.recommendation && (
        <div className="recommendation-panel">
          <h4><Icons.Clipboard /> Recommendations</h4>
          <p>{result.recommendation}</p>
        </div>
      )}
    </div>
  );
};

// Main App Component
function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [dashboardStats, setDashboardStats] = useState(null);
  const [validationResult, setValidationResult] = useState(null);
  const [trainingResult, setTrainingResult] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [assessmentResult, setAssessmentResult] = useState(null);
  const [chartData, setChartData] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedTarget, setSelectedTarget] = useState('');
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Fetch dashboard stats
  const fetchDashboardStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/dashboard-stats`);
      if (res.ok) {
        const data = await res.json();
        setDashboardStats(data);
        setDatasetInfo(data);
        
        // If validated but we don't have validation result, fetch it
        if (data.validated && !validationResult) {
          const valRes = await fetch(`${API_URL}/api/validate`, { method: 'POST' });
          const valData = await valRes.json();
          if (valData.success) {
            setValidationResult(valData);
            setSelectedTarget(valData.recommended_target || '');
            if (valData.schema) {
              const numericFeatures = valData.schema
                .filter(col => col.Type === 'numeric' || col.Type === 'boolean')
                .filter(col => col.Column !== valData.recommended_target)
                .map(col => col.Column);
              setSelectedFeatures(numericFeatures.slice(0, 15));
            }
          }
        }
      }
    } catch (err) {
      console.log('No dataset loaded');
    }
  }, [validationResult]);

  // Fetch chart data
  const fetchChartData = useCallback(async (chartType) => {
    try {
      const res = await fetch(`${API_URL}/api/chart-data/${chartType}`);
      if (res.ok) {
        const data = await res.json();
        if (data.success) {
          setChartData(prev => ({ ...prev, [chartType]: data.data }));
        }
      }
    } catch (err) {
      console.log(`Error fetching ${chartType} chart`);
    }
  }, []);

  useEffect(() => {
    fetchDashboardStats();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (datasetInfo?.validated) {
      fetchChartData('age_distribution');
      fetchChartData('risk_by_diagnosis');
      fetchChartData('risk_by_age_group');
      fetchChartData('gender_risk');
      fetchChartData('comorbidity_impact');
    }
  }, [datasetInfo?.validated, fetchChartData]);

  // Generate sample data
  const generateSampleData = async (nPatients = 1000) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/generate-sample?n_patients=${nPatients}`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        await fetchDashboardStats();
        setValidationResult(null);
        setTrainingResult(null);
        setPredictions(null);
        setAssessmentResult(null);
      }
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  // Validate dataset
  const validateDataset = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/validate`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        setValidationResult(data);
        setSelectedTarget(data.recommended_target || '');
        if (data.schema) {
          const numericFeatures = data.schema
            .filter(col => col.Type === 'numeric' || col.Type === 'boolean')
            .filter(col => col.Column !== data.recommended_target)
            .map(col => col.Column);
          setSelectedFeatures(numericFeatures.slice(0, 15));
        }
        await fetchDashboardStats();
      }
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  // Train models
  const trainModels = async () => {
    if (!selectedTarget || selectedFeatures.length < 2) {
      setError('Please select a target and at least 2 features');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_column: selectedTarget,
          feature_columns: selectedFeatures,
          test_size: 0.2
        })
      });
      const data = await res.json();
      if (data.success) {
        setTrainingResult(data);
        setFeatureImportance(data.feature_importance);
        await fetchDashboardStats();
      } else {
        setError(data.detail || 'Training failed');
      }
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  // Get predictions
  const getPredictions = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/predict?model=best`);
      const data = await res.json();
      if (data.success) {
        setPredictions(data);
      } else {
        setError(data.detail || 'Prediction failed');
      }
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  // Assess individual patient
  const assessPatient = async (patientData) => {
    setLoading(true);
    setError(null);
    setAssessmentResult(null);
    try {
      const res = await fetch(`${API_URL}/api/assess-patient`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patientData)
      });
      const data = await res.json();
      if (data.success) {
        setAssessmentResult(data);
      } else {
        setError(data.detail || 'Assessment failed');
      }
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  // Toggle feature selection
  const toggleFeature = (feature) => {
    if (selectedFeatures.includes(feature)) {
      setSelectedFeatures(selectedFeatures.filter(f => f !== feature));
    } else {
      setSelectedFeatures([...selectedFeatures, feature]);
    }
  };

  // Export predictions
  const exportPredictions = () => {
    window.open(`${API_URL}/api/export-predictions?model=best`, '_blank');
  };

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: <Icons.PieChart /> },
    { id: 'assessment', label: 'Patient Assessment', icon: <Icons.UserPlus /> },
    { id: 'data', label: 'Data Management', icon: <Icons.Database /> },
    { id: 'models', label: 'Model Performance', icon: <Icons.Brain /> },
    { id: 'predictions', label: 'Risk Predictions', icon: <Icons.Target /> },
  ];

  return (
    <div className="app" data-testid="healthcare-analytics-app">
      {/* Sidebar */}
      <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`} data-testid="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <Icons.Hospital />
            {!sidebarCollapsed && <span>HealthRisk AI</span>}
          </div>
          <button className="collapse-btn" onClick={() => setSidebarCollapsed(!sidebarCollapsed)}>
            <Icons.ChevronRight />
          </button>
        </div>

        <nav className="sidebar-nav">
          {navItems.map(item => (
            <button
              key={item.id}
              className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
              data-testid={`nav-${item.id}`}
            >
              {item.icon}
              {!sidebarCollapsed && <span>{item.label}</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          {!sidebarCollapsed && (
            <div className="sidebar-stats">
              <div className="stat-item">
                <span className={`status-dot ${datasetInfo?.validated ? 'green' : 'red'}`}></span>
                <span>Data {datasetInfo?.validated ? 'Ready' : 'Needed'}</span>
              </div>
              <div className="stat-item">
                <span className={`status-dot ${datasetInfo?.model_trained ? 'green' : 'red'}`}></span>
                <span>Model {datasetInfo?.model_trained ? 'Trained' : 'Not Trained'}</span>
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {/* Header */}
        <header className="main-header">
          <div className="header-left">
            <h1>{navItems.find(i => i.id === activeTab)?.label || 'Dashboard'}</h1>
            <p className="header-subtitle">Healthcare Risk Analytics Platform</p>
          </div>
          <div className="header-right">
            {datasetInfo && (
              <div className="header-badges">
                <span className={`badge ${datasetInfo.validated ? 'success' : 'warning'}`}>
                  {datasetInfo.validated ? '✓ Validated' : '○ Not Validated'}
                </span>
                <span className={`badge ${datasetInfo.model_trained ? 'success' : 'warning'}`}>
                  {datasetInfo.model_trained ? '✓ Model Ready' : '○ Train Model'}
                </span>
              </div>
            )}
          </div>
        </header>

        {/* Error Banner */}
        {error && (
          <div className="error-banner" data-testid="error-banner">
            <Icons.AlertTriangle />
            <span>{error}</span>
            <button onClick={() => setError(null)}><Icons.X /></button>
          </div>
        )}

        {/* Content Area */}
        <div className="content-area">
          {/* Dashboard Tab */}
          {activeTab === 'dashboard' && (
            <div className="tab-content dashboard" data-testid="dashboard-tab">
              {!dashboardStats ? (
                <div className="empty-state">
                  <Icons.Database />
                  <h3>No Data Loaded</h3>
                  <p>Generate sample data or upload a dataset to get started</p>
                  <button className="btn btn-primary" onClick={() => generateSampleData(1000)} disabled={loading}>
                    {loading ? 'Generating...' : 'Generate Sample Data'}
                  </button>
                </div>
              ) : (
                <>
                  {/* KPI Cards */}
                  <div className="kpi-row">
                    <KPICard 
                      icon={<Icons.Users />} 
                      value={dashboardStats.total_patients?.toLocaleString() || '0'} 
                      label="Total Patients" 
                      color="primary"
                      delay={0}
                    />
                    <KPICard 
                      icon={<Icons.AlertTriangle />} 
                      value={dashboardStats.high_risk_count?.toLocaleString() || '0'} 
                      label="High Risk" 
                      color="danger"
                      delay={100}
                    />
                    <KPICard 
                      icon={<Icons.Activity />} 
                      value={dashboardStats.avg_risk_score?.toFixed(2) || '0.00'} 
                      label="Avg Risk Score" 
                      color="warning"
                      delay={200}
                    />
                    <KPICard 
                      icon={<Icons.TrendingUp />} 
                      value={`${dashboardStats.readmission_rate || 0}%`} 
                      label="Readmission Rate" 
                      color="info"
                      delay={300}
                    />
                  </div>

                  <div className="kpi-row">
                    <KPICard 
                      icon={<Icons.Heart />} 
                      value={`${dashboardStats.avg_age || 0} yrs`} 
                      label="Avg Age" 
                      color="secondary"
                      delay={400}
                    />
                    <KPICard 
                      icon={<Icons.Hospital />} 
                      value={`${dashboardStats.avg_los || 0} days`} 
                      label="Avg Stay" 
                      color="secondary"
                      delay={500}
                    />
                    <KPICard 
                      icon={<Icons.Stethoscope />} 
                      value={`${dashboardStats.emergency_rate || 0}%`} 
                      label="Emergency Rate" 
                      color="secondary"
                      delay={600}
                    />
                    <KPICard 
                      icon={<Icons.Activity />} 
                      value={`${dashboardStats.icu_rate || 0}%`} 
                      label="ICU Rate" 
                      color="secondary"
                      delay={700}
                    />
                  </div>

                  {/* Charts Section */}
                  <div className="charts-grid">
                    {/* Risk by Diagnosis */}
                    {chartData.risk_by_diagnosis && (
                      <div className="chart-card">
                        <h3><Icons.BarChart /> Readmission Risk by Diagnosis</h3>
                        <div className="chart-bars">
                          {chartData.risk_by_diagnosis.slice(0, 8).map((item, idx) => (
                            <div key={idx} className="chart-bar-row">
                              <span className="bar-label">{item.diagnosis}</span>
                              <div className="bar-container">
                                <div 
                                  className="bar-fill" 
                                  style={{ 
                                    width: `${item.risk_rate}%`,
                                    backgroundColor: item.risk_rate > 50 ? '#ef4444' : item.risk_rate > 30 ? '#f59e0b' : '#22c55e'
                                  }}
                                />
                              </div>
                              <span className="bar-value">{item.risk_rate}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Risk by Age Group */}
                    {chartData.risk_by_age_group && (
                      <div className="chart-card">
                        <h3><Icons.Users /> Risk Score by Age Group</h3>
                        <div className="age-group-chart">
                          {chartData.risk_by_age_group.map((item, idx) => (
                            <div key={idx} className="age-bar-col">
                              <div 
                                className="age-bar" 
                                style={{ height: `${item.avg_risk * 200}px` }}
                              >
                                <span className="age-bar-value">{(item.avg_risk * 100).toFixed(0)}%</span>
                              </div>
                              <span className="age-bar-label">{item.age_group}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Gender Distribution */}
                    {dashboardStats.gender_distribution && (
                      <div className="chart-card">
                        <h3><Icons.PieChart /> Gender Distribution</h3>
                        <div className="pie-chart-container">
                          {Object.entries(dashboardStats.gender_distribution).map(([gender, count], idx) => (
                            <div key={idx} className="pie-segment">
                              <div className={`pie-circle ${gender.toLowerCase()}`}>
                                <span>{((count / dashboardStats.total_patients) * 100).toFixed(0)}%</span>
                              </div>
                              <span className="pie-label">{gender}</span>
                              <span className="pie-count">{count.toLocaleString()}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Top Diagnoses */}
                    {dashboardStats.top_diagnoses && (
                      <div className="chart-card">
                        <h3><Icons.Clipboard /> Top Diagnoses</h3>
                        <div className="diagnoses-list">
                          {Object.entries(dashboardStats.top_diagnoses).map(([diagnosis, count], idx) => (
                            <div key={idx} className="diagnosis-item">
                              <span className="diagnosis-rank">#{idx + 1}</span>
                              <span className="diagnosis-name">{diagnosis}</span>
                              <span className="diagnosis-count">{count.toLocaleString()}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}

          {/* Patient Assessment Tab */}
          {activeTab === 'assessment' && (
            <div className="tab-content assessment" data-testid="assessment-tab">
              {!datasetInfo?.model_trained ? (
                <div className="empty-state">
                  <Icons.Brain />
                  <h3>Model Not Trained</h3>
                  <p>Train a model first to enable patient risk assessment</p>
                  <button className="btn btn-primary" onClick={() => setActiveTab('models')}>
                    Go to Model Training
                  </button>
                </div>
              ) : (
                <div className="assessment-container">
                  <div className="assessment-intro">
                    <Icons.UserPlus />
                    <div>
                      <h2>Individual Patient Risk Assessment</h2>
                      <p>Enter patient details to get an instant risk prediction with personalized insights</p>
                    </div>
                  </div>
                  
                  <div className="assessment-content">
                    <PatientAssessmentForm onAssess={assessPatient} loading={loading} />
                    <AssessmentResult result={assessmentResult} />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Data Management Tab */}
          {activeTab === 'data' && (
            <div className="tab-content data-management" data-testid="data-tab">
              <div className="data-actions">
                <div className="action-card">
                  <Icons.Database />
                  <h3>Generate Sample Data</h3>
                  <p>Create realistic healthcare patient data for demonstration</p>
                  <button 
                    className="btn btn-primary" 
                    onClick={() => generateSampleData(1000)}
                    disabled={loading}
                    data-testid="generate-sample-btn"
                  >
                    {loading ? 'Generating...' : 'Generate 1,000 Patients'}
                  </button>
                </div>
                
                <div className="action-card">
                  <Icons.Upload />
                  <h3>Upload CSV File</h3>
                  <p>Upload your own healthcare dataset for analysis</p>
                  <label className="upload-btn">
                    <input type="file" accept=".csv" onChange={(e) => {/* handle upload */}} />
                    <span>Choose File</span>
                  </label>
                </div>
              </div>

              {datasetInfo && (
                <div className="data-info-panel">
                  <h3>Dataset Information</h3>
                  <div className="info-grid">
                    <div className="info-item">
                      <span className="info-value">{datasetInfo.total_patients?.toLocaleString()}</span>
                      <span className="info-label">Total Rows</span>
                    </div>
                    <div className="info-item">
                      <span className="info-value">{datasetInfo.total_columns}</span>
                      <span className="info-label">Columns</span>
                    </div>
                    <div className="info-item">
                      <span className={`info-status ${datasetInfo.validated ? 'valid' : 'invalid'}`}>
                        {datasetInfo.validated ? 'Validated' : 'Not Validated'}
                      </span>
                      <span className="info-label">Status</span>
                    </div>
                  </div>

                  {!datasetInfo.validated && (
                    <button 
                      className="btn btn-primary btn-lg"
                      onClick={validateDataset}
                      disabled={loading}
                      data-testid="validate-btn"
                    >
                      {loading ? 'Validating...' : 'Validate Dataset'}
                    </button>
                  )}

                  {validationResult && (
                    <div className="validation-results">
                      <h4>Schema Analysis</h4>
                      <div className="schema-table-container">
                        <table className="schema-table">
                          <thead>
                            <tr>
                              <th>Column</th>
                              <th>Type</th>
                              <th>Missing %</th>
                              <th>Target</th>
                            </tr>
                          </thead>
                          <tbody>
                            {validationResult.schema?.slice(0, 10).map((col, idx) => (
                              <tr key={idx}>
                                <td>{col.Column}</td>
                                <td><span className={`type-badge ${col.Type}`}>{col.Type}</span></td>
                                <td>{col['Missing %']}%</td>
                                <td>{col['Target Candidate'] ? '✓' : ''}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Model Performance Tab */}
          {activeTab === 'models' && (
            <div className="tab-content models" data-testid="models-tab">
              {!validationResult && !datasetInfo?.validated ? (
                <div className="empty-state">
                  <Icons.Database />
                  <h3>Dataset Not Validated</h3>
                  <p>Validate your dataset first to enable model training</p>
                  <button className="btn btn-primary" onClick={() => setActiveTab('data')}>
                    Go to Data Management
                  </button>
                </div>
              ) : (
                <>
                  {/* Feature Selection */}
                  <div className="feature-selection-panel">
                    <div className="panel-header">
                      <Icons.Settings />
                      <div>
                        <h3>Feature Selection</h3>
                        <p>Select features for model training ({selectedFeatures.length} selected)</p>
                      </div>
                    </div>
                    
                    <div className="feature-chips">
                      {validationResult?.schema
                        ?.filter(col => col.Type === 'numeric' || col.Type === 'boolean')
                        .filter(col => col.Column !== selectedTarget)
                        .map((col, idx) => (
                          <button
                            key={idx}
                            className={`feature-chip ${selectedFeatures.includes(col.Column) ? 'selected' : ''}`}
                            onClick={() => toggleFeature(col.Column)}
                          >
                            {selectedFeatures.includes(col.Column) && <Icons.Check />}
                            {col.Column}
                          </button>
                        ))}
                    </div>

                    <button 
                      className="btn btn-primary btn-lg"
                      onClick={trainModels}
                      disabled={loading || selectedFeatures.length < 2}
                      data-testid="train-btn"
                    >
                      {loading ? (
                        <><span className="spinner"></span> Training Models...</>
                      ) : (
                        <><Icons.Brain /> Train Models</>
                      )}
                    </button>
                  </div>

                  {/* Training Results */}
                  {trainingResult && (
                    <>
                      <div className="models-grid">
                        {Object.entries(trainingResult.results || {}).map(([name, metrics]) => (
                          <ModelCard 
                            key={name}
                            name={name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                            metrics={metrics}
                            isBest={name === trainingResult.best_model}
                          />
                        ))}
                      </div>

                      {/* Feature Importance */}
                      {featureImportance && featureImportance.length > 0 && (
                        <div className="feature-importance-panel">
                          <div className="panel-header">
                            <Icons.BarChart />
                            <h3>Feature Importance</h3>
                          </div>
                          <div className="feature-bars">
                            {featureImportance.slice(0, 12).map((item, idx) => (
                              <FeatureBar
                                key={idx}
                                feature={item.Feature}
                                importance={item['Importance %']}
                                maxImportance={featureImportance[0]['Importance %']}
                                index={idx}
                              />
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </>
              )}
            </div>
          )}

          {/* Risk Predictions Tab */}
          {activeTab === 'predictions' && (
            <div className="tab-content predictions" data-testid="predictions-tab">
              {!trainingResult && !datasetInfo?.model_trained ? (
                <div className="empty-state">
                  <Icons.Brain />
                  <h3>Models Not Trained</h3>
                  <p>Train models first to generate risk predictions</p>
                  <button className="btn btn-primary" onClick={() => setActiveTab('models')}>
                    Go to Model Training
                  </button>
                </div>
              ) : (
                <>
                  <div className="predictions-header">
                    <button 
                      className="btn btn-primary btn-lg"
                      onClick={getPredictions}
                      disabled={loading}
                      data-testid="predict-btn"
                    >
                      {loading ? (
                        <><span className="spinner"></span> Generating...</>
                      ) : (
                        <><Icons.Target /> Generate Risk Predictions</>
                      )}
                    </button>
                    
                    {predictions && (
                      <button className="btn btn-secondary" onClick={exportPredictions} data-testid="export-btn">
                        <Icons.Download /> Export CSV
                      </button>
                    )}
                  </div>

                  {predictions && (
                    <>
                      {/* Risk Segments */}
                      <div className="risk-segments-grid">
                        <RiskSegmentCard
                          label="Low Risk"
                          count={predictions.risk_segments?.['Low Risk']?.count || 0}
                          percentage={predictions.risk_segments?.['Low Risk']?.percentage || 0}
                          avgProb={predictions.risk_segments?.['Low Risk']?.avg_probability}
                          colorClass="low"
                        />
                        <RiskSegmentCard
                          label="Medium Risk"
                          count={predictions.risk_segments?.['Medium Risk']?.count || 0}
                          percentage={predictions.risk_segments?.['Medium Risk']?.percentage || 0}
                          avgProb={predictions.risk_segments?.['Medium Risk']?.avg_probability}
                          colorClass="medium"
                        />
                        <RiskSegmentCard
                          label="High Risk"
                          count={predictions.risk_segments?.['High Risk']?.count || 0}
                          percentage={predictions.risk_segments?.['High Risk']?.percentage || 0}
                          avgProb={predictions.risk_segments?.['High Risk']?.avg_probability}
                          colorClass="high"
                        />
                      </div>

                      {/* Predictions Table */}
                      <div className="predictions-table-panel">
                        <h3>Patient Risk Predictions (Top 100)</h3>
                        <div className="table-container">
                          <table className="predictions-table">
                            <thead>
                              <tr>
                                <th>Patient #</th>
                                <th>Risk Probability</th>
                                <th>Risk Category</th>
                                <th>Predicted Outcome</th>
                              </tr>
                            </thead>
                            <tbody>
                              {predictions.predictions?.map((pred, idx) => (
                                <tr key={idx} className={`risk-row ${pred.risk_category.toLowerCase().replace(' ', '-')}`}>
                                  <td>{pred.index + 1}</td>
                                  <td>
                                    <div className="probability-cell">
                                      <div className="prob-bar">
                                        <div 
                                          className="prob-fill" 
                                          style={{ 
                                            width: `${pred.risk_probability * 100}%`,
                                            backgroundColor: pred.risk_probability < 0.3 ? '#22c55e' : pred.risk_probability < 0.7 ? '#f59e0b' : '#ef4444'
                                          }}
                                        />
                                      </div>
                                      <span>{(pred.risk_probability * 100).toFixed(1)}%</span>
                                    </div>
                                  </td>
                                  <td>
                                    <span className={`risk-badge ${pred.risk_category.toLowerCase().replace(' ', '-')}`}>
                                      {pred.risk_category}
                                    </span>
                                  </td>
                                  <td>{pred.predicted_outcome === 1 ? 'Yes' : 'No'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
