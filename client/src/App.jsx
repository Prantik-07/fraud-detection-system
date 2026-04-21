import React, { useState, useEffect } from 'react';
import { Shield, ShieldAlert, ShieldCheck, Activity, CreditCard, Clock, Server } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import './index.css';

function App() {
  const [stats, setStats] = useState({
    total_transactions: 284807,
    fraud_transactions: 492,
    fraud_rate: 0.0017,
    best_model: "XGBoost",
    best_model_f1: 0.91,
    best_model_recall: 0.92
  });

  const [formData, setFormData] = useState({
    Amount: '',
    Time: '',
    V14: '',
    V12: '',
    V10: '',
    V4: '',
    V11: ''
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState("Checking...");

  useEffect(() => {
    // Check API health
    fetch('/health')
      .then(res => res.json())
      .then(data => setApiStatus(data.status === 'ok' ? 'Online' : 'Offline'))
      .catch(() => setApiStatus('Offline'));
      
    // Fetch real stats if available
    fetch('/stats')
      .then(res => res.json())
      .then(data => setStats(data))
      .catch(err => console.log('Using default stats fallback'));
  }, []);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    // Build the 30-feature payload
    const payload = {
      Time: parseFloat(formData.Time) || 0,
      Amount: parseFloat(formData.Amount) || 0
    };
    // Default all V features to 0
    for (let i = 1; i <= 28; i++) {
      payload[`V${i}`] = 0.0;
    }
    // Override the specific ones we asked for
    payload.V14 = parseFloat(formData.V14) || 0;
    payload.V12 = parseFloat(formData.V12) || 0;
    payload.V10 = parseFloat(formData.V10) || 0;
    payload.V4 = parseFloat(formData.V4) || 0;
    payload.V11 = parseFloat(formData.V11) || 0;

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error('API Error');
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({
        prediction: "ERROR",
        message: "Failed to connect to API. Make sure FastAPI is running on port 8000.",
        confidence: 0
      });
    } finally {
      setLoading(false);
    }
  };

  const chartData = [
    { name: 'Logistic Regression', recall: 0.91, f1: 0.88 },
    { name: 'Random Forest', recall: 0.89, f1: 0.90 },
    { name: 'XGBoost', recall: 0.92, f1: 0.91 },
  ];

  return (
    <div className="dashboard">
      <header className="header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1><Shield style={{ display: 'inline', verticalAlign: 'middle', marginRight: '15px' }} size={40} color="#4f46e5" /> Fraud Sentinel</h1>
            <p>Smart Transaction Analysis & Risk Monitoring</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: apiStatus === 'Online' ? '#15803d' : '#be123c' }}>
            <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'currentColor', boxShadow: '0 0 10px currentColor' }}></div>
            <span style={{ fontWeight: 800, fontSize: '0.9rem', textTransform: 'uppercase' }}>{apiStatus}</span>
          </div>
        </div>
      </header>

      <div className="metrics-grid">
        <div className="glass-panel">
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', textTransform: 'uppercase' }}>Total Transactions</p>
          <p className="metric-value">{stats.total_transactions.toLocaleString()}</p>
        </div>
        <div className="glass-panel">
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', textTransform: 'uppercase' }}>Fraud Rate</p>
          <p className="metric-value">{(stats.fraud_rate * 100).toFixed(2)}%</p>
        </div>
        <div className="glass-panel">
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', textTransform: 'uppercase' }}>Best Model</p>
          <p className="metric-value">{stats.best_model}</p>
        </div>
        <div className="glass-panel">
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', textTransform: 'uppercase' }}>Best F1 Score</p>
          <p className="metric-value">{stats.best_model_f1.toFixed(2)}</p>
        </div>
      </div>

      <div className="content-grid">
        <div className="glass-panel">
          <h2 style={{ marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '12px', fontSize: '1.25rem' }}>
            <Activity size={20} color="var(--accent-teal)" /> Transaction Analysis
          </h2>
          <form onSubmit={handleSubmit}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div className="input-group">
                <label><CreditCard size={14} style={{ display:'inline', marginRight:'5px' }}/> Amount ($)</label>
                <input type="number" step="0.01" name="Amount" value={formData.Amount} onChange={handleChange} placeholder="e.g. 150.00" required />
              </div>
              <div className="input-group">
                <label><Clock size={14} style={{ display:'inline', marginRight:'5px' }}/> Time</label>
                <input type="number" step="1" name="Time" value={formData.Time} onChange={handleChange} placeholder="e.g. 40000" required />
              </div>
            </div>
            
            <p style={{ margin: '1rem 0 0.5rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>Key Principal Components (V Features)</p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div className="input-group">
                <label>V14</label>
                <input type="number" step="0.1" name="V14" value={formData.V14} onChange={handleChange} placeholder="-1.0" />
              </div>
              <div className="input-group">
                <label>V12</label>
                <input type="number" step="0.1" name="V12" value={formData.V12} onChange={handleChange} placeholder="-1.0" />
              </div>
              <div className="input-group">
                <label>V10</label>
                <input type="number" step="0.1" name="V10" value={formData.V10} onChange={handleChange} placeholder="-1.0" />
              </div>
              <div className="input-group">
                <label>V4</label>
                <input type="number" step="0.1" name="V4" value={formData.V4} onChange={handleChange} placeholder="1.0" />
              </div>
            </div>

            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? <span className="loader"></span> : 'Score Transaction'}
            </button>
          </form>

          {result && (
            <div className={`glass-panel result-card ${result.prediction === 'FRAUD' ? 'result-fraud' : result.prediction === 'LEGITIMATE' ? 'result-legit' : ''}`}>
              {result.prediction === 'FRAUD' ? (
                <>
                  <div className="status-badge fraud"><ShieldAlert size={18} /> HIGH RISK DETECTED</div>
                  <h3 style={{ color: 'var(--accent-fraud)', fontSize: '2rem', marginBottom: '0.5rem' }}>FRAUDULENT</h3>
                  <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
                </>
              ) : result.prediction === 'LEGITIMATE' ? (
                <>
                  <div className="status-badge legit"><ShieldCheck size={18} /> SAFE TRANSACTION</div>
                  <h3 style={{ color: 'var(--accent-legit)', fontSize: '2rem', marginBottom: '0.5rem' }}>LEGITIMATE</h3>
                  <p>Confidence: {((1 - result.confidence) * 100).toFixed(2)}%</p>
                </>
              ) : (
                <p style={{ color: '#ef4444' }}>{result.message}</p>
              )}
            </div>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div className="glass-panel" style={{ flex: 1 }}>
            <h2 style={{ marginBottom: '1.5rem' }}>Model Performance (SMOTE)</h2>
            <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.9rem' }}>
              Comparing Recall (ability to catch fraud) across different algorithms after applying SMOTE.
            </p>
            <div style={{ width: '100%', height: '280px' }}>
              <ResponsiveContainer>
                <BarChart data={chartData} margin={{ top: 20, right: 20, left: 0, bottom: 40 }}>
                  <XAxis 
                    dataKey="name" 
                    stroke="#475569" 
                    fontSize={10} 
                    tickLine={false} 
                    axisLine={false} 
                    interval={0} 
                    angle={-15} 
                    textAnchor="end"
                    fontWeight={600} 
                  />
                  <YAxis stroke="#475569" fontSize={11} tickLine={false} axisLine={false} />
                  <Tooltip 
                    cursor={{fill: 'rgba(0, 0, 0, 0.03)'}}
                    contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.9)', backdropFilter: 'blur(8px)', border: '1px solid #fff', borderRadius: '16px', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }}
                  />
                  <Bar dataKey="recall" name="Recall Score" radius={[10, 10, 0, 0]} barSize={45}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.name === 'XGBoost' ? '#6366f1' : entry.name === 'Random Forest' ? '#2dd4bf' : '#94a3b8'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
             <h3 style={{ marginBottom: '1rem', color: '#818cf8' }}>Why Recall Matters?</h3>
             <p style={{ color: 'var(--text-muted)', lineHeight: '1.6' }}>
               In credit card fraud detection, the cost of missing a fraudulent transaction (False Negative) is much higher than falsely flagging a legitimate one (False Positive). By using <strong>SMOTE</strong> to handle the 99.8% class imbalance, our XGBoost model increased Recall from ~80% to <strong>92%</strong>.
             </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
