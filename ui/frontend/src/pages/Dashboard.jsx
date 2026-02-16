import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const API_URL = 'http://localhost:8080/api';

function Dashboard() {
  const [stats, setStats] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000); // Refresh every 5s
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      const [statsRes, statusRes] = await Promise.all([
        axios.get(`${API_URL}/dashboard/stats`),
        axios.get(`${API_URL}/system/status`)
      ]);
      
      setStats(statsRes.data);
      setSystemStatus(statusRes.data);
      setLoading(false);
    } catch (error) {
      console.error('Error loading data:', error);
    }
  };

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <h1>Dashboard</h1>
      
      {/* Stats Cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label">Total Experiments</div>
          <div className="stat-value">{stats?.total_experiments || 0}</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-label">Running</div>
          <div className="stat-value" style={{color: '#3498db'}}>
            {stats?.running || 0}
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-label">Completed</div>
          <div className="stat-value" style={{color: '#2ecc71'}}>
            {stats?.completed || 0}
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-label">Failed</div>
          <div className="stat-value" style={{color: '#e74c3c'}}>
            {stats?.failed || 0}
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="card">
        <h2>System Status</h2>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px'}}>
          <div>
            <div className="stat-label">CPU Usage</div>
            <div className="stat-value" style={{fontSize: '24px'}}>
              {systemStatus?.cpu_percent?.toFixed(1)}%
            </div>
          </div>
          
          <div>
            <div className="stat-label">Memory Usage</div>
            <div className="stat-value" style={{fontSize: '24px'}}>
              {systemStatus?.memory_percent?.toFixed(1)}%
            </div>
          </div>
          
          <div>
            <div className="stat-label">GPUs Available</div>
            <div className="stat-value" style={{fontSize: '24px'}}>
              {systemStatus?.gpu_count || 0}
            </div>
          </div>
        </div>
      </div>

      {/* Recent Experiments */}
      <div className="card">
        <h2>Recent Experiments</h2>
        {stats?.recent && stats.recent.length > 0 ? (
          <table style={{width: '100%', borderCollapse: 'collapse'}}>
            <thead>
              <tr style={{borderBottom: '2px solid #ecf0f1'}}>
                <th style={{padding: '10px', textAlign: 'left'}}>Name</th>
                <th style={{padding: '10px', textAlign: 'left'}}>Status</th>
                <th style={{padding: '10px', textAlign: 'left'}}>Created</th>
              </tr>
            </thead>
            <tbody>
              {stats.recent.map((exp, i) => (
                <tr key={i} style={{borderBottom: '1px solid #ecf0f1'}}>
                  <td style={{padding: '10px'}}>{exp.name}</td>
                  <td style={{padding: '10px'}}>
                    <span style={{
                      padding: '4px 8px',
                      borderRadius: '4px',
                      background: exp.status === 'completed' ? '#2ecc71' : 
                                 exp.status === 'running' ? '#3498db' : '#e74c3c',
                      color: 'white',
                      fontSize: '12px'
                    }}>
                      {exp.status}
                    </span>
                  </td>
                  <td style={{padding: '10px'}}>
                    {new Date(exp.created_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>No experiments yet</p>
        )}
      </div>
    </div>
  );
}

export default Dashboard;