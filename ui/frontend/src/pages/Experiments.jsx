import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const API_URL = 'http://localhost:8080/api';

function Experiments() {
  const [experiments, setExperiments] = useState([]);
  const [selectedExp, setSelectedExp] = useState(null);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    loadExperiments();
  }, []);

  const loadExperiments = async () => {
    try {
      const response = await axios.get(`${API_URL}/experiments?limit=50`);
      setExperiments(response.data);
    } catch (error) {
      console.error('Error loading experiments:', error);
    }
  };

  const loadExperimentMetrics = async (expId) => {
    try {
      const response = await axios.get(`${API_URL}/experiments/${expId}/metrics`);
      setMetrics(response.data);
    } catch (error) {
      console.error('Error loading metrics:', error);
    }
  };

  const handleSelectExperiment = (exp) => {
    setSelectedExp(exp);
    loadExperimentMetrics(exp.experiment_id);
  };

  return (
    <div>
      <h1>Experiments</h1>
      
      <div style={{display: 'grid', gridTemplateColumns: '300px 1fr', gap: '20px'}}>
        {/* Experiments List */}
        <div className="card">
          <h3>All Experiments</h3>
          <div style={{maxHeight: '600px', overflowY: 'auto'}}>
            {experiments.map((exp) => (
              <div
                key={exp.experiment_id}
                onClick={() => handleSelectExperiment(exp)}
                style={{
                  padding: '10px',
                  margin: '5px 0',
                  background: selectedExp?.experiment_id === exp.experiment_id ? '#ecf0f1' : 'white',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                <div style={{fontWeight: 'bold'}}>{exp.name}</div>
                <div style={{fontSize: '12px', color: '#7f8c8d'}}>
                  {exp.status}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Experiment Details */}
        <div>
          {selectedExp ? (
            <>
              <div className="card">
                <h2>{selectedExp.name}</h2>
                <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginTop: '20px'}}>
                  <div>
                    <div className="stat-label">Status</div>
                    <div style={{marginTop: '5px', fontWeight: 'bold'}}>
                      {selectedExp.status}
                    </div>
                  </div>
                  <div>
                    <div className="stat-label">Experiment ID</div>
                    <div style={{marginTop: '5px', fontSize: '12px', fontFamily: 'monospace'}}>
                      {selectedExp.experiment_id}
                    </div>
                  </div>
                  <div>
                    <div className="stat-label">Created</div>
                    <div style={{marginTop: '5px'}}>
                      {new Date(selectedExp.created_at).toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>

              {/* Metrics */}
              {metrics && (
                <div className="card">
                  <h3>Metrics</h3>
                  <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px'}}>
                    {Object.entries(metrics).map(([name, summary]) => (
                      <div key={name} style={{padding: '15px', background: '#f8f9fa', borderRadius: '4px'}}>
                        <div className="stat-label">{name}</div>
                        <div style={{fontSize: '20px', fontWeight: 'bold', margin: '5px 0'}}>
                          {summary.avg_value?.toFixed(4)}
                        </div>
                        <div style={{fontSize: '12px', color: '#7f8c8d'}}>
                          Min: {summary.min_value?.toFixed(4)} | Max: {summary.max_value?.toFixed(4)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="card">
              <p>Select an experiment to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Experiments;