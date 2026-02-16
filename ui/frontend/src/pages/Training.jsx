import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

function Training() {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingData, setTrainingData] = useState([]);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(100);
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    // WebSocket connection for real-time training updates
    if (isTraining) {
      const ws = new WebSocket('ws://localhost:8080/api/training/stream');
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        setCurrentEpoch(data.epoch);
        setTotalEpochs(data.total_epochs);
        
        setTrainingData(prev => [...prev, {
          epoch: data.epoch,
          loss: data.metrics.loss,
          val_loss: data.metrics.val_loss
        }].slice(-50)); // Keep last 50 points
        
        setLogs(prev => [...prev, 
          `Epoch ${data.epoch}/${data.total_epochs}: loss=${data.metrics.loss.toFixed(4)}, val_loss=${data.metrics.val_loss.toFixed(4)}`
        ].slice(-20)); // Keep last 20 logs
      };

      return () => ws.close();
    }
  }, [isTraining]);

  const handleStartTraining = () => {
    setIsTraining(true);
    setTrainingData([]);
    setLogs(['Training started...']);
  };

  const handleStopTraining = () => {
    setIsTraining(false);
    setLogs(prev => [...prev, 'Training stopped']);
  };

  const progress = (currentEpoch / totalEpochs) * 100;

  return (
    <div>
      <h1>Training</h1>

      {/* Training Controls */}
      <div className="card">
        <h2>Training Control</h2>
        
        <div style={{marginTop: '20px'}}>
          {!isTraining ? (
            <button
              onClick={handleStartTraining}
              style={{
                padding: '12px 24px',
                background: '#2ecc71',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                fontSize: '16px',
                cursor: 'pointer',
                fontWeight: 'bold'
              }}
            >
              Start Training
            </button>
          ) : (
            <button
              onClick={handleStopTraining}
              style={{
                padding: '12px 24px',
                background: '#e74c3c',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                fontSize: '16px',
                cursor: 'pointer',
                fontWeight: 'bold'
              }}
            >
              Stop Training
            </button>
          )}
        </div>

        {/* Progress Bar */}
        {isTraining && (
          <div style={{marginTop: '30px'}}>
            <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '10px'}}>
              <span>Epoch {currentEpoch} / {totalEpochs}</span>
              <span>{progress.toFixed(1)}%</span>
            </div>
            <div style={{
              width: '100%',
              height: '30px',
              background: '#ecf0f1',
              borderRadius: '15px',
              overflow: 'hidden'
            }}>
              <div style={{
                width: `${progress}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #3498db, #2ecc71)',
                transition: 'width 0.3s ease'
              }} />
            </div>
          </div>
        )}
      </div>

      {/* Training Metrics Chart */}
      {trainingData.length > 0 && (
        <div className="card">
          <h3>Training Metrics</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="loss" 
                stroke="#e74c3c" 
                name="Training Loss"
                strokeWidth={2}
              />
              <Line 
                type="monotone" 
                dataKey="val_loss" 
                stroke="#3498db" 
                name="Validation Loss"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Training Logs */}
      <div className="card">
        <h3>Training Logs</h3>
        <div style={{
          background: '#2c3e50',
          color: '#ecf0f1',
          padding: '15px',
          borderRadius: '4px',
          fontFamily: 'monospace',
          fontSize: '14px',
          maxHeight: '300px',
          overflowY: 'auto'
        }}>
          {logs.length === 0 ? (
            <div style={{color: '#7f8c8d'}}>No training logs yet</div>
          ) : (
            logs.map((log, i) => (
              <div key={i} style={{marginBottom: '5px'}}>
                {log}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default Training;