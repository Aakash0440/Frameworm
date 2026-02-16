import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Download, Upload, Trash2, Play } from 'lucide-react';

const API_URL = 'http://localhost:8080/api';

function Models() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [exportFormat, setExportFormat] = useState('torchscript');
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const response = await axios.get(`${API_URL}/models`);
      setModels(response.data);
    } catch (error) {
      console.error('Error loading models:', error);
    }
  };

  const handleExport = async (model) => {
    setExporting(true);
    try {
      const response = await axios.post(`${API_URL}/models/export`, {
        checkpoint_path: model.path,
        format: exportFormat,
        output_dir: 'exported_models'
      });

      alert(`Model exported successfully to ${response.data.output_path}`);
      loadModels();
    } catch (error) {
      alert(`Export failed: ${error.message}`);
    } finally {
      setExporting(false);
    }
  };

  const formatBytes = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const formatDate = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  return (
    <div>
      <h1>Models</h1>
      
      <div className="card">
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px'}}>
          <h2>Model Registry</h2>
          <button 
            onClick={loadModels}
            style={{
              padding: '8px 16px',
              background: '#3498db',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Refresh
          </button>
        </div>

        {models.length === 0 ? (
          <div style={{textAlign: 'center', padding: '40px', color: '#7f8c8d'}}>
            <p>No models found</p>
            <p style={{fontSize: '14px'}}>Train a model to see it here</p>
          </div>
        ) : (
          <table style={{width: '100%', borderCollapse: 'collapse'}}>
            <thead>
              <tr style={{borderBottom: '2px solid #ecf0f1', background: '#f8f9fa'}}>
                <th style={{padding: '12px', textAlign: 'left'}}>Name</th>
                <th style={{padding: '12px', textAlign: 'left'}}>Size</th>
                <th style={{padding: '12px', textAlign: 'left'}}>Modified</th>
                <th style={{padding: '12px', textAlign: 'left'}}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model, i) => (
                <tr key={i} style={{borderBottom: '1px solid #ecf0f1'}}>
                  <td style={{padding: '12px'}}>
                    <div style={{fontWeight: 'bold'}}>{model.name}</div>
                    <div style={{fontSize: '12px', color: '#7f8c8d', fontFamily: 'monospace'}}>
                      {model.path}
                    </div>
                  </td>
                  <td style={{padding: '12px'}}>
                    {model.size_mb.toFixed(1)} MB
                  </td>
                  <td style={{padding: '12px'}}>
                    {formatDate(model.modified)}
                  </td>
                  <td style={{padding: '12px'}}>
                    <div style={{display: 'flex', gap: '10px'}}>
                      <button
                        onClick={() => {
                          setSelectedModel(model);
                          handleExport(model);
                        }}
                        disabled={exporting}
                        style={{
                          padding: '6px 12px',
                          background: '#2ecc71',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: exporting ? 'not-allowed' : 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '5px',
                          opacity: exporting ? 0.6 : 1
                        }}
                      >
                        <Download size={16} />
                        Export
                      </button>
                      
                      <button
                        style={{
                          padding: '6px 12px',
                          background: '#3498db',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '5px'
                        }}
                      >
                        <Play size={16} />
                        Deploy
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Export Options */}
      <div className="card">
        <h3>Export Settings</h3>
        <div style={{display: 'flex', gap: '20px', alignItems: 'center'}}>
          <label style={{display: 'flex', alignItems: 'center', gap: '10px'}}>
            Format:
            <select 
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value)}
              style={{
                padding: '8px',
                borderRadius: '4px',
                border: '1px solid #ddd'
              }}
            >
              <option value="torchscript">TorchScript (.pt)</option>
              <option value="onnx">ONNX (.onnx)</option>
            </select>
          </label>
          
          <div style={{fontSize: '14px', color: '#7f8c8d'}}>
            {exportFormat === 'torchscript' && '✓ PyTorch native, C++ deployable'}
            {exportFormat === 'onnx' && '✓ Framework agnostic, TensorRT compatible'}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Models;