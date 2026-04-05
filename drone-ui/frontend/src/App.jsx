import React, { useState } from 'react';
import axios from 'axios';
import { Play, AlertCircle } from 'lucide-react';
import ImageUploader from './components/ImageUploader';
import BenchmarkResults from './components/BenchmarkResults';
import EnsembleResults from './components/EnsembleResults';
import Documentation from './components/Documentation';

const API_BASE_URL = 'http://localhost:8000/api';

function App() {
  const [activeTab, setActiveTab] = useState('benchmark'); // 'benchmark' | 'ensemble' | 'documentation'
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [optimizeZoomed, setOptimizeZoomed] = useState(false);
  const [padScale, setPadScale] = useState(0.15);
  const [confThresh, setConfThresh] = useState(0.30);
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const [benchmarkResults, setBenchmarkResults] = useState(null);
  const [ensembleResults, setEnsembleResults] = useState(null);

  const handleImageSelect = (file) => {
    setSelectedFile(file);
    if (!file) {
      setPreviewUrl(null);
      setBenchmarkResults(null);
      setEnsembleResults(null);
      setError(null);
      return;
    }
    
    const reader = new FileReader();
    reader.onload = () => setPreviewUrl(reader.result);
    reader.readAsDataURL(file);
    
    // Clear previous results
    setBenchmarkResults(null);
    setEnsembleResults(null);
    setError(null);
  };

  const handleRunInference = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('pad_image', optimizeZoomed);
    formData.append('pad_scale', padScale);
    formData.append('conf_thresh', confThresh);

    try {
      if (activeTab === 'benchmark') {
        const response = await axios.post(`${API_BASE_URL}/benchmark`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setBenchmarkResults(response.data.results);
      } else {
        const response = await axios.post(`${API_BASE_URL}/ensemble`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setEnsembleResults(response.data);
      }
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || err.message || 'An error occurred during inference');
    } finally {
      setIsLoading(false);
    }
  };

  // Change tab and optionally clear results if they don't match the tab
  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setError(null);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Drone Detection AI</h1>
        <p>Advanced real-time aerial object detection using an ensemble of state-of-the-art vision models.</p>
      </header>

      <div className="tabs">
        <button 
          className={`tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
          onClick={() => handleTabChange('benchmark')}
        >
          Single Model Benchmarks
        </button>
        <button 
          className={`tab-btn ${activeTab === 'ensemble' ? 'active' : ''}`}
          onClick={() => handleTabChange('ensemble')}
        >
          Multi-Model Architecture
        </button>
        <button 
          className={`tab-btn ${activeTab === 'documentation' ? 'active' : ''}`}
          onClick={() => handleTabChange('documentation')}
        >
          Architecture Documentation
        </button>
      </div>

      {activeTab === 'documentation' ? (
        <Documentation />
      ) : (
        <>
          <ImageUploader 
            onImageSelect={handleImageSelect} 
            selectedImagePreview={previewUrl} 
          />

      {previewUrl && (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginBottom: '3rem', gap: '1.5rem' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer', color: 'var(--text-secondary)', fontSize: '1rem' }}>
            <input 
              type="checkbox" 
              checked={optimizeZoomed}
              onChange={(e) => setOptimizeZoomed(e.target.checked)}
              style={{ width: '1.25rem', height: '1.25rem', accentColor: 'var(--accent-primary)', cursor: 'pointer' }}
            />
            Padding
          </label>
          {optimizeZoomed && (
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              Scale Tweaker ({padScale.toFixed(2)}): 
              <input 
                type="range" 
                min="0.1" 
                max="1.0" 
                step="0.05"
                value={padScale}
                onChange={(e) => setPadScale(parseFloat(e.target.value))}
                style={{ accentColor: 'var(--accent-primary)', width: '200px' }}
              />
            </label>
          )}

          <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.5rem' }}>
            Confidence Threshold ({confThresh.toFixed(2)}): 
            <input 
              type="range" 
              min="0.10" 
              max="0.95" 
              step="0.05"
              value={confThresh}
              onChange={(e) => setConfThresh(parseFloat(e.target.value))}
              style={{ accentColor: 'var(--warning)', width: '200px' }}
            />
          </label>

          <button 
            className="btn-primary" 
            onClick={handleRunInference}
            disabled={isLoading}
          >
            {isLoading ? (
              <>Running Inference...</>
            ) : (
              <>
                <Play size={20} fill="currentColor" />
                Run {activeTab === 'benchmark' ? 'Benchmark' : 'Ensemble'}
              </>
            )}
          </button>
        </div>
      )}

      {error && (
        <div className="error-message" style={{ maxWidth: '600px', margin: '0 auto 2rem auto' }}>
          <AlertCircle size={24} />
          <span><strong>Error:</strong> {error}</span>
        </div>
      )}

      {isLoading && (
        <div className="loader">
          <div className="spinner"></div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1.2rem' }}>
            Processing image through neural networks...
          </p>
        </div>
      )}

      {!isLoading && !error && activeTab === 'benchmark' && benchmarkResults && (
        <BenchmarkResults results={benchmarkResults} />
      )}

      {!isLoading && !error && activeTab === 'ensemble' && ensembleResults && (
        <EnsembleResults data={ensembleResults} />
      )}
        </>
      )}

    </div>
  );
}

export default App;
