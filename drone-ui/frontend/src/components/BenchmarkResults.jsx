import React, { useState } from 'react';
import { Camera, Zap, Activity, AlertCircle, X } from 'lucide-react';

const ImageModal = ({ src, alt, onClose }) => {
  if (!src) return null;
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}><X /></button>
        <img src={src} alt={alt} className="modal-img" />
      </div>
    </div>
  );
};

const BenchmarkResults = ({ results }) => {
  const [modalImg, setModalImg] = useState(null);

  if (!results || results.length === 0) return null;

  return (
    <div>
      <h2 style={{ textAlign: 'center', marginBottom: '1rem', color: '#93c5fd' }}>
        Single Model Benchmarks
      </h2>
      <p style={{ textAlign: 'center', color: 'var(--text-secondary)', marginBottom: '2rem' }}>
        Results from all 4 models ran independently on the same image.
      </p>

      <div className="results-grid">
        {results.map((result, idx) => (
          <div key={idx} className="glass-panel glass-panel-interactive model-card">
            <div className="model-header">
              <h3>
                <Camera size={24} color="var(--accent-primary)" />
                {result.model}
              </h3>
              {result.metrics?.latency_ms && (
                <span className="badge">{result.metrics.latency_ms} ms</span>
              )}
            </div>

            {result.error ? (
              <div className="error-message">
                <AlertCircle size={20} />
                <span>{result.error}</span>
              </div>
            ) : (
              <>
                <img 
                  src={`data:image/jpeg;base64,${result.annotated_image}`} 
                  alt={`${result.model} detections`}
                  className="model-image"
                  onClick={() => setModalImg(`data:image/jpeg;base64,${result.annotated_image}`)}
                />

                <div className="metrics-grid">
                  <div className="metric-box">
                    <span className="metric-label">Avg Confidence</span>
                    <span className={`metric-value ${result.metrics.avg_confidence > 0.5 ? 'good' : 'warn'}`}>
                      {result.metrics.avg_confidence > 0 ? result.metrics.avg_confidence.toFixed(2) : '0.00'}
                    </span>
                  </div>
                  <div className="metric-box">
                    <span className="metric-label">FPS</span>
                    <span className="metric-value">{result.metrics.fps}</span>
                  </div>
                  <div className="metric-box">
                    <span className="metric-label">Detections</span>
                    <span className="metric-value">{result.metrics.total_detections}</span>
                  </div>
                </div>

                <div className="class-counts">
                  {Object.entries(result.metrics.class_counts || {}).map(([name, count]) => {
                    if (count === 0) return null;
                    return (
                      <span key={name} className="class-tag">
                        {name}: {count}
                      </span>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        ))}
      </div>

      {modalImg && (
        <ImageModal src={modalImg} alt="Enlarged result" onClose={() => setModalImg(null)} />
      )}
    </div>
  );
};

export default BenchmarkResults;
