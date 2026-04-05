import React, { useState } from 'react';
import { Layers, FastForward, Activity, X, ArrowRight } from 'lucide-react';

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

const EnsembleResults = ({ data }) => {
  const [modalImg, setModalImg] = useState(null);

  if (!data) return null;

  const { wbf, cascade } = data;

  return (
    <div>
      <h2 style={{ textAlign: 'center', marginBottom: '1rem', color: '#c4b5fd' }}>
        Multi-Model Architecture
      </h2>
      <p style={{ textAlign: 'center', color: 'var(--text-secondary)', marginBottom: '2rem' }}>
        Compare standard WBF merging vs. Speed-Accuracy Cascade
      </p>

      <div className="results-grid">
        {/* Full WBF Ensemble */}
        <div className="glass-panel glass-panel-interactive model-card">
          <div className="model-header">
            <h3>
              <Layers size={24} color="#fcd34d" />
              Full WBF Ensemble
            </h3>
            <span className="badge" style={{ color: '#fcd34d', borderColor: 'rgba(252, 211, 77, 0.4)', background: 'rgba(252, 211, 77, 0.1)' }}>
              {wbf.total_ms} ms
            </span>
          </div>

          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Runs all 4 models and merges boxes weighted by confidence.
          </p>

          <img 
            src={`data:image/jpeg;base64,${wbf.annotated_image}`} 
            alt="WBF Ensemble"
            className="model-image"
            onClick={() => setModalImg(`data:image/jpeg;base64,${wbf.annotated_image}`)}
          />

          <div className="metrics-grid">
            <div className="metric-box">
              <span className="metric-label">Total Detections</span>
              <span className="metric-value">{wbf.count}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Latency</span>
              <span className="metric-value">{wbf.total_ms} ms</span>
            </div>
          </div>
        </div>

        {/* Cascade */}
        <div className="glass-panel glass-panel-interactive model-card">
          <div className="model-header">
            <h3>
              <FastForward size={24} color="#00c8ff" />
              Speed-Accuracy Cascade
            </h3>
            <span className="badge" style={{ color: '#00c8ff', borderColor: 'rgba(0, 200, 255, 0.4)', background: 'rgba(0, 200, 255, 0.1)' }}>
              {cascade.total_ms} ms
            </span>
          </div>

          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Starts with fastest model (SSD), escalates only if uncertain.
          </p>

          <img 
            src={`data:image/jpeg;base64,${cascade.annotated_image}`} 
            alt="Cascade"
            className="model-image"
            onClick={() => setModalImg(`data:image/jpeg;base64,${cascade.annotated_image}`)}
          />

          <div className="cascade-path">
            <span className="metric-label" style={{ width: '100%' }}>Execution Path:</span>
            {cascade.stages_used && cascade.stages_used.map((stage, idx) => (
              <React.Fragment key={stage}>
                {idx > 0 && <ArrowRight size={16} className="path-arrow" />}
                <span className="path-badge">{stage}</span>
              </React.Fragment>
            ))}
          </div>

          <div className="metrics-grid" style={{ marginTop: '0.5rem' }}>
            <div className="metric-box">
              <span className="metric-label">Stopped At</span>
              <span className="metric-value" style={{ fontSize: '1rem', color: '#c4b5fd' }}>{cascade.stopped_at}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Detections</span>
              <span className="metric-value">{cascade.count}</span>
            </div>
          </div>
        </div>
      </div>

      {modalImg && (
        <ImageModal src={modalImg} alt="Enlarged result" onClose={() => setModalImg(null)} />
      )}
    </div>
  );
};

export default EnsembleResults;
