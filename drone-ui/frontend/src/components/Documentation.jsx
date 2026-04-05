import React from 'react';
import { Network, Database, Zap, BookOpen, Cpu, Sliders } from 'lucide-react';

const Documentation = () => {
  return (
    <div className="documentation-container" style={{ padding: '1rem', maxWidth: '1100px', margin: '0 auto', animation: 'fadeIn 0.5s ease-out' }}>
      <h2 style={{ textAlign: 'center', marginBottom: '2rem', color: '#c4b5fd', fontSize: '2.5rem' }}>
        System Architecture & Methodology
      </h2>

      {/* Intro Section */}
      <section className="doc-section glass-panel" style={{ marginBottom: '3rem' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.8rem', marginBottom: '1rem' }}>
          <BookOpen color="var(--accent-primary)" /> Project Overview
        </h3>
        <p style={{ color: 'var(--text-secondary)', lineHeight: '1.8', fontSize: '1.1rem', marginBottom: '1rem' }}>
          This standalone web application serves as a comprehensive dashboard for detecting autonomous micro-aerial vehicles (Drones, Airplanes, Helicopters) in dynamic environments. 
          To establish extremely high levels of precision and overcome traditional blindspots in computer vision pipelines, we deployed four distinctly different neural network architectures utilizing a deeply integrated FastAPI Python backend.
        </p>
      </section>

      {/* UI Features Section */}
      <section className="doc-section glass-panel" style={{ marginBottom: '3rem' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.8rem', marginBottom: '1.5rem' }}>
          <Sliders color="#ec4899" /> Dynamic Pipeline Tweaks
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
          <div>
            <h4 style={{ color: '#ec4899', fontSize: '1.3rem', marginBottom: '0.5rem' }}>Auto-Padding for Scale Normalization</h4>
            <p style={{ color: 'var(--text-secondary)', lineHeight: '1.6' }}>
              Models heavily overfit to the scale of objects they train on (e.g. associating small specks with drones). When fed high-res, zoomed-in images from the web, the models mistakenly classify drones as helicopters merely due to massive visual size. The dynamic **Auto-Pad Tweaker** leverages `cv2.copyMakeBorder` to artificially inject geometric boundary padding, tricking the neural network by scaling the original zoomed image down to replicate the expected far-away data distribution.
            </p>
          </div>
          <div>
            <h4 style={{ color: '#8b5cf6', fontSize: '1.3rem', marginBottom: '0.5rem' }}>Confidence Thresholding Filtering</h4>
            <p style={{ color: 'var(--text-secondary)', lineHeight: '1.6' }}>
              Because the initial dataset lacked sufficient "Background Negative" examples (photos without targets like birds, streetlamps, etc.), the networks occassionally attempt to blindly attribute abstract background artifacts to the bounding box classifications. The dynamic UI slider forcibly overrides native detection behavior, dropping low-scoring false positives instantly.
            </p>
          </div>
        </div>
      </section>

      {/* Model Breakdown */}
      <section className="doc-section glass-panel" style={{ marginBottom: '3rem' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.8rem', marginBottom: '1.5rem' }}>
          <Network color="#10b981" /> The Four Core Models (Architecture & Specs)
        </h3>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
          <div>
            <h4 style={{ color: '#60a5fa', fontSize: '1.4rem' }}>🟡 YOLOv8s (You Only Look Once v8 Small)</h4>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}><b>Backbone:</b> CSP-Darknet C2f | <b>Neck:</b> PANet | <b>Head:</b> Anchor-free Decoupled | <b>Params:</b> ~11.1M</p>
            <p style={{ color: 'var(--text-secondary)' }}>YOLOv8s acts as the real-time anchor of our system. It perfectly balances detection speed with accurate precision and excels at detecting small feature-sets natively due to the `Mosaic` augmentation parameters injected during training.</p>
          </div>

          <div>
            <h4 style={{ color: '#fb923c', fontSize: '1.4rem' }}>🔴 RT-DETR-L (Real-Time Detection Transformer)</h4>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}><b>Backbone:</b> ResNet-50+HybridEnc | <b>Structure:</b> Transformer Decoder | <b>Params:</b> ~32M</p>
            <p style={{ color: 'var(--text-secondary)' }}>Utilizing visual attention mechanisms, RT-DETR-L theoretically achieves immense mathematical accuracy. <b>Hardware Note:</b> During Apple Silicon (MPS) execution tests, MPS struggles heavily to map Transformer-based memory heuristics. This hardware handicap results in anomalously low raw confidence scoring and severe ~400ms latency bottlenecks that simply do not exist on CUDA architectures.</p>
          </div>

          <div>
            <h4 style={{ color: '#818cf8', fontSize: '1.4rem' }}>🔵 Faster R-CNN (MobileNetV3-Large 320 FPN)</h4>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}><b>Structure:</b> Two-Stage RPN + RoI-Align | <b>Params:</b> ~19.4M</p>
            <p style={{ color: 'var(--text-secondary)' }}>An extremely reliable two-stage detector. Rather than guessing everything at once, Stage 1 generates 2000 region proposals, and Stage 2 rigorously classifies them. While massively slower due to the dual-evaluation pipeline (~200ms), it sweeps complex backgrounds with supreme absolute accuracy.</p>
          </div>

          <div>
            <h4 style={{ color: '#f472b6', fontSize: '1.4rem' }}>🟢 SSD MobileNet V3 (SSDLite-320)</h4>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}><b>Structure:</b> Anchor-Based Single-Shot Head | <b>Params:</b> ~4.5M</p>
            <p style={{ color: 'var(--text-secondary)' }}>The absolute lightest array pipeline targeting pure Mobile/Edge functionality. Running at over `~35 FPS` with a tiny `11 MB` deployment profile, it extracts multi-scale anchors natively without needing a pyramid network, bypassing deep computation to strike targets instantaneously.</p>
          </div>
        </div>

        <div style={{ marginTop: '2.5rem', textAlign: 'center' }}>
          <img src="/assets/fig4_tradeoff.png" alt="Speed vs Accuracy Tradeoff" style={{ maxWidth: '100%', borderRadius: '1rem', border: '1px solid var(--glass-border)' }} />
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.5rem' }}>Figure 1: Tradeoff matrix illustrating Faster R-CNN's accuracy dominance against SSD's extreme velocity.</p>
        </div>
      </section>

      {/* Ensemble Methodology */}
      <section className="doc-section glass-panel" style={{ marginBottom: '3rem' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.8rem', marginBottom: '1.5rem' }}>
          <Database color="#fcd34d" /> Ensemble Methodologies (Multi-Model)
        </h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
          <div>
            <h4 style={{ color: '#fcd34d', fontSize: '1.3rem', marginBottom: '0.5rem' }}>Method 1: WBF (Weighted Box Fusion)</h4>
            <p style={{ color: 'var(--text-secondary)', lineHeight: '1.6' }}>
              Unlike traditional NMS (Non-Maximum Suppression) which outright deletes overlapping lower-confidence geometry targets, WBF utilizes algorithmic math to average overlapping bounding boxes based on mathematically tied confidence multipliers. This constructs a master 'super-box' generated precisely from the positional agreement of all four models—boosting overall mAP reliably!
            </p>
          </div>

          <div>
            <h4 style={{ color: '#00c8ff', fontSize: '1.3rem', marginBottom: '0.5rem' }}>Method 2: Speed-Accuracy Cascade</h4>
            <p style={{ color: 'var(--text-secondary)', lineHeight: '1.6' }}>
              For dynamic edge environments, computing all 4 models continuously annihilates frames per second capabilities. The Cascade pipeline systematically triggers SSD first. If SSD detects a target confidently (`≥0.70`), the pipeline halts natively returning immediately and saving computation! If SSD fails or struggles, logic natively escalates to firing YOLO, then RT-DETR if necessary.
            </p>
          </div>
        </div>

        <div style={{ marginTop: '2.5rem', textAlign: 'center' }}>
          <img src="/assets/ensemble_demo.png" alt="Ensemble Output Display" style={{ maxWidth: '100%', borderRadius: '1rem', border: '1px solid var(--glass-border)' }} />
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.5rem' }}>Figure 2: Standalone outputs mapping individually compared to the fully fused WBF ensemble execution.</p>
        </div>
      </section>

      {/* Full Statistical Summary */}
      <section className="doc-section glass-panel">
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.8rem', marginBottom: '1.5rem' }}>
          <Cpu color="#f87171" /> Post-Evaluation Metrics
        </h3>
        <p style={{ color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: '2rem' }}>
          Based on systematic GPU batch trials resolving multiple architectures natively against identically labeled test environments (`~600 test samples`):
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
          <div style={{ textAlign: 'center' }}>
            <img src="/assets/fig1_latency.png" alt="Latency Comparison" style={{ width: '100%', borderRadius: '1rem', border: '1px solid var(--glass-border)', objectFit: 'contain' }} />
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.5rem' }}>Figure 3: System Latency Distribution Metrics.</p>
          </div>
          <div style={{ textAlign: 'center' }}>
            <img src="/assets/fig6_map.png" alt="mAP Metrics" style={{ width: '100%', borderRadius: '1rem', border: '1px solid var(--glass-border)', objectFit: 'contain' }} />
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.5rem' }}>Figure 4: Global mAP@0.5 Benchmarks Comparison.</p>
          </div>
        </div>
      </section>

    </div>
  );
};

export default Documentation;
