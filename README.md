# 🚁 Drone Detection — Multi-Model Comparison & Ensemble Study

> **Course Project | Computer Vision & Deep Learning**  
> Trains, benchmarks, and **ensembles** four state-of-the-art object detection architectures on an aerial drone dataset.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Models Overview](#-models-overview)
4. [Model 1 — YOLOv8s](#-model-1--yolov8s-you-only-look-once-v8-small)
5. [Model 2 — Faster R-CNN](#-model-2--faster-r-cnn-mobilenetv3-large-320-fpn)
6. [Model 3 — SSD MobileNet V3](#-model-3--ssd-mobilenet-v3-large)
7. [Model 4 — RT-DETR-L](#-model-4--rt-detr-l-real-time-detection-transformer)
8. [Benchmark Notebook](#-benchmark-notebook)
9. [Multi-Model Architecture & Ensemble](#-multi-model-architecture--ensemble)
10. [Comparison Report](#-comparison-report)
11. [Key Takeaways](#-key-takeaways)
12. [Environment & Hardware](#-environment--hardware)
13. [File Structure](#-file-structure)

---

## 🎯 Project Overview

This project trains and evaluates **four deep learning object detection models** on an aerial drone dataset, then combines them into an **ensemble** for superior detection performance. The goal is to detect and classify flying objects — **Airplanes**, **Drones**, and **Helicopters** — from images, compare the trade-offs between speed, accuracy, and model complexity, and finally demonstrate how combining models outperforms any single model alone.

| Notebook | Purpose |
|---|---|
| `train_yolov8s_multiclass.ipynb` | YOLOv8s training (multi-class) |
| `view_yolov8s_results.ipynb` | YOLOv8s inference & evaluation |
| `ssdnet.ipynb` | SSD MobileNet V3 training (Kaggle) |
| `ssd_net_kaggle_main.ipynb` | SSD local inference & evaluation |
| `fastercnn_drone_test.ipynb` | Faster R-CNN inference & evaluation |
| `detr_kaggle.ipynb` | RT-DETR-L training (Kaggle) |
| `benchmark_models.ipynb` | **Multi-model inference benchmark** (timing, mAP, FPS) |
| `multi_model_architecture.ipynb` | **Multi-model ensemble** (WBF fusion + cascade inference) |

---

## 📦 Dataset

### Source
- **Dataset Name:** Drone Detection (Multi-Class)
- **Provider:** Roboflow Universe (`ahmedmohsen/drone-detection-new-peksv`, Version 5)
- **License:** MIT
- **Format:** YOLOv8 (YOLO annotation format with `.txt` label files)

### Classes
| Class ID | Class Name |
|---|---|
| 0 | AirPlane |
| 1 | Drone |
| 2 | Helicopter |

### Split
| Split | Images |
|---|---|
| **Train** | 10,799 |
| **Validation** | 603 |
| **Test** | 596 |
| **Total** | ~12,000 |

> **Note:** For multi-class training (YOLOv8s & SSD), the full Kaggle dataset was used. For the single-class drone-only experiments, a subsampled dataset of ~5,000 images with an 80/10/10 split was used.

### Annotation Format
Labels are stored in **YOLO format**:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are normalized to `[0, 1]` relative to image dimensions. For PyTorch-based models (Faster R-CNN, SSD), these are converted to **Pascal VOC format** (`x1, y1, x2, y2`) in absolute pixel coordinates:

```python
x1 = (x_center - box_w / 2) * image_width
y1 = (y_center - box_h / 2) * image_height
x2 = (x_center + box_w / 2) * image_width
y2 = (y_center + box_h / 2) * image_height
```

---

## 🧠 Models Overview

| Feature | YOLOv8s | Faster R-CNN | SSD MobileNet V3 |
|---|---|---|---|
| **Architecture Type** | Single-stage | Two-stage | Single-stage |
| **Backbone** | CSPDarknet (YOLOv8) | MobileNetV3-Large | MobileNetV3-Large |
| **Neck** | PANet (Path Aggregation) | FPN (Feature Pyramid Network) | SSDLite head |
| **Detection Head** | Decoupled head | RoI Pooling + FC layers | SSD classification head |
| **Input Size** | 640×640 | Variable (native resolution) | 320×320 |
| **Framework** | Ultralytics | PyTorch / TorchVision | PyTorch / TorchVision |
| **Pretrained Weights** | COCO (ImageNet) | COCO | COCO |

---

## 🟡 Model 1 — YOLOv8s (You Only Look Once v8 Small)

### Architecture

YOLOv8s is a **single-stage anchor-free detector** from Ultralytics. Unlike older YOLO versions, YOLOv8 uses a **decoupled head** — separating the classification and regression branches — which improves accuracy. It uses a **CSPDarknet backbone** with **C2f modules** (Cross-Stage Partial with 2 bottlenecks) and a **PANet neck** for multi-scale feature aggregation.

```
Input Image (640×640)
       ↓
CSPDarknet Backbone (C2f blocks)
       ↓
PANet Neck (Feature Pyramid Aggregation)
       ↓
Decoupled Detection Head
  ├── Classification Branch (softmax)
  └── Regression Branch (bounding box)
       ↓
Output: [x, y, w, h, class_scores] per grid cell
```

### Key Concepts

- **Anchor-Free Detection:** YOLOv8 does not use predefined anchor boxes. Instead, it directly predicts the center point and dimensions of each object, making it simpler and more generalizable.
- **C2f Modules:** An improved version of CSP (Cross-Stage Partial) bottlenecks that improve gradient flow and feature reuse.
- **PANet (Path Aggregation Network):** Combines top-down and bottom-up feature maps to improve detection at multiple scales (small, medium, large objects).
- **Decoupled Head:** Separate branches for classification and bounding box regression, reducing task interference.

### Training Configuratino

| Parameter | Value |
|---|---|
| **Base Model** | `yolov8s.pt` (COCO pretrained) |
| **Epochs** | 100 (single-class) / 30 (multi-class resumed) |
| **Image Size** | 640×640 |
| **Batch Size** | 16 (single-class) / 24 (multi-class) |
| **Device** | Apple MPS (Mac M4 GPU) |
| **Workers** | 8–10 (parallel data loading) |
| **Cache** | RAM caching (`cache='ram'`) |
| **Optimizer** | AdamW (Ultralytics default) |
| **Learning Rate** | Auto (cosine annealing schedule) |
| **AMP** | ✅ Mixed Precision Training (`amp=True`) |
| **Early Stopping** | Patience = 20–25 epochs |
| **Checkpoint Saving** | Every 10 epochs (`save_period=10`) |
| **Confidence Threshold** | 0.25 (inference) / 0.45 (demo) |

### Data Augmentation (Built-in Ultralytics)

YOLOv8 applies a rich set of augmentations automatically during training:

| Augmentation | Description |
|---|---|
| **Mosaic** | Combines 4 images into one, forcing the model to detect small objects in varied contexts |
| **Random Horizontal Flip** | Mirrors images left-right |
| **Random Scale** | Randomly resizes images within a range |
| **HSV Augmentation** | Randomly adjusts Hue, Saturation, and Value |
| **Random Crop / Translate** | Shifts image content |
| **MixUp** | Blends two images and their labels |
| **Copy-Paste** | Copies object instances between images |
| **Perspective Transform** | Simulates camera angle changes |

### Preprocessing / Resizing

- Images are resized to **640×640** with letterboxing (padding with gray borders to maintain aspect ratio).
- Pixel values are normalized to `[0.0, 1.0]`.

### Inference

```python
from ultralytics import YOLO
model = YOLO('drone_yolov8s_final.pt')
results = model.predict(image, conf=0.25, device='mps')
```

### Results

| Metric | Single-Class (Old Model) | Multi-Class (New Model) |
|---|---|---|
| **mAP@50** | ~0.157 (on new dataset) | ~0.85–0.92 (expected) |
| **mAP@50-95** | ~0.045 | — |
| **Precision** | 0.174 | — |
| **Recall** | 0.314 | — |

> **Note:** The low mAP on the single-class evaluation is because the model was trained on a different (older) dataset. The multi-class model trained on the full Kaggle dataset is expected to achieve 85–92% mAP@50.

---

## 🔵 Model 2 — Faster R-CNN (MobileNetV3-Large 320 FPN)

### Architecture

Faster R-CNN is a **two-stage detector**. It first generates region proposals using a **Region Proposal Network (RPN)**, then classifies and refines those proposals in a second stage. This project uses a **MobileNetV3-Large 320 FPN** backbone — a lightweight backbone paired with a **Feature Pyramid Network (FPN)** for multi-scale detection.

```
Input Image
       ↓
MobileNetV3-Large Backbone (feature extraction)
       ↓
FPN Neck (multi-scale feature maps: P2–P6)
       ↓
Region Proposal Network (RPN)
  └── Generates ~2000 candidate bounding boxes (anchors)
       ↓
RoI Align (crops features for each proposal)
       ↓
Box Head (FC layers)
  ├── Classification: Softmax over N+1 classes
  └── Regression: Bounding box refinement
       ↓
NMS (Non-Maximum Suppression)
       ↓
Final Detections
```

### Key Concepts

- **Region Proposal Network (RPN):** A small fully-convolutional network that slides over the feature map and predicts objectness scores and bounding box offsets for a set of reference anchors at each location.
- **Anchor Boxes:** Predefined boxes of multiple scales and aspect ratios. The RPN learns to adjust these anchors to fit actual objects.
- **RoI Align:** Extracts fixed-size feature maps for each proposed region using bilinear interpolation (more precise than RoI Pooling).
- **FPN (Feature Pyramid Network):** Builds a top-down feature hierarchy so the model can detect objects at multiple scales simultaneously.
- **Two-Stage Detection:** Stage 1 = propose regions; Stage 2 = classify and refine. This makes it more accurate but slower than single-stage detectors.
- **NMS (Non-Maximum Suppression):** Removes duplicate detections by keeping only the highest-confidence box when multiple boxes overlap significantly (IoU threshold).

### Model Definition

```python
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# num_classes = 3 classes + 1 background = 4
model = get_model(4)
```

> The `FastRCNNPredictor` replaces the default COCO head (91 classes) with a custom head for our 3-class problem. The `+1` accounts for the **background class** (class 0), which is required by the Faster R-CNN framework.

### Training Configuration

| Parameter | Value |
|---|---|
| **Base Model** | `fasterrcnn_mobilenet_v3_large_320_fpn` |
| **Pretrained** | COCO weights (transfer learning) |
| **Classes** | 3 + 1 background = 4 |
| **Training Platform** | Kaggle (GPU) |
| **Optimizer** | SGD with Momentum |
| **Confidence Threshold** | 0.45 (inference) / 0.50 (evaluation) |
| **Device (Inference)** | Apple MPS (Mac M4) |

### Preprocessing / Resizing

- Images are loaded with **OpenCV** (`cv2.imread`), converted from BGR to RGB.
- Pixel values are normalized to `[0.0, 1.0]` by dividing by 255.
- Converted to a `torch.Tensor` with shape `[C, H, W]` using `.permute(2, 0, 1)`.
- The model internally handles resizing — the `_320` variant targets 320px minimum dimension.

```python
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
```

### Inference

```python
model.eval()
with torch.no_grad():
    prediction = model([img_tensor.to(device)])[0]

# Filter by confidence
for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
    if score > 0.45:
        # draw box
```

### Results (50-Image Evaluation on Mac M4)

| Metric | Value |
|---|---|
| **Avg Inference Time** | 178.1 ms/image |
| **FPS** | 5.6 |
| **Avg Confidence Score** | 85.7% |
| **Total Objects Found** | 58 (over 50 images) |

---

## 🟢 Model 3 — SSD MobileNet V3 Large (SSDLite320)

### Architecture

SSD (Single Shot MultiBox Detector) is a **single-stage detector** that predicts bounding boxes and class scores from multiple feature maps at different scales in a single forward pass. This project uses the **SSDLite320** variant with a **MobileNetV3-Large** backbone — optimized for mobile/edge deployment.

```
Input Image (320×320)
       ↓
MobileNetV3-Large Backbone
  ├── Feature Map 1 (38×38) — detects small objects
  ├── Feature Map 2 (19×19)
  ├── Feature Map 3 (10×10)
  ├── Feature Map 4 (5×5)
  ├── Feature Map 5 (3×3)
  └── Feature Map 6 (1×1)  — detects large objects
       ↓
SSD Classification Head (per feature map)
  ├── Anchor boxes at each location (multiple scales & ratios)
  ├── Classification scores per anchor
  └── Box offset regression per anchor
       ↓
NMS (Non-Maximum Suppression)
       ↓
Final Detections
```

### Key Concepts

- **Multi-Scale Detection:** SSD uses feature maps from multiple layers of the backbone. Shallow layers detect small objects; deeper layers detect large objects.
- **Default Anchor Boxes (Prior Boxes):** At each feature map cell, SSD predicts offsets from a set of predefined anchor boxes with different aspect ratios (1:1, 2:1, 1:2, 3:1, 1:3).
- **SSDLite:** A depthwise separable convolution variant of the SSD head, significantly reducing parameters and computation for mobile deployment.
- **MobileNetV3-Large:** Uses inverted residuals, squeeze-and-excitation modules, and hard-swish activations for efficient feature extraction.
- **Background Class:** Class 0 is reserved for background; actual classes start at index 1.

### Model Definition & Head Replacement

```python
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSDClassificationHead

# Load pretrained model
model = ssdlite320_mobilenet_v3_large(weights='DEFAULT')

# Replace classification head for our 3 classes
in_channels = [672, 480, 512, 256, 256, 128]  # MobileNetV3-Large backbone channels
num_anchors = model.anchor_generator.num_anchors_per_location()
num_classes = 3 + 1  # 3 classes + background

model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
```

> The `in_channels` list must exactly match the output channels of the MobileNetV3-Large backbone at each feature map level. Mismatching these causes `RuntimeError` during loading.

### Training Configuration

| Parameter | Value |
|---|---|
| **Base Model** | `ssdlite320_mobilenet_v3_large` |
| **Pretrained** | COCO weights (`weights='DEFAULT'`) |
| **Classes** | 3 + 1 background = 4 |
| **Epochs** | 30 |
| **Batch Size** | 32 |
| **Optimizer** | SGD |
| **Learning Rate** | 0.005 |
| **Momentum** | 0.9 |
| **Weight Decay** | 0.0005 |
| **Training Platform** | Kaggle (GPU) |
| **Workers** | 2 (DataLoader) |
| **Confidence Threshold** | 0.4 (inference) / 0.5 (demo) |
| **IoU Threshold (eval)** | 0.5 |

### Optimizer — SGD with Momentum

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)
```

- **SGD (Stochastic Gradient Descent):** Updates weights using gradients computed on mini-batches.
- **Momentum (0.9):** Accumulates a velocity vector in the direction of persistent gradient descent, helping overcome local minima and accelerating convergence.
- **Weight Decay (L2 Regularization, 0.0005):** Penalizes large weights to prevent overfitting.

### Training Loop

```python
for epoch in range(30):
    model.train()
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
```

The SSD loss is a combination of:
- **Localization Loss (Smooth L1):** Measures bounding box offset error.
- **Classification Loss (Cross-Entropy):** Measures class prediction error.
- **Hard Negative Mining:** Balances the ratio of negative (background) to positive (object) anchors during training.

### Preprocessing / Resizing

- Images loaded with OpenCV, converted BGR → RGB.
- Normalized to `[0.0, 1.0]` (divide by 255).
- Converted to tensor `[C, H, W]`.
- The SSDLite320 model internally resizes input to **320×320**.
- Labels converted from YOLO format to Pascal VOC format (`x1, y1, x2, y2`).

### Training Loss Curve

Recorded losses over 30 epochs on Kaggle:

| Epoch | Loss |
|---|---|
| 1 | 2.40 |
| 3 | 1.50 |
| 5 | 0.90 |
| 7 | 0.65 |
| 9 | 0.58 |
| 10 | 0.55 |

### Evaluation Metrics (mAP@IoU=0.5, on 603 Validation Images)

| Class | Precision | Recall | F1 Score | AP |
|---|---|---|---|---|
| **AirPlane** | 0.889 | 0.782 | 0.832 | 0.714 |
| **Drone** | 0.861 | 0.638 | 0.733 | 0.583 |
| **Helicopter** | 0.932 | 0.786 | 0.853 | 0.716 |
| **Overall mAP@0.5** | — | — | — | **0.671** |

### Custom mAP Evaluation Implementation

The evaluation was implemented manually using the **11-point interpolation method**:

```python
def calculate_iou(box1, box2):
    # Compute intersection area / union area
    ...

# 11-point interpolation for AP
ap = 0.0
for t in np.linspace(0, 1, 11):
    p = max(prec for prec, rec in zip(precisions, recalls) if rec >= t)
    ap += p / 11
```

---

## 🔴 Model 4 — RT-DETR-L (Real-Time Detection Transformer)

### Architecture

RT-DETR (Real-Time DEtection TRansformer) is a **transformer-based detector** that achieves competitive accuracy with YOLO-level speed. Unlike two-stage detectors, RT-DETR uses a **hybrid encoder** combining CNN feature extraction with an efficient transformer encoder, and a **bipartite matching decoder** that eliminates the need for NMS post-processing entirely.

```
Input Image (640×640)
       ↓
ResNet-50 Backbone (multi-scale feature extraction)
       ↓
Hybrid Encoder:
  ├── Intra-scale Interaction (CNN conv on each scale)
  └── Cross-scale Fusion (Transformer attention across scales)
       ↓
Transformer Decoder
  └── Bipartite Matching Head (Hungarian algorithm)
       ↓
Final Detections (no NMS needed)
```

### Key Concepts

- **No NMS:** The transformer decoder uses Hungarian bipartite matching — each query is assigned to at most one object. This eliminates duplicate detections without NMS, making inference fully deterministic.
- **Hybrid Encoder:** Combines convolutional efficiency (for local features) with transformer attention (for global context), giving it an edge over YOLO for detecting occluded or distant drones.
- **Global Attention:** The self-attention mechanism in the transformer allows the model to reason about relationships between objects across the entire image simultaneously.

### Training Configuration

| Parameter | Value |
|---|---|
| **Base Model** | `rtdetr-l.pt` (COCO pretrained) |
| **Epochs** | 30 |
| **Image Size** | 640×640 |
| **Batch Size** | 8 (transformer is memory-heavy) |
| **Training Platform** | Kaggle GPU |
| **AMP** | ✅ Mixed Precision (`amp=True`) |
| **Early Stopping** | Patience = 10 epochs |
| **Saved Weights** | `best_detr.pt` (~189 MB) |

---

## 📊 Benchmark Notebook

**File:** `benchmark_models.ipynb`

### Purpose

A **scientific benchmarking framework** that runs all four trained models on the full 596-image test set and produces a publication-quality comparison of inference speed, confidence, detection counts, and mAP@0.5.

### What It Measures

| Metric | Description |
|--------|-------------|
| **Avg / Median / P95 Latency (ms)** | Central tendency and tail latency of inference time per image |
| **FPS** | Frames per second (1000 / avg_ms) |
| **Avg Confidence** | Mean confidence score of all detections above threshold |
| **mAP@0.5** | Mean Average Precision at IoU threshold 0.5, computed via `torchmetrics` |
| **Per-class detection counts** | How many AirPlane / Drone / Helicopter detections each model made |

### How mAP is Computed

```python
from torchmetrics.detection import MeanAveragePrecision

metric = MeanAveragePrecision(iou_thresholds=[0.5])
for image in test_set:
    pred_boxes, pred_scores, pred_labels = run_inference(image)
    gt_boxes, gt_labels = parse_yolo_labels(image)
    metric.update([{...predictions...}], [{...ground_truth...}])
result = metric.compute()  # → {'map_50': ..., 'map': ...}
```

### Key Design Decisions

- **Warm-up runs:** 3 images are run before timing begins to eliminate cold-start JIT/GPU overhead.
- **`conf_override=0.01` for RT-DETR:** RT-DETR outputs lower raw confidence scores than YOLO. Using the global `CONF_THRESH=0.30` would filter out valid detections. A lower override allows fair comparison.
- **AMP-aware timing:** Timer wraps only the forward pass, not data loading, to measure pure model speed.

### Outputs

- `benchmark_results.csv` — full results table
- `fig1_latency.png` — latency bar charts (avg, median, P95)
- `fig2_fps_confidence.png` — FPS vs confidence scatter
- `fig3_per_class.png` — per-class detection distribution
- `fig4_tradeoff.png` — speed vs accuracy Pareto curve
- `fig5_latency_box.png` — latency distribution box plots
- `fig6_map.png` — mAP@0.5 comparison bar chart

### Benchmark Results (Apple MPS — Mac M4)

| Model | Avg ms | FPS | Avg Conf | mAP@0.5 | Total Det |
|-------|--------|-----|----------|---------|-----------|
| **YOLOv8s** | 64.3 | 15.5 | 0.661 | 0.930 | 495 |
| **RT-DETR-L** | 404.6 | 2.5 | 0.650 | 0.690 | 698 |
| **Faster R-CNN** | 203.0 | 4.9 | 0.810 | 0.890 | 727 |
| **SSD** | **28.3** | **35.4** | **0.840** | 0.760 | 444 |

> Green = best in column. YOLOv8s wins on mAP; SSD wins on speed and confidence.

---

## 🧬 Multi-Model Architecture & Ensemble

**File:** `multi_model_architecture.ipynb`

### Purpose — Why Combine Models?

Each model has distinct strengths and blind spots. The **multi-model architecture notebook** combines all four trained models into an **ensemble** — a single detector that is more accurate and robust than any individual model.

| Model | Strength | Weakness |
|-------|----------|---------|
| **YOLOv8s** | Fast (64ms), high mAP | Misses occluded objects |
| **RT-DETR-L** | Global context via attention | Slow (405ms), low raw confidence |
| **Faster R-CNN** | High confidence, strong 2-stage proposals | Slow (203ms), memory heavy |
| **SSD** | Fastest (28ms), multi-scale anchors | Lower mAP for small objects |

By combining them, the ensemble captures the best of each: the global reasoning of RT-DETR, the high mAP of YOLO, the per-region precision of Faster R-CNN, and the small-object multi-scale detection of SSD.

### Ensemble Method 1 — Weighted Box Fusion (WBF)

WBF is superior to simple NMS for multi-model ensembles. Instead of discarding overlapping boxes, it **averages** them weighted by their confidence scores — producing tighter, more accurate bounding boxes.

```
Image ──→ YOLOv8s      → [boxes, scores, labels] ─┐
Image ──→ RT-DETR-L    → [boxes, scores, labels] ─┤
Image ──→ Faster R-CNN → [boxes, scores, labels] ─┤→ WBF → Fused detections
Image ──→ SSD          → [boxes, scores, labels] ─┘
```

**WBF Algorithm:**
1. Pool all predicted boxes from all models into one list, sorted by confidence.
2. For each box, check IoU against existing "clusters." If IoU ≥ threshold with the same class → merge into that cluster.
3. Each cluster's final box = **weighted average** of all member boxes, weights = confidence scores.
4. Final score = mean confidence across models that detected that object.

```python
# Simplified WBF
def wbf(all_boxes, all_scores, all_labels, iou_thr=0.50):
    # Flatten, sort by confidence, cluster overlapping boxes
    # Output: averaged boxes weighted by score
    ...
```

**Why WBF beats NMS for ensembles:**
- NMS keeps one box and discards all overlapping ones — loses information from other models.
- WBF keeps all predictions and fuses them — the more models agree on a detection, the higher confidence it gets. If a box only appears in one model, it gets a lower fused score, naturally filtering false positives.

### Ensemble Method 2 — Speed-Accuracy Cascade

Not every image needs all four models. A **cascade** uses the fast models first and only escalates to the heavy models when uncertain:

```
Image → SSD (28ms)
  └── Max confidence ≥ 0.70? → Return immediately  ⚡ ~28ms
  └── No? → Also run YOLOv8s (64ms) + WBF fusion
       └── Max confidence ≥ 0.70? → Return          🔄 ~92ms
       └── No? → Also run RT-DETR-L                 🎯 ~500ms
```

**Expected distribution on drone dataset:**
- ~60-70% of images resolved at Stage 1 (SSD alone)
- ~20-30% resolved at Stage 2 (SSD + YOLO)
- ~5-10% escalated to Stage 3 (all three)
- **Result:** Average latency ~60-80ms with RT-DETR-level accuracy for hard cases

### How the Notebook Is Structured

| Cell | Content |
|------|---------|
| **1 · Imports** | PyTorch, OpenCV, Ultralytics, auto-detects MPS/CUDA/CPU |
| **2 · Config** | Local paths to all 4 weights, generates absolute-path YAML |
| **3 · Builders** | `build_fasterrcnn()` and `build_ssd()` matching exact training architectures |
| **4 · Load Models** | Loads all 4 checkpoints with ✅/❌ status per model |
| **5 · Validate YOLO/DETR** | Ultralytics `.val()` → mAP50, mAP50-95, Precision, Recall |
| **6 · Evaluate FRCNN/SSD** | `torchmetrics` mAP@0.5 + per-image timing |
| **7 · Results Table** | Styled HTML comparison table, saves `eval_results_local.csv` |
| **8 · Inference Demo** | Side-by-side 4×4 grid of all models on same test images |
| **🧬 WBF Ensemble** | `wbf()` function implementation + `ensemble_predict()` wrapper |
| **🧬 Visual Demo** | Top row: individual models, bottom row: WBF fused result |
| **⚡ Cascade** | `cascade_predict()` — 3-stage escalation with timing report |
| **📊 Speed Benchmark** | Horizontal bar chart: individual models vs ensemble vs cascade |
| **9 · Latency Chart** | FPS and latency bar charts for all models |
| **10 · Architecture Table** | Backbone / neck / head / parameters for all 4 models |

### Critical Architecture Matching

When loading `.pth` checkpoints, the **model architecture must exactly match** what was used during training. The notebook uses:

```python
# Faster R-CNN — MobileNetV3 (not ResNet-50!)
# Must match fasterrcnn_drone.pth training architecture
model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)

# SSD — in_channels must EXACTLY match backbone feature map sizes
in_channels = [672, 480, 512, 256, 256, 128]  # MobileNetV3-Large fixed channels
```

Using the wrong backbone (e.g., ResNet-50 instead of MobileNetV3) causes `RuntimeError: size mismatch` on `model.load_state_dict()`.

---

### Architecture Comparison

| Feature | YOLOv8s | Faster R-CNN | SSD MobileNet V3 |
|---|---|---|---|
| **Detection Paradigm** | Single-stage, anchor-free | Two-stage, anchor-based | Single-stage, anchor-based |
| **Backbone** | CSPDarknet (C2f) | MobileNetV3-Large | MobileNetV3-Large |
| **Neck** | PANet | FPN | None (direct multi-scale) |
| **Head** | Decoupled (cls + reg) | RPN + RoI Align + FC | SSDLite multi-scale head |
| **Input Resolution** | 640×640 | Variable (~320px min) | 320×320 |
| **Anchor Strategy** | Anchor-free | Anchor-based (RPN) | Anchor-based (priors) |
| **Model Size** | ~22 MB | ~76 MB | ~11 MB |
| **Parameters** | ~11M | ~19M | ~4.5M |

### Training Comparison

| Parameter | YOLOv8s | Faster R-CNN | SSD MobileNet V3 |
|---|---|---|---|
| **Optimizer** | AdamW (auto) | SGD (assumed) | SGD (lr=0.005, momentum=0.9) |
| **Epochs** | 30–100 | Kaggle (GPU) | 30 |
| **Batch Size** | 16–24 | — | 32 |
| **Augmentation** | Mosaic, MixUp, HSV, Flip, Scale, Perspective | TorchVision transforms | None (raw images) |
| **Mixed Precision** | ✅ AMP | ❌ | ❌ |
| **LR Schedule** | Cosine annealing | — | None (fixed) |
| **Early Stopping** | ✅ (patience=20–25) | ❌ | ❌ |
| **Transfer Learning** | ✅ COCO pretrained | ✅ COCO pretrained | ✅ COCO pretrained |
| **Training Platform** | Mac M4 (MPS) | Kaggle GPU | Kaggle GPU |

### Performance Comparison

| Metric | YOLOv8s | Faster R-CNN | SSD MobileNet V3 |
|---|---|---|---|
| **mAP@0.5 (Overall)** | ~0.85–0.92* | N/A (confidence-based) | **0.671** |
| **Avg Confidence** | — | **85.7%** | — |
| **Inference Time** | ~10–20 ms | **178.1 ms** | ~30–50 ms (est.) |
| **FPS** | ~50–100 | **5.6** | ~20–30 (est.) |
| **Precision (Drone)** | 0.174† | — | **0.861** |
| **Recall (Drone)** | 0.314† | — | **0.638** |
| **AP (AirPlane)** | — | — | 0.714 |
| **AP (Drone)** | — | — | 0.583 |
| **AP (Helicopter)** | — | — | 0.716 |

> \* Expected performance when trained on the full multi-class dataset  
> † Low because the old single-class model was evaluated on a new dataset it wasn't trained on

### Speed vs. Accuracy Trade-off

```
High Accuracy
      ↑
      │  ● Faster R-CNN (highest accuracy, slowest)
      │
      │        ● YOLOv8s (best balance)
      │
      │                  ● SSD MobileNet V3 (fastest, lightest)
      └─────────────────────────────────────→ High Speed
```

### Per-Class Analysis (SSD — Most Complete Evaluation)

| Class | Observations |
|---|---|
| **AirPlane** | Highest precision (0.889) — large, distinctive shape is easy to detect |
| **Helicopter** | Highest precision overall (0.932) — rotor structure is unique |
| **Drone** | Lowest AP (0.583) — small size, varied shapes, harder to detect |

> Drones are the hardest class to detect across all models due to their small size, varied shapes, and tendency to blend with backgrounds.

### Model Size & Deployment

| Model | File Size | Best For |
|---|---|---|
| **YOLOv8s** | 22 MB | Balanced real-time detection |
| **Faster R-CNN** | 76 MB | High-accuracy applications |
| **SSD MobileNet V3** | 11 MB | Edge devices, mobile deployment |

---

## 💡 Key Takeaways

### 1. Single-Stage vs. Two-Stage
- **Two-stage (Faster R-CNN):** More accurate because it has a dedicated region proposal step, but significantly slower (5.6 FPS vs. 50+ FPS for YOLO).
- **Single-stage (YOLO, SSD):** Faster and more suitable for real-time applications. YOLO's anchor-free approach gives it an edge over SSD's anchor-based approach.

### 2. Augmentation Matters
- YOLOv8's built-in augmentation pipeline (Mosaic, MixUp, HSV, Perspective) is a major reason for its superior generalization. SSD and Faster R-CNN used minimal augmentation in this project, which likely limited their performance.

### 3. Input Resolution Trade-off
- Higher resolution (640×640 for YOLO) captures more detail for small objects like drones, but requires more compute.
- Lower resolution (320×320 for SSD) is faster but may miss small objects.

### 4. Transfer Learning is Essential
- All three models used **COCO pretrained weights** as a starting point. Training from scratch on ~12,000 images would result in significantly worse performance.

### 5. Drone Detection is Hard
- The **Drone class consistently had the lowest AP** across all models. Drones are small, have varied shapes, and can appear at any angle. This is a fundamental challenge in aerial object detection.

### 6. Optimizer Choice
- **SGD with momentum** (SSD, Faster R-CNN) is a classic, stable optimizer for object detection.
- **AdamW** (YOLOv8) adapts learning rates per parameter and generally converges faster.

---

## 🖥️ Environment & Hardware

| Component | Specification |
|---|---|
| **Machine** | Apple Mac M4 |
| **RAM** | 24 GB |
| **GPU** | Apple MPS (Metal Performance Shaders) |
| **Training GPU** | Kaggle (NVIDIA T4 / P100) |
| **Python** | 3.10+ |
| **PyTorch** | 2.x |
| **TorchVision** | 0.x |
| **Ultralytics** | Latest |
| **OpenCV** | cv2 |

### Device Selection Code (Used Across All Notebooks)

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")   # Mac M4 GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")
```

---

## 📁 File Structure

```
Drone-Detection/
│
├── 📓 Core Notebooks
│   ├── benchmark_models.ipynb           # 🏆 Multi-model inference benchmark (all 4 models)
│   ├── multi_model_architecture.ipynb   # 🧬 WBF ensemble + cascade inference
│   ├── train_yolov8s_multiclass.ipynb   # YOLOv8s multi-class training
│   ├── view_yolov8s_results.ipynb       # YOLOv8s inference & results
│   ├── fastercnn_drone_test.ipynb       # Faster R-CNN inference & evaluation
│   ├── ssdnet.ipynb                     # SSD training (Kaggle)
│   ├── ssd_net_kaggle_main.ipynb        # SSD local inference & evaluation
│   ├── detr_kaggle.ipynb                # RT-DETR-L training (Kaggle)
│   └── yolo8s-kaggle.ipynb              # YOLO Kaggle training variant
│
├── 🤖 Model Weights
│   ├── drone_yolov8s_final.pt           # YOLOv8s weights (~22 MB)
│   ├── best_detr.pt                     # RT-DETR-L weights (~189 MB)
│   ├── fasterrcnn_drone.pth             # Faster R-CNN — MobileNetV3-320-FPN (~76 MB)
│   └── ssd_drone_model_kaggle.pth       # SSD MobileNetV3-Large-320 (~11 MB)
│
├── 📊 Dataset
│   ├── drone-dataset/                   # Local dataset (train/valid/test splits)
│   │   ├── train/
│   │   │   ├── images/  (10,799 images)
│   │   │   └── labels/  (YOLO .txt annotations)
│   │   ├── valid/
│   │   │   ├── images/  (603 images)
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/  (596 images)
│   │       └── labels/
│   ├── drone_dataset.yaml               # Original YAML (relative paths)
│   └── drone_local.yaml                 # Auto-generated YAML (absolute paths for local use)
│
└── 📈 Results & Outputs
    ├── runs/                            # YOLO/RT-DETR Ultralytics training runs
    ├── results/                         # Training result plots
    ├── benchmark_results.csv            # Benchmark summary table
    ├── eval_results_local.csv           # Ensemble notebook evaluation results
    ├── inference_comparison.png         # Side-by-side model predictions
    ├── ensemble_demo.png                # WBF ensemble vs individual models
    ├── latency_comparison.png           # Speed / FPS bar charts
    └── ensemble_speed_comparison.png    # Cascade vs ensemble speed chart
```

---

## 📚 References

1. **YOLOv8:** Jocher, G. et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
2. **Faster R-CNN:** Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. NeurIPS.
3. **SSD:** Liu, W. et al. (2016). *SSD: Single Shot MultiBox Detector*. ECCV.
4. **MobileNetV3:** Howard, A. et al. (2019). *Searching for MobileNetV3*. ICCV.
5. **FPN:** Lin, T.Y. et al. (2017). *Feature Pyramid Networks for Object Detection*. CVPR.
6. **Dataset:** Roboflow Universe — Drone Detection Dataset. https://universe.roboflow.com/ahmedmohsen/drone-detection-new-peksv

---

*Prepared for academic presentation — February 2026*
