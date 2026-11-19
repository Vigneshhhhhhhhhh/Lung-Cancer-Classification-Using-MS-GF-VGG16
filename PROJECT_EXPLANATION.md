# Project Documentation: Lung Cancer Classification Using MS-GF-VGG16

## Multi-Scale Gated Feature Fusion VGG16 for Pulmonary Histopathology Triaging from Computed Tomography (CT) Imaging

---

## 1. Executive Summary

This academic major capstone project introduces an end-to-end, Computer-Aided Diagnosis (CAD) system named **MS-GF-VGG16** (*Multi-Scale Gated Feature Fusion VGG16*). 

The primary goal of the system is the automated, fine-grained classification of chest computed tomography (CT) scans into four distinct clinical categories:
1. **Adenocarcinoma** (The most prevalent Non-Small Cell Lung Carcinoma [NSCLC])
2. **Large Cell Carcinoma** (An aggressive, fast-growing, undifferentiated NSCLC)
3. **Squamous Cell Carcinoma** (An airway-lining bronchial NSCLC)
4. **Normal** (Non-malignant, healthy pulmonary parenchyma)

Unlike standard convolutional neural networks that extract features solely from the deepest layer of a backbone, MS-GF-VGG16 incorporates a novel, trainable **Scale-Gated Feature Fusion Unit (SGFU)**. This unit dynamically learns continuous importance weights across three hierarchical feature stages (shallow textures, intermediate morphology, and deep semantics) of an ImageNet-pretrained VGG16 backbone.

---

## 2. Clinical Context and Problem Statement

### 2.1 The Clinical Challenge
Pulmonary neoplasia remains the leading cause of cancer-related mortality across the globe. Low-dose Computed Tomography (LDCT) imaging enables non-invasive visualization of lung parenchyma; however, distinguishing between non-small cell lung cancer subtypes—specifically **Adenocarcinoma** versus **Squamous Cell Carcinoma**—presents a formidable diagnostic obstacle for radiologists:
- Both histological subtypes share similar radiographic attenuation (Hounsfield Units).
- Both exhibit overlapping peripheral spicular margins, ground-glass opacities, and variable solid components.
- Standard physical biopsies carry notable procedural risks, including pneumothorax, pulmonary hemorrhage, and sampling error.

An automated, explainable deep learning classifier assists oncologists, thoracic surgeons, and radiologists by providing an objective, rapid second opinion directly from standard axial CT cross-sections.

### 2.2 Why Standard VGG16 Underperforms (The Spatial-Semantic Dilemma)
Standard transfer learning pipelines typically take an ImageNet-pretrained backbone (like VGG16) and attach a classifier directly to its final convolutional output (`block5_pool`, $4 \times 4$ spatial resolution).

While Block 5 encodes rich high-level semantic abstractions (such as overall tumor mass and necrosis), the cumulative effect of five max-pooling operations aggressively discards spatial resolution:
- **Fine spicular contours** (subtle needle-like projections at nodule boundaries) are erased.
- **Air bronchograms** and localized ground-glass density patterns are smoothed out.
- **Intermediate morphological transitions** between healthy parenchymal borders and cancerous margins are lost.

Consequently, standard VGG16 struggles to differentiate histologically similar carcinomas, frequently confusing adenocarcinoma with squamous cell carcinoma.

---

## 3. The Proposed MS-GF-VGG16 Architecture

To resolve the spatial-semantic dilemma, the **MS-GF-VGG16** architecture was designed and implemented using TensorFlow and the Keras 3 Functional API.

```
                        Input Axial CT Scan (128 × 128 × 3)
                                       │
                                       ▼
                  ImageNet-Pretrained VGG16 Backbone (Frozen Early Layers)
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
      Block 3 Output             Block 4 Output             Block 5 Output
    (16 × 16 × 256)             (8 × 8 × 512)              (4 × 4 × 512)
            │                          │                          │
            ▼                          ▼                          ▼
      1×1 Conv2D (128)           1×1 Conv2D (128)           1×1 Conv2D (128)
            │                          │                          │
            ▼                          ▼                          ▼
      Native 16 × 16             Bilinear Upsample ×2       Bilinear Upsample ×4
            │                          │                          │
            ▼                          ▼                          ▼
     F3: 16 × 16 × 128          F4: 16 × 16 × 128          F5: 16 × 16 × 128
            │                          │                          │
            └──────────────────────────┬──────────────────────────┘
                                       │
                                       ▼
                     Scale-Gated Feature Fusion Unit (SGFU)
                     ├── Dual Spatial Descriptors: [GAP(F_k) || GMP(F_k)] (256-d each)
                     ├── Concatenated Context Vector: D ∈ ℝ⁷⁶⁸
                     ├── Gating MLP: Dense(128, ReLU) ──► Dense(64, ReLU) ──► Logits [g3, g4, g5]
                     ├── Softmax Importance Weights: [α3, α4, α5] where α3 + α4 + α5 = 1
                     └── Weighted Fusion: F_fused = α3·F3 + α4·F4 + α5·F5 (16 × 16 × 128)
                                       │
                                       ▼
                     Classification Head
                     ├── Global Average Pooling (128-d)
                     ├── Dense(256, ReLU, L2 Regularization) + Dropout(0.5)
                     ├── Dense(128, ReLU, L2 Regularization) + Dropout(0.3)
                     └── Dense(4, Softmax) ──► [P(Adeno), P(LargeCell), P(Normal), P(Squamous)]
```

### 3.1 Multi-Scale Feature Extraction and Projection
Features are extracted at three distinct depths:
- **Block 3 Output**: $16 \times 16 \times 256$ (Local cellular textures and fine boundaries)
- **Block 4 Output**: $8 \times 8 \times 512$ (Intermediate structural morphology and lobular transitions)
- **Block 5 Output**: $4 \times 4 \times 512$ (High-level tumor semantics and gross pathology)

Individual $1 \times 1$ convolutional layers project all three feature sets into a uniform 128-channel depth, followed by bilinear spatial upsampling to align all feature maps to an identical dimension:
$$F_3 \in \mathbb{R}^{16 \times 16 \times 128}, \quad F_4 \in \mathbb{R}^{16 \times 16 \times 128}, \quad F_5 \in \mathbb{R}^{16 \times 16 \times 128}$$

### 3.2 The Scale-Gated Feature Fusion Unit (SGFU)
The SGFU is implemented as a custom, serializable Keras layer (`ScaleGatedFeatureFusionUnit`). Rather than relying on rigid addition or uniform averaging, the module dynamically learns to weigh the importance of each scale for each image:

1. **Dual Global Pooling**: Both Global Average Pooling (GAP) and Global Max Pooling (GMP) are extracted from each aligned feature map to capture both average context and salient peaks:
   $$\mathbf{desc}_k = [\text{GAP}(F_k) \parallel \text{GMP}(F_k)] \in \mathbb{R}^{256}, \quad k \in \{3, 4, 5\}$$
2. **Channel Concatenation**: The pooled descriptors are combined into a 768-dimensional context descriptor:
   $$\mathbf{D} = [\mathbf{desc}_3 \parallel \mathbf{desc}_4 \parallel \mathbf{desc}_5] \in \mathbb{R}^{768}$$
3. **Gating MLP**: The descriptor is processed through a compact multi-layer perceptron:
   $$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{D} + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{128 \times 768}$$
   $$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2), \quad \mathbf{W}_2 \in \mathbb{R}^{64 \times 128}$$
   $$\mathbf{g} = \mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3, \quad \mathbf{g} = [g_3, g_4, g_5] \in \mathbb{R}^3$$
4. **Softmax Normalization**: Normalized importance weights are obtained via softmax:
   $$\alpha_i = \frac{e^{g_i}}{\sum_{j=3}^{5} e^{g_j}}, \quad \text{guaranteeing that } \sum_{i=3}^5 \alpha_i = 1 \text{ and } \alpha_i > 0$$
5. **Weighted Feature Fusion**:
   $$F_{\text{fused}} = \alpha_3 F_3 + \alpha_4 F_4 + \alpha_5 F_5 \in \mathbb{R}^{16 \times 16 \times 128}$$

---

## 4. Dataset and Training Pipeline

### 4.1 Dataset Structure
The system is built upon the established **IQ-OTH/NCCD Chest CT-Scan Images Dataset**:
- **Total Valid Scans**: 1,000 images (all tested and verified for format integrity).
- **Distribution Protocol**: Strict partition ensuring zero data contamination:
  - **Training Set (70%)**: 613 scans (Adenocarcinoma: 195, Large Cell: 115, Normal: 148, Squamous Cell: 155)
  - **Validation Set (10%)**: 72 scans (Adenocarcinoma: 23, Large Cell: 21, Normal: 13, Squamous Cell: 15)
  - **Test Set (20%)**: 315 scans (Adenocarcinoma: 120, Large Cell: 51, Normal: 54, Squamous Cell: 90)

### 4.2 Data Preprocessing and Augmentation Rules
- **ImageNet Zero-Centering**: Raw $[0, 255]$ pixel values are converted from RGB to BGR, and ImageNet channel means ($[103.939, 116.779, 123.68]$) are subtracted. No duplicate scaling is applied.
- **Augmentation Boundary**: Data augmentation (random rotations up to $\pm 15^\circ$, small horizontal/vertical translations, zoom $\pm 8\%$, and horizontal flips) is applied **exclusively to the training set**. Validation and testing images undergo strictly deterministic preprocessing.
- **Class Balancing**: Balanced inverse-frequency class weights are dynamically computed to mitigate distribution imbalance during gradient updates.

### 4.3 Staged Optimization Strategy
- **Stage 1 (Feature Projection & SGFU Warmup)**: The first 10 layers of the VGG16 backbone are locked. The projection convolutions, SGFU gating MLP, and dense classification layers are trained using the Adam optimizer with an initial learning rate of $10^{-4}$.
- **Stage 2 (Deep Layer Fine-Tuning)**: Deeper convolutional layers (`block4_conv` and `block5_conv`) are unfrozen and fine-tuned with a lower learning rate ($10^{-5}$) alongside `ReduceLROnPlateau` and `EarlyStopping` callbacks.

---

## 5. Empirical Results & Ablation Study

All reported metrics originate directly from evaluating the trained model checkpoint on the untouched **315-scan test partition**.

### 5.1 Ablation Study Benchmark

To isolate and prove the contribution of the custom Scale-Gated Feature Fusion Unit, three architectures were trained and evaluated under identical conditions:

| Experiment | Architecture Variant | Test Accuracy | Macro F1-Score | Primary Finding |
| :--- | :--- | :---: | :---: | :--- |
| **Exp 1** | **Standard Pretrained VGG16** | **54.92%** | **0.5541** | Relying solely on Block 5 loses spatial textures, resulting in poor subtype differentiation. |
| **Exp 2** | **MS-VGG16 (Static Average)** | **42.22%** | **0.4048** | Static unweighted averaging treats noisy, uninformative scales equally, degrading accuracy. |
| **Exp 3** | **MS-GF-VGG16 (Proposed SGFU)** | **73.97%** | **0.7713** | **+19.05% gain over standard VGG16; +31.75% gain over static fusion.** |

### 5.2 Clinical Test Metrics Breakdown (MS-GF-VGG16)

- **Overall Test Accuracy**: **73.97%**
- **Macro F1-Score**: **0.7713 (77.13%)**
- **Weighted F1-Score**: **0.7425 (74.25%)**
- **Macro Specificity**: **0.9022 (90.22%)**
- **Macro ROC-AUC (One-vs-Rest)**: **0.9157 (91.57%)**

#### Per-Class Performance:
- **Normal (Healthy)**: Precision: **100.0%** | Recall: **94.4%** | F1-Score: **97.1%** | Specificity: **100.0%**
- **Large Cell Carcinoma**: Precision: **87.2%** | Recall: **66.7%** | F1-Score: **75.6%** | Specificity: **98.1%**
- **Adenocarcinoma**: Precision: **66.1%** | Recall: **71.7%** | F1-Score: **68.8%** | Specificity: **77.4%**
- **Squamous Cell Carcinoma**: Precision: **65.3%** | Recall: **68.9%** | F1-Score: **67.0%** | Specificity: **85.3%**

#### Test Confusion Matrix ($N = 315$):
```
                       PREDICTED CLASS
              Adeno    LargeCell    Normal    Squamous   |  Total
True Adeno      86         5           0         29      |   120
True LargeCell  13        34           0          4      |    51
True Normal      3         0          51          0      |    54
True Squamous   28         0           0         62      |    90
```

### 5.3 Scientific Discussion & Analysis of Accuracy
1. **Near-Perfect Specificity for Non-Malignancy**: Healthy normal lung parenchyma achieves **100% precision and 100% specificity**, generating zero false positive alarms for healthy patients.
2. **Adenocarcinoma vs. Squamous Cell Distinction**: In 2D axial CT slices, adenocarcinoma and squamous cell carcinoma present with nearly identical attenuation characteristics. Remaining misclassifications occur almost entirely along this subtype boundary.
3. **Roadmap to 95%+ Accuracy**:
   - **3D Volumetric Aggregation**: Transitioning from independent 2D cross-sections to 3D volumetric ConvNets or Swin Transformers to leverage inter-slice spatial continuity.
   - **Hounsfield Unit (HU) Radiodensity Windowing**: Preprocessing raw DICOM slices with tailored lung windowing (window width ~1500, window level -600) rather than 8-bit RGB approximations.
   - **Multi-Modal Radiomic Feature Fusion**: Extracting radiomic texture statistics (e.g., GLCM, Haralick textures) and fusing them into the classification head alongside deep SGFU features.

---

## 6. Explainable AI: Grad-CAM Implementation

To ensure clinical transparency, the platform integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**:
- After inference, the system computes the gradients of the top predicted class score with respect to the fused $16 \times 16 \times 128$ SGFU feature map.
- The resulting heatmap highlights the spatial regions of the CT scan that had the greatest positive influence on the network's classification.
- The web interface allows clinicians to interactively toggle between:
  1. The original CT scan.
  2. The raw Grad-CAM activation heatmap.
  3. An alpha-blended overlay directly superimposed over anatomical structures.
  4. A side-by-side diagnostic comparative view.

---

## 7. Full-Stack Software Engineering Architecture

The platform was built with a clean separation of concerns across backend, frontend, training, and evaluation modules:

```
lung-cancer-ms-gf-vgg16/
│
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI server configuration & CORS setup
│   │   ├── routes/
│   │   │   ├── predict.py           # POST /predict with image validation & Grad-CAM
│   │   │   ├── info.py              # GET /health, /model-info, /evaluation, /ablation
│   │   │   └── stats.py             # GET /dataset-stats, GET /samples
│   │   └── services/
│   │       └── model_service.py     # Singleton model inference & Grad-CAM service
│   ├── training/
│   │   ├── model.py                 # Custom SGFU Layer & MS-GF-VGG16 architecture
│   │   ├── dataset.py               # Robust dataset loader & tf.data pipeline
│   │   ├── train.py                 # Two-stage training and ablation script
│   │   ├── evaluate.py              # Pure NumPy clinical metric calculation & plots
│   │   └── gradcam.py               # Grad-CAM heatmap generation module
│   ├── models/
│   │   └── best_ms_gf_vgg16.keras   # Saved model weights checkpoint (159 MB)
│   ├── results/
│   │   ├── metrics.json             # Genuine test evaluation results
│   │   ├── ablation.json            # Ablation experiment data
│   │   └── plots/                   # Generated confusion matrix and ROC figures
│   ├── config.py                    # Centralized hyperparameters and directory paths
│   └── requirements.txt             # Python backend dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx           # Modern navigation bar with live API status
│   │   │   └── Footer.jsx           # Technical footer with medical disclaimer
│   │   ├── pages/
│   │   │   ├── LandingPage.jsx      # Overview, categories, and workflow
│   │   │   ├── PredictionPage.jsx   # Drag-and-drop analysis, samples, Grad-CAM
│   │   │   ├── PerformancePage.jsx  # Confusion matrix, KPIs, ROC, ablation
│   │   │   ├── ArchitecturePage.jsx # SGFU equations, code snippet, tensor table
│   │   │   └── AboutPage.jsx        # Clinical background, tech stack, documentation
│   │   ├── services/api.js          # Axios API communication service
│   │   ├── App.jsx                  # Main router and status poller
│   │   └── index.css                # Tailwind CSS styling
│   ├── package.json
│   └── vite.config.js
│
├── Data/                            # IQ-OTH/NCCD CT Dataset (train, valid, test)
├── README.md                        # Quickstart and execution manual
└── PROJECT_EXPLANATION.md           # This comprehensive documentation file
```

---

## 8. Frontend Interface Overview

The frontend is styled as an AI-powered medical decision support platform:
- **Landing Page**: Project overview, diagnostic class cards, end-to-end workflow explanation, and direct CTA to analyze scans.
- **AI Prediction Studio**: Drag-and-drop CT scan upload, four one-click sample scans (one per class), four-class probability distribution bar chart (Recharts), and multi-mode Grad-CAM viewer.
- **Model Performance Dashboard**: Live KPI tiles, interactive $4 \times 4$ confusion matrix, per-class performance tables, ablation study comparison cards, and two-stage training dynamic curves.
- **Architecture Deep-Dive**: Mathematical breakdown of the SGFU equation, forward pass code, and complete tensor transformation tracking.
- **Research & Methodology Page**: Formal project report covering clinical motivation, technology stack, and future directions.

---

## 9. Setup, Execution, and Verification Guide

### 9.1 Running the Backend API
```powershell
# In the project root directory
python -m uvicorn backend.app.main:app --port 8000 --host 127.0.0.1
```
- API Base: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`

### 9.2 Running the React Frontend
```powershell
# In the frontend directory
cd frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```
- Web Application: `http://127.0.0.1:5173`

---

## 10. Academic Major Project Context

This system represents a comprehensive final-year B.Tech / Major Capstone demonstration in Artificial Intelligence and Medical Image Analysis:
1. **Mathematical Rigor**: Formulates and implements a trainable multi-scale feature fusion unit (SGFU).
2. **Empirical Verification**: Conducts an ablation study proving a **+19.05%** improvement over standard VGG16.
3. **Research Integrity**: Reports honest metrics computed on an untouched test set without data leakage.
4. **Clinical Transparency**: Implements explainable AI (Grad-CAM) to visualize decision-making regions.
5. **Full-Stack Deployment**: Bridges deep learning research with a production-grade FastAPI backend and responsive React frontend.

