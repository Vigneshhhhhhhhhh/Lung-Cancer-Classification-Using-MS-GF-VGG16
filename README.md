# Lung Cancer Classification Using MS-GF-VGG16
### Multi-Scale Gated Feature Fusion VGG16 for Pulmonary CT Scan Histopathology Classification

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

---

## 1. Project Overview

This repository houses the complete, end-to-end implementation of an academic major engineering capstone project: **Lung Cancer Classification Using MS-GF-VGG16**.

The system automates the multi-class categorization of chest computed tomography (CT) scans into four distinct diagnostic classes:
1. **Adenocarcinoma** (Most prevalent Non-Small Cell Lung Carcinoma)
2. **Large Cell Carcinoma** (Aggressive undifferentiated epithelial neoplasm)
3. **Squamous Cell Carcinoma** (Central bronchial airway neoplasia)
4. **Normal** (Non-malignant healthy pulmonary parenchyma)

Unlike standard transfer learning approaches that discard intermediate spatial representations, this project introduces the **Scale-Gated Feature Fusion Unit (SGFU)**, a custom trainable neural network layer that adaptively weights feature maps across three hierarchical stages of an ImageNet-pretrained VGG16 backbone.

---

## 2. Architectural Specification: MS-GF-VGG16

### 2.1 The Spatial-Semantic Dilemma in CT Imaging
Standard CNN backbones (e.g., vanilla VGG16) extract features solely from their terminal pooling layer (`block5_pool`, 4×4 spatial resolution). While Block 5 captures high-level lesion semantics, excessive spatial downsampling obliterates localized micro-textures—such as fine spicular margins, ground-glass opacity patterns, and subtle bronchial infiltrations—causing severe misclassification between adenocarcinoma and squamous cell carcinoma.

### 2.2 Multi-Scale Feature Extraction & Alignment
MS-GF-VGG16 resolves this limitation by extracting multi-scale feature hierarchies:
- **Block 3 Output**: $16 \times 16 \times 256$ (Fine textures, localized contours)
- **Block 4 Output**: $8 \times 8 \times 512$ (Intermediate tissue morphology)
- **Block 5 Output**: $4 \times 4 \times 512$ (High-level tumor semantics)

Individual $1 \times 1$ 2D convolutional projections compress each map to 128 channels:
- $F_3$: $16 \times 16 \times 128$ (Retained at native $16 \times 16$)
- $F_4$: $8 \times 8 \times 128 \xrightarrow{\text{Upsample } \times 2} 16 \times 16 \times 128$
- $F_5$: $4 \times 4 \times 128 \xrightarrow{\text{Upsample } \times 4} 16 \times 16 \times 128$

### 2.3 Scale-Gated Feature Fusion Unit (SGFU)
The SGFU dynamically computes importance weights for each scale conditioned on global spatial descriptors:

1. **Dual Global Pooling**: Computes both Global Average Pooling (GAP) and Global Max Pooling (GMP) for each aligned scale ($F_3, F_4, F_5$):
   $$\mathbf{desc}_k = [\text{GAP}(F_k) \parallel \text{GMP}(F_k)] \in \mathbb{R}^{256}$$
2. **Channel Concatenation**: Concatenates pooled descriptors into a 768-dimensional context vector:
   $$\mathbf{D} = [\mathbf{desc}_3 \parallel \mathbf{desc}_4 \parallel \mathbf{desc}_5] \in \mathbb{R}^{768}$$
3. **Gating MLP**: Passes $\mathbf{D}$ through a compact gating network:
   $$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{D} + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{128 \times 768}$$
   $$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2), \quad \mathbf{W}_2 \in \mathbb{R}^{64 \times 128}$$
   $$\mathbf{g} = \mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3, \quad \mathbf{g} = [g_3, g_4, g_5] \in \mathbb{R}^3$$
4. **Softmax Scale Normalization**:
   $$\alpha_i = \frac{e^{g_i}}{\sum_{j=3}^{5} e^{g_j}}, \quad \text{such that } \sum_{i=3}^5 \alpha_i = 1$$
5. **Weighted Feature Fusion**:
   $$F_{\text{fused}} = \alpha_3 F_3 + \alpha_4 F_4 + \alpha_5 F_5 \in \mathbb{R}^{16 \times 16 \times 128}$$

### 2.4 Classification Head
- Global Average Pooling $\rightarrow 128$-d
- Dense(256, ReLU, L2 Regularization) + Dropout(0.5)
- Dense(128, ReLU, L2 Regularization) + Dropout(0.3)
- Dense(4, Softmax) $\rightarrow$ 4-class discrete probability distribution.

---

## 3. Dataset & Preprocessing

The system utilizes the established **IQ-OTH/NCCD Chest CT-Scan Images Dataset**:
- **Total Valid Scans**: 1,000 images (0 corrupted images detected).
- **Partition Ratio**: 70% Training / 10% Validation / 20% Testing (strict split).
  - `Data/train/`: 613 scans
  - `Data/valid/`: 72 scans
  - `Data/test/`: 315 scans (completely untouched during training)
- **Canonical Class Resolution**: Subdirectories with arbitrary naming conventions (e.g., `adenocarcinoma_left.lower.lobe_...`) are automatically canonicalized.
- **Data Augmentation**: Small rotations ($\pm 15^\circ$), height/width shifts ($\pm 8\%$), zoom ($\pm 8\%$), and horizontal flips are applied **exclusively to the training set**.
- **Normalization**: VGG16 ImageNet channel zero-centering ($[-103.939, -116.779, -123.68]$) on raw $[0, 255]$ inputs without duplicate scaling.

---

## 4. Empirical Evaluation & Ablation Study

Evaluated on the independent **315-image test partition**:

| Experiment Architecture | Test Accuracy | Macro F1-Score | Macro Specificity | Macro ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Exp 1: Standard VGG16** | 54.92% | 0.5541 | 82.10% | 0.7812 |
| **Exp 2: MS-VGG16 (Static Average)** | 42.22% | 0.4048 | 79.40% | 0.7105 |
| **Exp 3: MS-GF-VGG16 (Proposed)** | **73.97%** | **0.7713** | **90.22%** | **0.9157** |

### Per-Class Test Performance (MS-GF-VGG16):
- **Normal (Healthy)**: Precision: **100.0%** | Recall: **94.4%** | F1: **97.1%** | Specificity: **100.0%**
- **Large Cell Carcinoma**: Precision: **87.2%** | Recall: **66.7%** | F1: **75.6%** | Specificity: **98.1%**
- **Adenocarcinoma**: Precision: **66.1%** | Recall: **71.7%** | F1: **68.8%** | Specificity: **77.4%**
- **Squamous Cell Carcinoma**: Precision: **65.3%** | Recall: **68.9%** | F1: **67.0%** | Specificity: **85.3%**

> **Clinical Finding**: The Normal class is identified with near-perfect clinical specificity (100%) and 0 false alarms. The primary remaining misclassifications occur between Adenocarcinoma and Squamous Cell Carcinoma, which are known non-small-cell subtypes sharing identical attenuation densities.

---

## 5. System Architecture & Tech Stack

```
lung-cancer-ms-gf-vgg16/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI Application & CORS configuration
│   │   ├── routes/
│   │   │   ├── predict.py           # POST /predict with image validation & Grad-CAM
│   │   │   ├── info.py              # GET /health, /model-info, /evaluation, /ablation
│   │   │   └── stats.py             # GET /dataset-stats, /samples
│   │   └── services/
│   │       └── model_service.py     # Singleton model inference & Grad-CAM pipeline
│   ├── training/
│   │   ├── model.py                 # SGFU custom Layer & MS-GF-VGG16 architecture
│   │   ├── dataset.py               # Dataset scanning, canonical mapping, tf.data
│   │   ├── train.py                 # Two-stage training and ablation study script
│   │   ├── evaluate.py              # Pure NumPy metrics, confusion matrix, ROC plots
│   │   └── gradcam.py               # SGFU activation map computation
│   ├── models/
│   │   └── best_ms_gf_vgg16.keras   # Saved model weights checkpoint
│   ├── results/
│   │   ├── metrics.json             # Genuine test evaluation metrics
│   │   ├── ablation.json            # Ablation study comparison
│   │   └── plots/                   # Saved confusion matrix & ROC figures
│   ├── config.py                    # Centralized parameters & hyperparameters
│   └── requirements.txt             # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/              # Navbar, Footer
│   │   ├── pages/                   # LandingPage, PredictionPage, PerformancePage,
│   │   │                            # ArchitecturePage, AboutPage
│   │   ├── services/api.js          # Axios API client
│   │   ├── App.jsx                  # Main application router
│   │   └── index.css                # Tailwind CSS styling
│   ├── package.json
│   └── vite.config.js
│
├── Data/                            # IQ-OTH/NCCD CT Dataset (train, valid, test)
└── README.md
```

---

## 6. Installation & Execution Guide

### 6.1 Prerequisites
- Python 3.10 or 3.11
- Node.js v18+ and npm
- Windows 10/11 or Linux

### 6.2 Running the Backend API
```bash
# 1. Install backend dependencies
pip install -r backend/requirements.txt

# 2. Start the FastAPI ASGI server
python -m uvicorn backend.app.main:app --port 8000 --reload
```
The backend API documentation will be accessible at `http://127.0.0.1:8000/docs`.

### 6.3 Running the React Frontend
```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm.cmd install

# 3. Start development server
npm.cmd run dev
```
Open your browser at `http://localhost:5173`.

### 6.4 Retraining or Running Ablation Study
```bash
python -m backend.training.train
```

---

## 7. Explainable AI (Grad-CAM)
The platform includes an integrated **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualization module. Upon image submission, the system computes the gradients of the top predicted class score with respect to the fused $16 \times 16 \times 128$ SGFU tensor. 

Users can toggle between:
- **Original CT Scan**
- **SGFU Activation Heatmap** (Jet colormap)
- **Alpha-Blended Saliency Overlay**
- **Side-by-Side Diagnostic Comparison**

---

## 8. Academic Project Disclaimer
This software is an **academic AI research prototype** developed for educational and experimental demonstration. It is **not** certified as a clinical medical device. Predictions generated by this system must never replace formal histopathological examination or consultation with licensed medical practitioners.

