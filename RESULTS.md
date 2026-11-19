# Model Performance & Empirical Evaluation Results

## Multi-Scale Gated Feature Fusion VGG16 (MS-GF-VGG16)

This document presents the complete, verified empirical evaluation results of the **MS-GF-VGG16** deep learning model.

All metrics, confusion matrix values, ROC-AUC scores, and visual graphs reported below are derived strictly from evaluating the trained model checkpoint (`backend/models/best_ms_gf_vgg16.keras`) on the independent, untouched test partition ($N = 315$ pulmonary CT scans) without any data leakage or simulated values.

---

## 1. Summary of Overall Test Metrics

| Evaluation Metric | Score (Decimal) | Score (Percentage) | Mathematical Formula / Definition |
| :--- | :---: | :---: | :--- |
| **Overall Test Accuracy** | **0.7397** | **73.97%** | $\frac{\text{Correct Predictions}}{\text{Total Test Samples}} = \frac{233}{315}$ |
| **Macro Precision** | **0.7965** | **79.65%** | $\frac{1}{K}\sum_{k=1}^K \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k}$ (Unweighted mean) |
| **Weighted Precision** | **0.7681** | **76.81%** | $\sum_{k=1}^K w_k \cdot \text{Precision}_k$ (Support-weighted) |
| **Macro Recall (Sensitivity)**| **0.7542** | **75.42%** | $\frac{1}{K}\sum_{k=1}^K \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}$ (True Positive Rate) |
| **Weighted Recall** | **0.7397** | **73.97%** | $\sum_{k=1}^K w_k \cdot \text{Recall}_k$ |
| **Macro F1-Score** | **0.7713** | **77.13%** | Harmonic mean of Macro Precision & Macro Recall |
| **Weighted F1-Score** | **0.7425** | **74.25%** | $\sum_{k=1}^K w_k \cdot \text{F1}_k$ |
| **Macro Specificity** | **0.9022** | **90.22%** | $\frac{1}{K}\sum_{k=1}^K \frac{\text{TN}_k}{\text{TN}_k + \text{FP}_k}$ (True Negative Rate) |
| **Macro ROC-AUC (OvR)** | **0.9157** | **91.57%** | One-vs-Rest Area Under Receiver Operating Characteristic |
| **Weighted ROC-AUC (OvR)**| **0.9088** | **90.88%** | Support-weighted Area Under the ROC Curve |
| **Total Test Dataset Size** | **315** | — | Independent partition: 0 images used in training |

---

## 2. Per-Class Diagnostic Performance

The dataset comprises four pulmonary histological classifications:
1. **Adenocarcinoma** (Peripheral non-small cell lung cancer)
2. **Large Cell Carcinoma** (Undifferentiated, aggressive neoplasm)
3. **Normal** (Non-malignant, healthy lung parenchyma)
4. **Squamous Cell Carcinoma** (Airway-lining central bronchial neoplasm)

| Diagnostic Class | Precision | Recall (Sensitivity) | F1-Score | Specificity (TNR) | ROC-AUC | Test Scans (Support) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal (Healthy)** | **100.0%** (1.000) | **94.4%** (0.944) | **97.1%** (0.971) | **100.0%** (1.000) | **0.985** | 54 scans |
| **Large Cell Carcinoma** | **87.2%** (0.872) | **66.7%** (0.667) | **75.6%** (0.756) | **98.1%** (0.981) | **0.912** | 51 scans |
| **Adenocarcinoma** | **66.1%** (0.661) | **71.7%** (0.717) | **68.8%** (0.688) | **77.4%** (0.774) | **0.887** | 120 scans |
| **Squamous Cell Carcinoma**| **65.3%** (0.653) | **68.9%** (0.689) | **67.0%** (0.670) | **85.3%** (0.853) | **0.879** | 90 scans |

### Visual Graph: Per-Class Diagnostic Metrics
![Per-Class Clinical Diagnostic Performance](./backend/results/plots/per_class_metrics.png)
> 🔗 **Open Image File**: [per_class_metrics.png](file:///c:/Users/Admin/OneDrive/Documents/Projects/Actual%20Project%20Files/DAIP%20(Lung%20Cancer)/backend/results/plots/per_class_metrics.png)

---

## 3. Confusion Matrix Analysis

The $4 \times 4$ multi-class confusion matrix evaluates predicted class outputs versus confirmed ground-truth labels across all 315 test scans:

```
                                      PREDICTED CLASS
                    Adenocarcinoma   Large Cell Carcinoma   Normal (Healthy)   Squamous Cell Carcinoma   │  Total
True Adenocarcinoma        86                 5                     0                     29             │   120
True Large Cell            13                34                     0                      4             │    51
True Normal                 3                 0                    51                      0             │    54
True Squamous Cell         28                 0                     0                     62             │    90
─────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────
Predicted Total           130                39                    51                     95             │   315
```

### Visual Graph: Confusion Matrix Heatmap
![Confusion Matrix Heatmap](./backend/results/plots/confusion_matrix.png)
> 🔗 **Open Image File**: [confusion_matrix.png](file:///c:/Users/Admin/OneDrive/Documents/Projects/Actual%20Project%20Files/DAIP%20(Lung%20Cancer)/backend/results/plots/confusion_matrix.png)

### Key Clinical Insights from Confusion Matrix:
- **Zero False Alarms for Non-Cancer**: The model achieved **100% precision and 100% specificity on healthy tissue**. No cancerous scan was misidentified as normal, and no normal scan was misclassified as large cell or squamous cell carcinoma (only 3 border-case scans were categorized as adenocarcinoma).
- **Subtype Boundary Homogeneity**: The predominant source of error occurred mutually between **Adenocarcinoma** ($n=29$ misclassified as Squamous) and **Squamous Cell Carcinoma** ($n=28$ misclassified as Adenocarcinoma). In 2D axial CT slices, these two non-small cell subtypes share nearly identical Hounsfield Unit (HU) attenuation values.

---

## 4. Multi-Class ROC Curves (One-vs-Rest)

Receiver Operating Characteristic (ROC) curves illustrate the trade-off between True Positive Rate (Sensitivity) and False Positive Rate ($1 - \text{Specificity}$) across continuous classification thresholds.

- **Normal (Healthy)**: $\text{AUC} = \mathbf{0.985}$ (Near-perfect discrimination)
- **Large Cell Carcinoma**: $\text{AUC} = \mathbf{0.912}$
- **Adenocarcinoma**: $\text{AUC} = \mathbf{0.887}$
- **Squamous Cell Carcinoma**: $\text{AUC} = \mathbf{0.879}$
- **Macro-Averaged ROC-AUC**: $\mathbf{0.9157}$

### Visual Graph: ROC Curves (One-vs-Rest)
![Multi-Class ROC Curves](./backend/results/plots/roc_curves.png)
> 🔗 **Open Image File**: [roc_curves.png](file:///c:/Users/Admin/OneDrive/Documents/Projects/Actual%20Project%20Files/DAIP%20(Lung%20Cancer)/backend/results/plots/roc_curves.png)

---

## 5. Two-Stage Training & Convergence Dynamics

The model was optimized in two successive stages:
- **Stage 1 (Warmup)**: The first 10 layers of VGG16 were frozen; only the $1 \times 1$ projection layers, the SGFU gating MLP, and the dense classification head were trained ($\text{Adam LR} = 10^{-4}$).
- **Stage 2 (Fine-Tuning)**: Deeper convolutional blocks (`block4` and `block5`) were unfrozen and fine-tuned with a reduced learning rate ($\text{Adam LR} = 10^{-5}$) alongside `ReduceLROnPlateau` and `EarlyStopping`.

### Visual Graph: Training & Validation Convergence
![Training and Validation Curves](./backend/results/plots/training_curves.png)
> 🔗 **Open Image File**: [training_curves.png](file:///c:/Users/Admin/OneDrive/Documents/Projects/Actual%20Project%20Files/DAIP%20(Lung%20Cancer)/backend/results/plots/training_curves.png)

- **Accuracy Trajectory**: Training accuracy progressed smoothly to over 86%, while validation accuracy stabilized at ~70-74%, bounded by the small validation sample size (72 scans).
- **Loss Trajectory**: Categorical crossentropy loss decreased from $>100$ down to $0.399$, with early stopping triggering at epoch 19 of Stage 2 to prevent catastrophic overfitting.

---

## 6. Ablation Study: Proving the SGFU Architectural Contribution

To verify whether the custom **Scale-Gated Feature Fusion Unit (SGFU)** genuinely enhances classification performance, three experimental variants were trained on the exact same dataset under identical preprocessing protocols:

| Experiment | Architecture Description | Test Accuracy | Macro F1-Score | Absolute Performance Difference |
| :--- | :--- | :---: | :---: | :---: |
| **Exp 1: Standard VGG16** | Baseline ImageNet VGG16 using only terminal Block 5 features ($4 \times 4 \times 512$). | **54.92%** | **0.5541** | Baseline |
| **Exp 2: MS-VGG16 (Static Average)** | Multi-scale fusion using static unweighted averaging ($F = \frac{F_3 + F_4 + F_5}{3}$). | **42.22%** | **0.4048** | $-12.70\%$ vs Exp 1 |
| **Exp 3: MS-GF-VGG16 (Proposed SGFU)** | Proposed multi-scale fusion with custom trainable **Scale-Gated Feature Fusion Unit**. | **73.97%** | **0.7713** | **+19.05% vs Exp 1<br>+31.75% vs Exp 2** |

### Visual Graph: Ablation Study Comparison
![Ablation Study Comparison](./backend/results/plots/ablation_comparison.png)
> 🔗 **Open Image File**: [ablation_comparison.png](file:///c:/Users/Admin/OneDrive/Documents/Projects/Actual%20Project%20Files/DAIP%20(Lung%20Cancer)/backend/results/plots/ablation_comparison.png)

### Architectural Interpretation:
1. **Why Standard VGG16 (Exp 1) Achieved Only 54.92%**: Standard VGG16 discards all intermediate representations. By the time features reach Block 5, five pooling layers have reduced spatial dimensions from $128 \times 128$ to $4 \times 4$, completely obliterating localized spicular contours and ground-glass nodule textures.
2. **Why Static Averaging (Exp 2) Failed (42.22%)**: Simply averaging the three scales treats noisy, uninformative spatial channels with the exact same weight as highly discriminative ones, introducing feature dilution.
3. **Why MS-GF-VGG16 (Exp 3) Succeeded (73.97%)**: The SGFU gating MLP learns continuous importance weights ($\alpha_3, \alpha_4, \alpha_5$) conditioned on both average and peak spatial activations, dynamically emphasizing shallow textures ($F_3$) for subtle margins and deep semantics ($F_5$) for gross tumor volume.

---

## 7. Honest Reporting on the 95% Accuracy Target & Engineering Roadmap

The initial project proposal targeted 95% test accuracy. In accordance with strict academic guidelines (*"Never artificially increase accuracy, manipulate evaluation results, use test images during training, or display fabricated metrics"*):

- **Genuine Achieved Performance**: **73.97% test accuracy** with **91.57% ROC-AUC** and **90.22% specificity**.
- **Benchmark Context**: In the published medical imaging literature on the IQ-OTH/NCCD benchmark, genuine non-leaked 2D slice classification typically ranges between 72% and 82%. Studies reporting 98-99% on this dataset frequently suffer from slice-level data leakage (where adjacent CT slices from the same patient appear in both the training and test splits).

### Concrete Recommendations for Reaching 95%+ in Future Work:
1. **3D Volumetric Context**: Replace 2D slice analysis with a 3D ConvNet or 3D Vision Transformer (e.g., Swin UNETR) to incorporate inter-slice volumetric continuity across consecutive axial slices.
2. **Hounsfield Unit (HU) Windowing**: Preprocess raw medical DICOM data with dedicated lung radiodensity windowing (window width $\approx 1500$, window level $\approx -600$) to accentuate parenchymal attenuation differences before CNN feature extraction.
3. **Radiomic Feature Integration**: Extract classical hand-crafted radiomic features (Gray-Level Co-occurrence Matrix [GLCM] texture, shape compactness, and sphericity) and concatenate them into the classification head alongside deep SGFU features.
4. **Targeted Subtype Contrastive Learning**: Utilize Supervised Contrastive Loss (SupCon) specifically penalizing embeddings that fail to separate Adenocarcinoma from Squamous Cell Carcinoma.

---

## 8. Summary of Saved Output Artifacts

All evaluation artifacts are preserved and accessible in the repository:
- **Trained Model Checkpoint**: [`backend/models/best_ms_gf_vgg16.keras`](file:///c:/Users/Admin/OneDrive/Documents/Projects/Actual%20Project%20Files/DAIP%20(Lung%20Cancer)/backend/models/best_ms_gf_vgg16.keras) (159 MB)
- **Detailed Metrics JSON**: [`backend/results/metrics.json`](file:///c:/Users/Admin/OneDrive/Documents/Projects/Actual%20Project%20Files/DAIP%20(Lung%20Cancer)/backend/results/metrics.json)
- **Ablation Study JSON**: [`backend/results/ablation.json`](file:///c:/Users/Admin/OneDrive/Documents/Projects/Actual%20Project%20Files/DAIP%20(Lung%20Cancer)/backend/results/ablation.json)
- **Confusion Matrix Figure**: [`backend/results/plots/confusion_matrix.png`](file:///c:/Users/Admin/OneDrive/Documents/Projects/Actual%20Project%20Files/DAIP%20(Lung%20Cancer)/backend/results/plots/confusion_matrix.png)
- **ROC Curves Figure**: [`backend/results/plots/roc_curves.png`](file:///c:/Users/Admin/OneDrive/Documents/Projects/Actual%20Project%20Files/DAIP%20(Lung%20Cancer)/backend/results/plots/roc_curves.png)
- **Training Trajectory Figure**: [`backend/results/plots/training_curves.png`](file:///c:/Users/Admin/OneDrive/Documents/Projects/Actual%20Project%20Files/DAIP%20(Lung%20Cancer)/backend/results/plots/training_curves.png)
- **Per-Class Metrics Bar Figure**: [`backend/results/plots/per_class_metrics.png`](file:///c:/Users/Admin/OneDrive/Documents/Projects/Actual%20Project%20Files/DAIP%20(Lung%20Cancer)/backend/results/plots/per_class_metrics.png)
- **Ablation Comparison Bar Figure**: [`backend/results/plots/ablation_comparison.png`](file:///c:/Users/Admin/OneDrive/Documents/Projects/Actual%20Project%20Files/DAIP%20(Lung%20Cancer)/backend/results/plots/ablation_comparison.png)

