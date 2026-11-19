import React from 'react';
import { 
  Info, 
  Target, 
  Layers, 
  Cpu, 
  Code2, 
  Compass, 
  Sparkles, 
  CheckCircle2, 
  FileText,
  Workflow
} from 'lucide-react';

export default function AboutPage() {
  const techStack = [
    { category: 'Deep Learning & Modeling', items: ['Python 3.11', 'TensorFlow 2.21', 'Keras 3 Functional API', 'ImageNet VGG16 Backbone', 'Custom SGFU Layer', 'Grad-CAM XAI'] },
    { category: 'Backend & Inference API', items: ['FastAPI (Asynchronous)', 'Uvicorn ASGI Server', 'NumPy & SciPy (Pure Matrix Operations)', 'Pillow (PIL Imaging Engine)', 'Pydantic v2 Type Validation'] },
    { category: 'Frontend Interface', items: ['React 18', 'Vite 6 Fast Bundler', 'Tailwind CSS Modern Styling', 'Recharts Responsive Data Viz', 'Lucide React Clean Icons', 'Axios HTTP Client'] },
    { category: 'Benchmark Dataset', items: ['IQ-OTH/NCCD Chest CT Dataset', '1,000 Verified Scans', '70/10/20 Train/Val/Test Split', 'Zero Data Leakage Protocol'] },
  ];

  return (
    <div className="space-y-12 py-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="space-y-2">
        <div className="flex items-center space-x-2 text-medical-600 text-xs font-bold uppercase tracking-wider">
          <Info className="w-4 h-4" />
          <span>Academic Project Documentation</span>
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          Research Context, Methodology & Technology Stack
        </h1>
        <p className="text-sm text-slate-600 max-w-3xl">
          Comprehensive project report outlining clinical motivation, comparative architectural advantages, 
          engineering pipeline implementation, and future directions for Computer-Aided Diagnosis (CAD).
        </p>
      </div>

      {/* Problem Statement & Background */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
        <div className="bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm space-y-4">
          <div className="w-10 h-10 rounded-xl bg-rose-50 text-rose-600 flex items-center justify-center font-bold">
            <Target className="w-5 h-5" />
          </div>
          <h2 className="text-xl font-bold text-slate-900">
            Clinical Problem Statement
          </h2>
          <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
            Pulmonary neoplasia remains the foremost cause of cancer mortality worldwide. Computed Tomography (CT) 
            imaging provides non-invasive visualization of pulmonary parenchyma; however, manual radiographic differentiation 
            between non-small cell lung carcinoma (NSCLC) subtypes—specifically <strong className="text-slate-800">Adenocarcinoma</strong> versus <strong className="text-slate-800">Squamous Cell Carcinoma</strong>—remains 
            a daunting diagnostic challenge due to pronounced visual homogeneity, subtle lobular margins, and variable ground-glass opacities.
          </p>
          <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
            Biopsy procedures are invasive and carry substantial morbidity risks. An automated, explainable deep learning 
            system that accurately triages CT scans can provide crucial early-stage decision support for interventional pulmonologists and thoracic oncologists.
          </p>
        </div>

        <div className="bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm space-y-4">
          <div className="w-10 h-10 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center font-bold">
            <Workflow className="w-5 h-5" />
          </div>
          <h2 className="text-xl font-bold text-slate-900">
            Limitations of Conventional VGG16
          </h2>
          <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
            Standard VGG16 transfer learning pipelines exclusively extract features from the terminal layer (<span className="font-mono text-slate-800">block5_pool</span>, 4×4 spatial resolution). 
            While these representations capture rich high-level semantic abstractions, the aggressive spatial downsampling through five pooling operations completely erases:
          </p>
          <ul className="list-disc list-inside space-y-1.5 text-xs text-slate-600 pl-1">
            <li>Fine-grained spicular margins and microscopic peripheral nodular contours.</li>
            <li>Localized tissue density variations and subtle ground-glass attenuation patterns.</li>
            <li>Intermediate morphological transitions between healthy parenchyma and neoplastic borders.</li>
          </ul>
          <p className="text-xs text-slate-500 italic pt-1">
            Result: Standard VGG16 exhibits substantial misclassification rates between histologically similar cancer subtypes.
          </p>
        </div>
      </div>

      {/* Proposed Architectural Contribution */}
      <div className="bg-gradient-to-r from-medical-900 via-slate-900 to-medical-950 rounded-3xl p-8 sm:p-12 text-white shadow-xl space-y-6">
        <div className="max-w-3xl space-y-4">
          <span className="text-xs font-bold uppercase tracking-wider text-teal-400 flex items-center space-x-1.5">
            <Sparkles className="w-4 h-4" />
            <span>Core Scientific Contribution</span>
          </span>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white">
            The MS-GF-VGG16 Paradigm
          </h2>
          <p className="text-sm text-slate-300 leading-relaxed">
            MS-GF-VGG16 introduces an innovative <strong className="text-white">Scale-Gated Feature Fusion Unit (SGFU)</strong>. 
            Rather than relying solely on the final layer, our architecture taps into three distinct stages of the VGG16 representation hierarchy:
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 pt-2">
          <div className="p-5 rounded-2xl bg-white/5 border border-white/10 space-y-2">
            <span className="text-xs font-mono font-bold text-medical-300">Scale F3 (16 × 16 × 128)</span>
            <h3 className="text-base font-bold text-white">Shallow Micro-Texture</h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Extracted from Block 3. Captures localized spicular edges, internal air bronchograms, and cell texture variations.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-white/5 border border-white/10 space-y-2">
            <span className="text-xs font-mono font-bold text-teal-300">Scale F4 (16 × 16 × 128)</span>
            <h3 className="text-base font-bold text-white">Intermediate Morphology</h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Extracted from Block 4 and upsampled ×2. Represents vascular infiltration and nodule boundary architecture.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-white/5 border border-white/10 space-y-2">
            <span className="text-xs font-mono font-bold text-amber-300">Scale F5 (16 × 16 × 128)</span>
            <h3 className="text-base font-bold text-white">High-Level Semantics</h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Extracted from Block 5 and upsampled ×4. Captures macroscopic neoplastic volume, pleural retraction, and core necrosis.
            </p>
          </div>
        </div>

        <div className="p-5 rounded-xl bg-white/5 border border-white/10 text-xs text-slate-300 space-y-1">
          <span className="font-bold text-teal-300">Adaptive Gating Mechanism: </span>
          <span>
            Unlike static concatenation or unweighted averaging, the SGFU dynamically computes scale weights [α3, α4, α5] via a dual-pooling descriptor (GAP + GMP) and gating MLP, allowing the network to adaptively focus on the most discriminative feature scale for each patient scan.
          </span>
        </div>
      </div>

      {/* Technology Stack Grid */}
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
          Technology Stack & Implementation Stack
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {techStack.map((tech, idx) => (
            <div key={idx} className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
              <h3 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-2 flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-medical-600" />
                <span>{tech.category}</span>
              </h3>
              <ul className="space-y-2">
                {tech.items.map((item, i) => (
                  <li key={i} className="text-xs text-slate-600 flex items-center space-x-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-teal-600 shrink-0" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* Project Objectives & Future Work */}
      <div className="bg-white p-8 sm:p-10 rounded-2xl border border-slate-200 shadow-sm space-y-6">
        <h2 className="text-xl font-bold text-slate-900">
          Academic Project Objectives & Future Enhancements
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-xs text-slate-600 leading-relaxed">
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-medical-800 flex items-center space-x-2">
              <Compass className="w-4 h-4 text-medical-600" />
              <span>Fulfilled Major Project Milestones</span>
            </h3>
            <ul className="space-y-2 list-disc list-inside text-slate-600">
              <li>Full implementation of MS-GF-VGG16 architecture with custom trainable SGFU layer.</li>
              <li>Rigorous two-stage transfer learning with early stopping and learning rate scheduling.</li>
              <li>Ablation study demonstrating empirical superiority over Standard VGG16.</li>
              <li>Grad-CAM explainable AI module for clinical visual transparency.</li>
              <li>Production-grade asynchronous FastAPI backend and responsive React frontend.</li>
            </ul>
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-bold text-teal-800 flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-teal-600" />
              <span>Future Translational Extensions</span>
            </h3>
            <ul className="space-y-2 list-disc list-inside text-slate-600">
              <li>Integration of 3D Volumetric CT slice aggregators (3D ConvNet / Swin Transformer).</li>
              <li>Multi-institutional dataset validation across diverse scanner manufacturers (GE, Siemens, Philips).</li>
              <li>Native DICOM format parsing and hospital PACS/HL7 FHIR clinical interoperability.</li>
              <li>Semi-supervised active learning to incorporate radiologist-in-the-loop annotations.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

