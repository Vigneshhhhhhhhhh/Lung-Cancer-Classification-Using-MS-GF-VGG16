import React from 'react';
import { 
  Activity, 
  ArrowRight, 
  Layers, 
  Sparkles, 
  Cpu, 
  CheckCircle2, 
  Eye, 
  ShieldCheck, 
  Database,
  BarChart2
} from 'lucide-react';

export default function LandingPage({ setActiveTab, datasetStats, modelInfo }) {
  const categories = [
    {
      name: 'Adenocarcinoma',
      color: 'blue',
      badge: 'Most Common NSCLC',
      description: 'Often originates in outer parts of lungs; glandular tissue differentiation with irregular glandular lumens.',
      bg: 'bg-blue-50 border-blue-200 text-blue-900',
      tagBg: 'bg-blue-100 text-blue-700',
    },
    {
      name: 'Large Cell Carcinoma',
      color: 'teal',
      badge: 'Aggressive NSCLC',
      description: 'Fast-growing undifferentiated malignant epithelial neoplasm lacking glandular or squamous architecture.',
      bg: 'bg-teal-50 border-teal-200 text-teal-900',
      tagBg: 'bg-teal-100 text-teal-700',
    },
    {
      name: 'Squamous Cell Carcinoma',
      color: 'amber',
      badge: 'Centrally Located',
      description: 'Typically arises near central bronchi; exhibits keratinization, intercellular bridges, or squamous pearls.',
      bg: 'bg-amber-50 border-amber-200 text-amber-900',
      tagBg: 'bg-amber-100 text-amber-800',
    },
    {
      name: 'Normal (Healthy)',
      color: 'emerald',
      badge: 'Non-Malignant',
      description: 'Clear lung parenchyma with healthy broncho-vascular arborization and absence of solitary/multiple nodules.',
      bg: 'bg-emerald-50 border-emerald-200 text-emerald-900',
      tagBg: 'bg-emerald-100 text-emerald-800',
    },
  ];

  const highlights = [
    {
      icon: Layers,
      title: 'Multi-Scale Feature Extraction',
      description: 'Simultaneously captures shallow fine-grained textures (Block 3), intermediate morphology (Block 4), and deep semantics (Block 5).',
    },
    {
      icon: Cpu,
      title: 'Scale-Gated Fusion Unit (SGFU)',
      description: 'Learns adaptive continuous gating weights (α3, α4, α5) conditioned on global average and max descriptors.',
    },
    {
      icon: Eye,
      title: 'Grad-CAM Explainability',
      description: 'Visualizes gradient-weighted spatial activations on the fused 16×16 feature tensor for clinical transparency.',
    },
    {
      icon: Database,
      title: 'Zero Leakage Validation',
      description: 'Strict 70/10/20 train/val/test data partition. Online augmentations applied exclusively to training set.',
    },
  ];

  return (
    <div className="space-y-16 py-8">
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-3xl bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 text-white p-8 md:p-14 shadow-2xl border border-slate-700">
        <div className="absolute inset-0 bg-[radial-gradient(#38bdf8_1px,transparent_1px)] [background-size:24px_24px] opacity-10" />
        <div className="relative z-10 max-w-4xl space-y-6">
          <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-medical-500/10 border border-medical-500/30 text-medical-300 text-xs font-semibold tracking-wide">
            <Sparkles className="w-3.5 h-3.5 text-teal-400" />
            <span>Multi-Scale Gated Feature Fusion Deep Learning Architecture</span>
          </div>

          <h1 className="text-3xl sm:text-4xl md:text-5xl font-black tracking-tight text-white leading-tight">
            Lung Cancer Classification <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-medical-400 via-teal-300 to-teal-400">
              Using MS-GF-VGG16
            </span>
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed max-w-3xl font-normal">
            A state-of-the-art Computer-Aided Diagnosis (CAD) system leveraging ImageNet-pretrained VGG16 
            and a novel trainable <strong className="text-white">Scale-Gated Feature Fusion Unit (SGFU)</strong> to classify 
            chest computed tomography (CT) scans into four distinct clinical categories with verifiable explainability.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-4">
            <button
              onClick={() => setActiveTab('predict')}
              className="inline-flex items-center space-x-2.5 bg-gradient-to-r from-medical-500 to-teal-500 hover:from-medical-600 hover:to-teal-600 text-white px-7 py-3.5 rounded-xl font-bold shadow-lg shadow-medical-500/30 hover:scale-[1.02] transition-all cursor-pointer"
            >
              <Activity className="w-5 h-5" />
              <span>Analyze CT Scan</span>
              <ArrowRight className="w-4 h-4 ml-1" />
            </button>

            <button
              onClick={() => setActiveTab('performance')}
              className="inline-flex items-center space-x-2 bg-slate-800/80 hover:bg-slate-700 text-slate-200 border border-slate-600 px-6 py-3.5 rounded-xl font-semibold hover:text-white transition-all cursor-pointer"
            >
              <BarChart2 className="w-4 h-4 text-teal-400" />
              <span>View Evaluation Metrics</span>
            </button>

            <button
              onClick={() => setActiveTab('architecture')}
              className="inline-flex items-center space-x-2 bg-transparent hover:bg-slate-800/50 text-slate-300 px-5 py-3.5 rounded-xl font-medium text-sm transition-all cursor-pointer"
            >
              <Cpu className="w-4 h-4 text-medical-400" />
              <span>Model Pipeline</span>
            </button>
          </div>

          {/* Quick Metrics Bar */}
          <div className="pt-8 border-t border-slate-700/80 grid grid-cols-2 sm:grid-cols-4 gap-4 text-left">
            <div>
              <p className="text-xs text-slate-400 font-medium">Dataset Total</p>
              <p className="text-xl sm:text-2xl font-bold text-white">1,000 Scans</p>
              <p className="text-[11px] text-teal-400">IQ-OTH/NCCD Benchmark</p>
            </div>
            <div>
              <p className="text-xs text-slate-400 font-medium">Model Input</p>
              <p className="text-xl sm:text-2xl font-bold text-white">128 × 128 × 3</p>
              <p className="text-[11px] text-slate-400">Zero-Centered BGR</p>
            </div>
            <div>
              <p className="text-xs text-slate-400 font-medium">Feature Hierarchy</p>
              <p className="text-xl sm:text-2xl font-bold text-white">Blocks 3, 4 & 5</p>
              <p className="text-[11px] text-teal-400">Aligned to 16×16×128</p>
            </div>
            <div>
              <p className="text-xs text-slate-400 font-medium">Gating Weights</p>
              <p className="text-xl sm:text-2xl font-bold text-white">Trainable α3, α4, α5</p>
              <p className="text-[11px] text-teal-400">Softmax Normalization</p>
            </div>
          </div>
        </div>
      </section>

      {/* 4 Diagnostic Categories */}
      <section className="space-y-6">
        <div className="text-center max-w-2xl mx-auto space-y-2">
          <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
            Target Diagnostic Classifications
          </h2>
          <p className="text-sm text-slate-600">
            The MS-GF-VGG16 model performs multi-class discrimination across the 4 major pulmonary histological types.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {categories.map((cat) => (
            <div 
              key={cat.name} 
              className={`p-6 rounded-2xl border ${cat.bg} shadow-sm hover:shadow-md transition-shadow flex flex-col justify-between`}
            >
              <div className="space-y-3">
                <span className={`inline-block px-2.5 py-1 rounded-full text-xs font-semibold ${cat.tagBg}`}>
                  {cat.badge}
                </span>
                <h3 className="text-lg font-bold tracking-tight">{cat.name}</h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  {cat.description}
                </p>
              </div>
              <div className="pt-4 mt-4 border-t border-slate-200/60 flex items-center justify-between text-xs font-medium">
                <span className="text-slate-500">CT Image Format</span>
                <span className="font-mono text-slate-700">JPG • PNG</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Architectural Contributions & Highlights */}
      <section className="bg-white rounded-3xl p-8 sm:p-12 border border-slate-200 shadow-sm space-y-10">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div className="space-y-2">
            <span className="text-xs font-bold uppercase tracking-wider text-medical-600">
              Methodological Innovation
            </span>
            <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
              Why Multi-Scale Gated Feature Fusion?
            </h2>
          </div>
          <p className="text-xs sm:text-sm text-slate-500 max-w-md">
            Conventional architectures like standard VGG16 discard intermediate feature maps, 
            losing critical localized micro-textures necessary to distinguish adenocarcinoma from squamous cell carcinoma.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {highlights.map((item, idx) => {
            const Icon = item.icon;
            return (
              <div key={idx} className="p-6 rounded-2xl bg-slate-50 border border-slate-200/80 hover:border-medical-300 transition-colors space-y-3">
                <div className="w-10 h-10 rounded-xl bg-medical-50 text-medical-600 flex items-center justify-center font-bold">
                  <Icon className="w-5 h-5" />
                </div>
                <h3 className="text-base font-bold text-slate-900">{item.title}</h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  {item.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* Workflow steps */}
        <div className="pt-8 border-t border-slate-100">
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-500 text-center mb-8">
            End-to-End Diagnostic Pipeline
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="flex items-start space-x-4 p-5 rounded-xl bg-white border border-slate-200 shadow-xs">
              <span className="w-8 h-8 rounded-full bg-medical-100 text-medical-700 font-bold flex items-center justify-center text-sm shrink-0">1</span>
              <div>
                <h4 className="text-sm font-bold text-slate-900">Upload & Preprocess</h4>
                <p className="text-xs text-slate-500 mt-1">Image validation, 128×128 spatial resizing, and ImageNet channel zero-centering.</p>
              </div>
            </div>

            <div className="flex items-start space-x-4 p-5 rounded-xl bg-white border border-slate-200 shadow-xs">
              <span className="w-8 h-8 rounded-full bg-teal-100 text-teal-700 font-bold flex items-center justify-center text-sm shrink-0">2</span>
              <div>
                <h4 className="text-sm font-bold text-slate-900">MS-GF-VGG16 Inference</h4>
                <p className="text-xs text-slate-500 mt-1">Blocks 3, 4, 5 projection, spatial alignment, and SGFU dynamic weighting.</p>
              </div>
            </div>

            <div className="flex items-start space-x-4 p-5 rounded-xl bg-white border border-slate-200 shadow-xs">
              <span className="w-8 h-8 rounded-full bg-indigo-100 text-indigo-700 font-bold flex items-center justify-center text-sm shrink-0">3</span>
              <div>
                <h4 className="text-sm font-bold text-slate-900">Explainable Output</h4>
                <p className="text-xs text-slate-500 mt-1">Four-class posterior probabilities, confidence score, and Grad-CAM saliency heatmaps.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Academic Disclaimer Callout */}
      <div className="p-5 rounded-2xl bg-amber-50 border border-amber-200/80 flex items-start space-x-3.5">
        <ShieldCheck className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
        <div className="text-xs text-amber-800 leading-relaxed">
          <strong className="font-semibold text-amber-900">Academic Project Disclaimer: </strong>
          This application is an academic AI research prototype developed for final-year engineering demonstration.
          Predictions and Grad-CAM heatmaps do not constitute certified medical advice, biopsy validation, or definitive diagnostic conclusions.
        </div>
      </div>
    </div>
  );
}

