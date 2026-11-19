import React from 'react';
import { ShieldAlert, Award, FileCode2, Layers } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="bg-slate-900 text-slate-400 text-sm border-t border-slate-800 mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Column 1: Info */}
          <div className="md:col-span-2 space-y-3">
            <div className="flex items-center space-x-2 text-white font-bold text-base">
              <span className="w-2.5 h-2.5 rounded-full bg-teal-400" />
              <span>MS-GF-VGG16 Clinical Decision Support System</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed max-w-lg">
              A deep convolutional neural network incorporating Multi-Scale Gated Feature Fusion (MS-GF)
              with an ImageNet-pretrained VGG16 backbone for four-class pulmonary neoplasm classification
              from high-resolution computed tomography (CT) imaging.
            </p>
            <div className="flex items-center space-x-4 pt-2 text-xs text-slate-400">
              <span className="flex items-center space-x-1.5"><Layers className="w-3.5 h-3.5 text-medical-400" /><span>IQ-OTH/NCCD Benchmark</span></span>
              <span className="flex items-center space-x-1.5"><Award className="w-3.5 h-3.5 text-teal-400" /><span>B.Tech AI Capstone</span></span>
              <span className="flex items-center space-x-1.5"><FileCode2 className="w-3.5 h-3.5 text-indigo-400" /><span>FastAPI + PyTorch/Keras</span></span>
            </div>
          </div>

          {/* Column 2: Classes */}
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-200 mb-3">Diagnostic Classes</h4>
            <ul className="space-y-1.5 text-xs text-slate-400">
              <li className="flex items-center space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                <span>Adenocarcinoma</span>
              </li>
              <li className="flex items-center space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-400" />
                <span>Large Cell Carcinoma</span>
              </li>
              <li className="flex items-center space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                <span>Squamous Cell Carcinoma</span>
              </li>
              <li className="flex items-center space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                <span>Normal (Non-malignant)</span>
              </li>
            </ul>
          </div>

          {/* Column 3: Medical Disclaimer */}
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-amber-400 flex items-center space-x-1 mb-3">
              <ShieldAlert className="w-3.5 h-3.5" />
              <span>Academic Prototype</span>
            </h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              This application is strictly an academic AI research prototype developed for educational and experimental demonstration. 
              It is not FDA/CE-approved and must never be utilized as a substitute for certified medical diagnosis, radiology reporting, or clinical decision making.
            </p>
          </div>
        </div>

        <div className="border-t border-slate-800/80 pt-6 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500">
          <p>© {new Date().getFullYear()} MS-GF-VGG16 AI Research. All evaluation metrics computed on genuine test partition.</p>
          <p className="mt-2 sm:mt-0 font-mono text-[11px] text-slate-400">
            FastAPI Backend • Keras Functional API • React 18 + Tailwind
          </p>
        </div>
      </div>
    </footer>
  );
}

