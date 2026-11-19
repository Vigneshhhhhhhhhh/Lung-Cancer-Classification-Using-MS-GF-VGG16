import React, { useState } from 'react';
import { 
  Cpu, 
  Layers, 
  Zap, 
  GitMerge, 
  ArrowDown, 
  CheckCircle2, 
  Code2, 
  Maximize2,
  Sparkles
} from 'lucide-react';

export default function ArchitecturePage() {
  const [activeStage, setActiveStage] = useState('sgfu');

  const pipelineStages = [
    {
      id: 'input',
      title: '1. Input & Preprocessing',
      tag: '128 × 128 × 3',
      description: 'Axial lung CT scans are standardized to 128×128 RGB, converted to BGR, and zero-centered by subtracting ImageNet channel means [103.939, 116.779, 123.68].',
      detail: 'No duplicate scaling or incompatible [0, 1] normalization is introduced, maintaining transfer learning fidelity.'
    },
    {
      id: 'backbone',
      title: '2. Pretrained VGG16 Backbone',
      tag: 'Blocks 1 to 5',
      description: 'Feature extraction backbone with ImageNet-pretrained weights. The initial 10 layers are frozen to retain general low-level edge and corner detectors.',
      detail: 'Outputs extracted: Block 3 (16×16×256), Block 4 (8×8×512), and Block 5 (4×4×512).'
    },
    {
      id: 'projection',
      title: '3. 1×1 Conv Projection & Alignment',
      tag: 'All 16 × 16 × 128',
      description: 'Individual 1×1 2D convolutions compress all three feature hierarchies into a uniform channel depth of 128 filters.',
      detail: 'Spatial alignment: F3 remains 16×16; F4 is upsampled ×2 to 16×16; F5 is upsampled ×4 to 16×16 using bilinear interpolation.'
    },
    {
      id: 'sgfu',
      title: '4. Scale-Gated Fusion Unit (SGFU)',
      tag: 'Adaptive Continuous Gating',
      description: 'Novel trainable fusion unit that pools average (GAP) and peak (GMP) activations across F3, F4, F5 into a 768-d descriptor, passing through an MLP to learn dynamic softmax weights α3, α4, α5.',
      detail: 'Computes F_fused = α3 F3 + α4 F4 + α5 F5 with α3 + α4 + α5 = 1. Resulting tensor is 16×16×128.'
    },
    {
      id: 'head',
      title: '5. Classification Head & Softmax',
      tag: 'Dense 256 → 128 → 4',
      description: 'Global Average Pooling condenses F_fused into a 128-dimensional vector, followed by Dense(256, ReLU, Dropout 0.5), Dense(128, ReLU, Dropout 0.3), and Dense(4, Softmax).',
      detail: 'Outputs discrete probability distribution over Adenocarcinoma, Large Cell, Squamous Cell, and Normal.'
    }
  ];

  return (
    <div className="space-y-12 py-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="space-y-2">
        <div className="flex items-center space-x-2 text-medical-600 text-xs font-bold uppercase tracking-wider">
          <Cpu className="w-4 h-4" />
          <span>Deep Learning Network Architecture</span>
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          MS-GF-VGG16 Architectural Specification
        </h1>
        <p className="text-sm text-slate-600 max-w-3xl">
          Multi-Scale Gated Feature Fusion VGG16 (MS-GF-VGG16) resolves the spatial-semantic dilemma 
          in medical CT imaging by adaptively fusing low-level micro-textures and high-level pathological context.
        </p>
      </div>

      {/* SGFU Mathematical Contribution Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 rounded-3xl p-8 sm:p-10 text-white shadow-xl border border-slate-700 space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <span className="text-xs font-bold text-teal-400 uppercase tracking-wider flex items-center space-x-1.5">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Core Architectural Contribution</span>
            </span>
            <h2 className="text-2xl font-black text-white">
              Scale-Gated Feature Fusion Equation
            </h2>
          </div>
          <span className="px-3 py-1 rounded-full bg-teal-500/20 border border-teal-500/40 text-teal-300 text-xs font-mono">
            Trainable End-to-End
          </span>
        </div>

        {/* Formula Box */}
        <div className="p-6 rounded-2xl bg-slate-950/80 border border-slate-800 text-center font-mono text-lg sm:text-2xl text-teal-300 shadow-inner">
          <span className="text-white font-bold">F</span>
          <span className="text-xs text-slate-400 align-sub">fused</span>
          <span className="text-white mx-3">=</span>
          <span className="text-medical-400 font-bold">α</span>
          <span className="text-xs text-slate-400 align-sub">3</span>
          <span className="text-slate-200"> · F</span>
          <span className="text-xs text-slate-400 align-sub">3</span>
          <span className="text-white mx-3">+</span>
          <span className="text-teal-400 font-bold">α</span>
          <span className="text-xs text-slate-400 align-sub">4</span>
          <span className="text-slate-200"> · F</span>
          <span className="text-xs text-slate-400 align-sub">4</span>
          <span className="text-white mx-3">+</span>
          <span className="text-amber-400 font-bold">α</span>
          <span className="text-xs text-slate-400 align-sub">5</span>
          <span className="text-slate-200"> · F</span>
          <span className="text-xs text-slate-400 align-sub">5</span>
          <div className="text-xs text-slate-400 mt-2 font-sans">
            Constraint: α3 + α4 + α5 = 1, where α_i = Softmax(MLP([GAP(F_k) || GMP(F_k)]))
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-300">
          <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700/60 space-y-1">
            <p className="font-bold text-medical-300">F3 (Block 3 - Shallow)</p>
            <p className="text-slate-400">Preserves local cellular patterns, ground-glass opacity textures, and fine nodular contours.</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700/60 space-y-1">
            <p className="font-bold text-teal-300">F4 (Block 4 - Intermediate)</p>
            <p className="text-slate-400">Captures tissue morphology, broncho-vascular infiltration, and lobular anatomical margins.</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700/60 space-y-1">
            <p className="font-bold text-amber-300">F5 (Block 5 - Deep)</p>
            <p className="text-slate-400">Encodes high-level tumor semantics, parenchymal destruction, and global malignancy patterns.</p>
          </div>
        </div>
      </div>

      {/* Interactive Step-by-Step Flowchart */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Navigation Sidebar (5 cols) */}
        <div className="lg:col-span-5 space-y-3">
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-500 mb-2">
            Sequential Data Pipeline
          </h3>
          {pipelineStages.map((stage) => {
            const isSelected = activeStage === stage.id;
            return (
              <div
                key={stage.id}
                onClick={() => setActiveStage(stage.id)}
                className={`p-4 rounded-2xl border transition-all cursor-pointer ${
                  isSelected
                    ? 'bg-white border-medical-500 shadow-md ring-2 ring-medical-500/10'
                    : 'bg-white/80 border-slate-200 hover:border-slate-300 hover:bg-white'
                }`}
              >
                <div className="flex items-center justify-between">
                  <h4 className={`text-sm font-bold ${isSelected ? 'text-medical-700' : 'text-slate-800'}`}>
                    {stage.title}
                  </h4>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-semibold">
                    {stage.tag}
                  </span>
                </div>
                <p className="text-xs text-slate-500 mt-1 line-clamp-2">
                  {stage.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* Detailed Stage Card (7 cols) */}
        <div className="lg:col-span-7 bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm space-y-6">
          {(() => {
            const current = pipelineStages.find((s) => s.id === activeStage);
            return (
              <div className="space-y-6 animate-in fade-in duration-200">
                <div className="flex items-center justify-between pb-4 border-b border-slate-100">
                  <div className="space-y-1">
                    <span className="text-xs font-bold text-medical-600 uppercase tracking-wider">
                      Selected Architecture Stage
                    </span>
                    <h3 className="text-2xl font-bold text-slate-900">{current.title}</h3>
                  </div>
                  <span className="px-3 py-1 rounded-full bg-medical-50 border border-medical-200 text-medical-700 text-xs font-mono font-bold">
                    {current.tag}
                  </span>
                </div>

                <div className="space-y-4">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                    Functional Overview
                  </h4>
                  <p className="text-sm text-slate-700 leading-relaxed">
                    {current.description}
                  </p>
                </div>

                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-2">
                  <p className="font-semibold text-slate-800 flex items-center space-x-1.5">
                    <CheckCircle2 className="w-4 h-4 text-teal-600" />
                    <span>Implementation Specification</span>
                  </p>
                  <p className="text-slate-500 leading-relaxed">
                    {current.detail}
                  </p>
                </div>

                {/* Architecture Visual Diagram / Code Snippet */}
                {activeStage === 'sgfu' && (
                  <div className="space-y-3 pt-2">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
                      <Code2 className="w-4 h-4" />
                      <span>SGFU Custom Layer Forward Pass (Keras)</span>
                    </h4>
                    <div className="p-4 rounded-xl bg-slate-900 text-slate-200 font-mono text-[11px] overflow-x-auto leading-relaxed border border-slate-800">
                      <p className="text-slate-400"># Step 2: Global Average & Max Pooling for each scale</p>
                      <p><span className="text-purple-400">gap3</span> = tf.reduce_mean(F3, axis=[1, 2])  <span className="text-slate-500"># (B, 128)</span></p>
                      <p><span className="text-purple-400">gmp3</span> = tf.reduce_max(F3, axis=[1, 2])   <span className="text-slate-500"># (B, 128)</span></p>
                      <p className="text-slate-400 mt-2"># Step 3: Concatenate pooled descriptors</p>
                      <p><span className="text-purple-400">desc</span> = tf.concat([gap3, gmp3, gap4, gmp4, gap5, gmp5], axis=-1)  <span className="text-slate-500"># (B, 768)</span></p>
                      <p className="text-slate-400 mt-2"># Step 4-6: Gating MLP & Softmax normalization</p>
                      <p><span className="text-teal-400">logits</span> = self.dense2(self.dense1(desc))  <span className="text-slate-500"># (B, 3)</span></p>
                      <p><span className="text-teal-400">weights</span> = tf.nn.softmax(logits, axis=-1)  <span className="text-slate-500"># α3, α4, α5</span></p>
                      <p className="text-slate-400 mt-2"># Step 7: Weighted feature fusion</p>
                      <p><span className="text-medical-400 font-bold">F_fused</span> = α3 * F3 + α4 * F4 + α5 * F5  <span className="text-slate-500"># (B, 16, 16, 128)</span></p>
                    </div>
                  </div>
                )}
              </div>
            );
          })()}
        </div>
      </div>

      {/* Tensor Dimension Mapping Matrix */}
      <div className="bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm space-y-4">
        <h3 className="text-base font-bold text-slate-900">
          Tensor Transformation Matrix
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500 font-bold uppercase text-[10px]">
                <th className="pb-3">Component / Sub-module</th>
                <th className="pb-3">Input Shape</th>
                <th className="pb-3">Transformation / Operation</th>
                <th className="pb-3">Output Shape</th>
                <th className="pb-3">Trainable Parameters</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-slate-700">
              <tr>
                <td className="py-2.5 font-sans font-semibold text-slate-900">Input Image</td>
                <td className="py-2.5 text-slate-500">H × W × C</td>
                <td className="py-2.5 font-sans">Resize & VGG16 Preprocess</td>
                <td className="py-2.5 text-medical-600 font-bold">128 × 128 × 3</td>
                <td className="py-2.5 text-slate-400">0</td>
              </tr>
              <tr>
                <td className="py-2.5 font-sans font-semibold text-slate-900">VGG16 Block 3 Output</td>
                <td className="py-2.5 text-slate-500">128 × 128 × 3</td>
                <td className="py-2.5 font-sans">Convolution & MaxPool</td>
                <td className="py-2.5 text-slate-800">16 × 16 × 256</td>
                <td className="py-2.5 text-slate-500">ImageNet Pretrained</td>
              </tr>
              <tr>
                <td className="py-2.5 font-sans font-semibold text-slate-900">VGG16 Block 4 Output</td>
                <td className="py-2.5 text-slate-500">16 × 16 × 256</td>
                <td className="py-2.5 font-sans">Convolution & MaxPool</td>
                <td className="py-2.5 text-slate-800">8 × 8 × 512</td>
                <td className="py-2.5 text-slate-500">ImageNet Pretrained</td>
              </tr>
              <tr>
                <td className="py-2.5 font-sans font-semibold text-slate-900">VGG16 Block 5 Output</td>
                <td className="py-2.5 text-slate-500">8 × 8 × 512</td>
                <td className="py-2.5 font-sans">Convolution & MaxPool</td>
                <td className="py-2.5 text-slate-800">4 × 4 × 512</td>
                <td className="py-2.5 text-slate-500">ImageNet Pretrained</td>
              </tr>
              <tr>
                <td className="py-2.5 font-sans font-semibold text-slate-900">1×1 Projections (F3, F4, F5)</td>
                <td className="py-2.5 text-slate-500">256 / 512 ch</td>
                <td className="py-2.5 font-sans">Conv2D(128, (1, 1), ReLU)</td>
                <td className="py-2.5 text-medical-600">F3: 16×16, F4: 8×8, F5: 4×4</td>
                <td className="py-2.5 text-teal-600">Trainable</td>
              </tr>
              <tr>
                <td className="py-2.5 font-sans font-semibold text-slate-900">Spatial Alignment</td>
                <td className="py-2.5 text-slate-500">8×8 / 4×4</td>
                <td className="py-2.5 font-sans">Bilinear UpSampling2D (×2, ×4)</td>
                <td className="py-2.5 text-slate-800">All 16 × 16 × 128</td>
                <td className="py-2.5 text-slate-400">0</td>
              </tr>
              <tr className="bg-medical-50/50">
                <td className="py-2.5 font-sans font-semibold text-medical-900">SGFU (Proposed Unit)</td>
                <td className="py-2.5 text-slate-500">3 × (16 × 16 × 128)</td>
                <td className="py-2.5 font-sans">GAP+GMP → Dense(128) → Dense(64) → Softmax → Weighted Sum</td>
                <td className="py-2.5 text-medical-700 font-bold">16 × 16 × 128</td>
                <td className="py-2.5 text-teal-600 font-bold">Trainable (106,755)</td>
              </tr>
              <tr>
                <td className="py-2.5 font-sans font-semibold text-slate-900">Head GAP & MLP</td>
                <td className="py-2.5 text-slate-500">16 × 16 × 128</td>
                <td className="py-2.5 font-sans">GAP → Dense(256) → Dense(128) → Dense(4)</td>
                <td className="py-2.5 text-indigo-600 font-bold">4 Class Probabilities</td>
                <td className="py-2.5 text-teal-600">Trainable</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

