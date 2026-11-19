import React, { useState, useRef } from 'react';
import { 
  Upload, 
  Image as ImageIcon, 
  Trash2, 
  Activity, 
  AlertCircle, 
  CheckCircle2, 
  Eye, 
  Layers, 
  RefreshCw,
  Sparkles,
  Info,
  ShieldAlert
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  Cell 
} from 'recharts';
import api from '../services/api';

export default function PredictionPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [viewMode, setViewMode] = useState('overlay'); // 'overlay' | 'heatmap' | 'original' | 'split'
  const [samples, setSamples] = useState([]);
  const fileInputRef = useRef(null);

  React.useEffect(() => {
    const fetchSamples = async () => {
      try {
        const res = await api.getSampleImages();
        if (res.samples) setSamples(res.samples);
      } catch (e) {
        console.error('Failed loading test samples:', e);
      }
    };
    fetchSamples();
  }, []);

  const handleSelectSample = async (sample) => {
    try {
      // Convert base64 data_url to File object
      const res = await fetch(sample.data_url);
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: blob.type });
      handleFileSelect(file);
    } catch (e) {
      setError('Failed to load sample image.');
    }
  };

  const handleFileSelect = (file) => {
    if (!file) return;

    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type) && !/\.(jpe?g|png)$/i.test(file.name)) {
      setError('Please upload a valid JPG, JPEG, or PNG CT scan image.');
      return;
    }

    if (file.size > 15 * 1024 * 1024) {
      setError('Image file size exceeds maximum limit of 15MB.');
      return;
    }

    setError(null);
    setResult(null);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const data = await api.predictImage(selectedFile, true);
      setResult(data);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Failed to analyze CT scan.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const getClassBadgeColor = (classId) => {
    switch (classId) {
      case 'adenocarcinoma':
        return { badge: 'bg-blue-100 text-blue-800 border-blue-200', bar: '#3b82f6' };
      case 'large_cell_carcinoma':
        return { badge: 'bg-teal-100 text-teal-800 border-teal-200', bar: '#0d9488' };
      case 'squamous_cell_carcinoma':
        return { badge: 'bg-amber-100 text-amber-800 border-amber-200', bar: '#d97706' };
      case 'normal':
        return { badge: 'bg-emerald-100 text-emerald-800 border-emerald-200', bar: '#10b981' };
      default:
        return { badge: 'bg-slate-100 text-slate-800 border-slate-200', bar: '#64748b' };
    }
  };

  return (
    <div className="space-y-10 py-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="space-y-2">
        <div className="flex items-center space-x-2 text-medical-600 text-xs font-bold uppercase tracking-wider">
          <Activity className="w-4 h-4" />
          <span>Computer-Aided Pulmonary Diagnostic Studio</span>
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          CT Scan Image Analysis & Inference
        </h1>
        <p className="text-sm text-slate-600 max-w-3xl">
          Upload a high-resolution chest computed tomography (CT) axial cross-section to generate real-time 
          probabilistic classifications and multi-scale Grad-CAM attention saliency maps.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Upload & Controls (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-6">
            <h2 className="text-base font-bold text-slate-900 flex items-center justify-between">
              <span>Upload CT Scan</span>
              {selectedFile && (
                <button
                  onClick={handleClear}
                  className="text-xs text-rose-600 hover:text-rose-700 font-medium flex items-center space-x-1"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Remove Image</span>
                </button>
              )}
            </h2>

            {/* Drag & Drop Area */}
            {!previewUrl ? (
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-300 hover:border-medical-500 rounded-2xl p-8 text-center cursor-pointer transition-colors bg-slate-50/50 hover:bg-medical-50/20 group"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/jpg,image/png"
                  onChange={(e) => handleFileSelect(e.target.files[0])}
                  className="hidden"
                />
                <div className="w-14 h-14 mx-auto rounded-2xl bg-white shadow-sm border border-slate-200 flex items-center justify-center text-slate-400 group-hover:text-medical-600 group-hover:scale-105 transition-all">
                  <Upload className="w-6 h-6" />
                </div>
                <div className="mt-4 space-y-1">
                  <p className="text-sm font-semibold text-slate-800">
                    Drag and drop CT image here
                  </p>
                  <p className="text-xs text-slate-500">
                    or <span className="text-medical-600 underline">browse from your computer</span>
                  </p>
                </div>
                <p className="mt-3 text-[11px] text-slate-400 font-mono">
                  Supported formats: JPG, JPEG, PNG (up to 15MB)
                </p>
              </div>
            ) : (
              /* Selected Image Preview */
              <div className="space-y-4">
                <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-200 shadow-inner aspect-square flex items-center justify-center group">
                  <img
                    src={previewUrl}
                    alt="CT Preview"
                    className="max-h-full max-w-full object-contain"
                  />
                  <div className="absolute inset-0 bg-slate-900/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="px-3 py-1.5 rounded-lg bg-white text-slate-900 text-xs font-semibold shadow-md hover:bg-slate-100"
                    >
                      Replace Image
                    </button>
                    <button
                      onClick={handleClear}
                      className="px-3 py-1.5 rounded-lg bg-rose-600 text-white text-xs font-semibold shadow-md hover:bg-rose-700"
                    >
                      Delete
                    </button>
                  </div>
                </div>

                <div className="flex items-center justify-between text-xs text-slate-500 font-mono bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                  <span className="truncate max-w-[200px]">{selectedFile?.name}</span>
                  <span>{selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB` : ''}</span>
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/jpg,image/png"
                  onChange={(e) => handleFileSelect(e.target.files[0])}
                  className="hidden"
                />
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-start space-x-2.5">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <span className="leading-relaxed">{error}</span>
              </div>
            )}

            {/* Quick Test Samples */}
            {samples.length > 0 && !previewUrl && (
              <div className="space-y-2 pt-1 border-t border-slate-100">
                <p className="text-xs font-semibold text-slate-700 flex items-center space-x-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-medical-600" />
                  <span>Or select a sample test CT scan:</span>
                </p>
                <div className="grid grid-cols-2 gap-2">
                  {samples.map((s) => (
                    <button
                      key={s.class_id}
                      onClick={() => handleSelectSample(s)}
                      className="p-2 rounded-xl bg-slate-50 hover:bg-medical-50/70 border border-slate-200 hover:border-medical-300 transition-all text-left flex items-center space-x-2.5 cursor-pointer group"
                    >
                      <img
                        src={s.data_url}
                        alt={s.display_name}
                        className="w-9 h-9 rounded-lg object-cover bg-black shrink-0 border border-slate-200"
                      />
                      <div className="overflow-hidden">
                        <p className="text-[11px] font-bold text-slate-800 group-hover:text-medical-700 truncate">
                          {s.display_name}
                        </p>
                        <p className="text-[10px] text-slate-400">Click to test</p>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Analyze Button */}
            <button
              onClick={handleAnalyze}
              disabled={!selectedFile || loading}
              className={`w-full py-3.5 px-4 rounded-xl font-bold text-sm shadow-md transition-all flex items-center justify-center space-x-2 ${
                !selectedFile || loading
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed shadow-none'
                  : 'bg-gradient-to-r from-medical-600 to-teal-600 hover:from-medical-700 hover:to-teal-700 text-white shadow-medical-600/25 hover:shadow-lg cursor-pointer'
              }`}
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Processing MS-GF-VGG16 Inference...</span>
                </>
              ) : (
                <>
                  <Activity className="w-4 h-4" />
                  <span>Analyze CT Scan with MS-GF-VGG16</span>
                </>
              )}
            </button>
          </div>

          {/* Guidelines Box */}
          <div className="p-5 rounded-2xl bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-2">
            <div className="flex items-center space-x-2 text-slate-800 font-semibold">
              <Info className="w-4 h-4 text-medical-600" />
              <span>Input Preprocessing Specifications</span>
            </div>
            <ul className="list-disc list-inside space-y-1 text-slate-500 pl-1 leading-relaxed">
              <li>Input tensors are automatically resized to 128×128×3 via bilinear interpolation.</li>
              <li>RGB to BGR conversion and ImageNet channel-mean zero-centering applied.</li>
              <li>Grad-CAM gradient propagation runs with respect to the fused SGFU layer.</li>
            </ul>
          </div>
        </div>

        {/* Right Column: Prediction Results & Explainability (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {!result && !loading && (
            <div className="bg-white rounded-2xl border border-dashed border-slate-300 p-12 text-center text-slate-400 space-y-3">
              <div className="w-12 h-12 mx-auto rounded-full bg-slate-100 flex items-center justify-center text-slate-400">
                <ImageIcon className="w-6 h-6" />
              </div>
              <h3 className="text-base font-semibold text-slate-700">Awaiting Image Submission</h3>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                Upload a lung CT scan and click "Analyze CT Scan" to inspect model predictions, confidence metrics, and Grad-CAM saliency maps.
              </p>
            </div>
          )}

          {loading && (
            <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center space-y-4 shadow-sm">
              <div className="relative w-16 h-16 mx-auto">
                <div className="w-16 h-16 rounded-full border-4 border-medical-100 border-t-medical-600 animate-spin" />
                <Activity className="w-6 h-6 text-medical-600 absolute inset-0 m-auto animate-pulse" />
              </div>
              <div className="space-y-1">
                <h3 className="text-base font-bold text-slate-800">Executing Deep Feature Extraction</h3>
                <p className="text-xs text-slate-500">
                  Extracting Blocks 3, 4, 5 feature maps, computing SGFU attention gating, and generating Grad-CAM heatmaps...
                </p>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-6 animate-in fade-in duration-300">
              {/* Primary Prediction Banner Card */}
              <div className="bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-100">
                  <div className="space-y-1">
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                      Diagnostic Prediction
                    </span>
                    <div className="flex items-center space-x-3">
                      <h2 className="text-2xl sm:text-3xl font-black text-slate-900 tracking-tight">
                        {result.display_name}
                      </h2>
                      <span className={`px-3 py-1 rounded-full text-xs font-bold border ${getClassBadgeColor(result.predicted_class).badge}`}>
                        {result.confidence_percentage}% Confidence
                      </span>
                    </div>
                  </div>

                  <div className="text-left sm:text-right">
                    <span className="text-xs text-slate-400 font-medium">Model Confidence</span>
                    <p className="text-3xl font-extrabold text-medical-600">
                      {(result.confidence * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>

                {/* Probability Distribution Chart */}
                <div className="space-y-3">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                    Full Posterior Probability Distribution
                  </h3>
                  <div className="h-48 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={result.probabilities}
                        layout="vertical"
                        margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                      >
                        <XAxis 
                          type="number" 
                          domain={[0, 100]} 
                          tickFormatter={(v) => `${v}%`}
                          fontSize={11}
                          stroke="#94a3b8"
                        />
                        <YAxis 
                          type="category" 
                          dataKey="display_name" 
                          fontSize={11}
                          stroke="#64748b"
                          width={140}
                          tickLine={false}
                        />
                        <Tooltip
                          formatter={(value) => [`${value}%`, 'Probability']}
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                        />
                        <Bar dataKey="percentage" radius={[0, 6, 6, 0]}>
                          {result.probabilities.map((entry, index) => (
                            <Cell 
                              key={`cell-${index}`} 
                              fill={getClassBadgeColor(entry.class_id).bar} 
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Probabilities Breakdown Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                  {result.probabilities.map((p) => {
                    const isSelected = p.class_id === result.predicted_class;
                    return (
                      <div
                        key={p.class_id}
                        className={`p-3 rounded-xl border text-center transition-all ${
                          isSelected
                            ? 'bg-medical-50/70 border-medical-300 ring-2 ring-medical-500/20 shadow-xs'
                            : 'bg-slate-50 border-slate-200'
                        }`}
                      >
                        <p className="text-[11px] font-semibold text-slate-500 truncate">{p.display_name}</p>
                        <p className="text-base font-bold text-slate-900 mt-1">{p.percentage}%</p>
                        <p className="text-[10px] text-slate-400 font-mono mt-0.5">{p.probability.toFixed(4)}</p>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Grad-CAM Explainability Section */}
              {result.gradcam && (
                <div className="bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm space-y-6">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div className="space-y-1">
                      <div className="flex items-center space-x-2 text-teal-600 text-xs font-bold uppercase tracking-wider">
                        <Eye className="w-4 h-4" />
                        <span>Explainable AI (XAI) Activation Map</span>
                      </div>
                      <h3 className="text-lg font-bold text-slate-900">
                        Grad-CAM Saliency Visualization
                      </h3>
                    </div>

                    {/* View mode toggle */}
                    <div className="flex items-center space-x-1 p-1 bg-slate-100 rounded-lg text-xs font-medium">
                      <button
                        onClick={() => setViewMode('overlay')}
                        className={`px-3 py-1 rounded-md transition-all ${viewMode === 'overlay' ? 'bg-white text-slate-900 shadow-xs font-bold' : 'text-slate-600 hover:text-slate-900'}`}
                      >
                        Overlay
                      </button>
                      <button
                        onClick={() => setViewMode('heatmap')}
                        className={`px-3 py-1 rounded-md transition-all ${viewMode === 'heatmap' ? 'bg-white text-slate-900 shadow-xs font-bold' : 'text-slate-600 hover:text-slate-900'}`}
                      >
                        Heatmap
                      </button>
                      <button
                        onClick={() => setViewMode('original')}
                        className={`px-3 py-1 rounded-md transition-all ${viewMode === 'original' ? 'bg-white text-slate-900 shadow-xs font-bold' : 'text-slate-600 hover:text-slate-900'}`}
                      >
                        Original
                      </button>
                      <button
                        onClick={() => setViewMode('split')}
                        className={`px-3 py-1 rounded-md transition-all ${viewMode === 'split' ? 'bg-white text-slate-900 shadow-xs font-bold' : 'text-slate-600 hover:text-slate-900'}`}
                      >
                        Side-by-Side
                      </button>
                    </div>
                  </div>

                  {/* Image Display */}
                  {viewMode !== 'split' ? (
                    <div className="relative rounded-2xl overflow-hidden bg-slate-950 aspect-video sm:aspect-[16/10] flex items-center justify-center border border-slate-200 shadow-inner">
                      <img
                        src={
                          viewMode === 'overlay'
                            ? result.gradcam.overlay_image
                            : viewMode === 'heatmap'
                            ? result.gradcam.heatmap_image
                            : result.gradcam.original_image
                        }
                        alt="Grad-CAM Display"
                        className="max-h-full max-w-full object-contain"
                      />
                      <div className="absolute bottom-3 left-3 bg-slate-900/80 backdrop-blur-md px-3 py-1 rounded-md text-[11px] text-white font-mono">
                        Target Layer: {result.gradcam.target_layer}
                      </div>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="rounded-xl overflow-hidden bg-slate-950 aspect-square flex items-center justify-center border border-slate-200">
                        <img
                          src={result.gradcam.original_image}
                          alt="Original CT"
                          className="max-h-full max-w-full object-contain"
                        />
                      </div>
                      <div className="rounded-xl overflow-hidden bg-slate-950 aspect-square flex items-center justify-center border border-slate-200">
                        <img
                          src={result.gradcam.overlay_image}
                          alt="Grad-CAM Overlay"
                          className="max-h-full max-w-full object-contain"
                        />
                      </div>
                    </div>
                  )}

                  <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-1.5">
                    <p className="font-semibold text-slate-800">How to interpret Grad-CAM heatmaps:</p>
                    <p className="text-slate-500 leading-relaxed">
                      Grad-CAM computes the gradients of the predicted class score with respect to the Multi-Scale Gated Feature Fusion layer.
                      Warmer regions (<span className="text-red-500 font-semibold">red/yellow</span>) indicate spatial features that contributed most strongly 
                      to the network's classification decision. Cooler regions (<span className="text-blue-500 font-semibold">blue</span>) had lower influence.
                    </p>
                  </div>
                </div>
              )}

              {/* Medical Notice */}
              <div className="p-4 rounded-xl bg-amber-50 border border-amber-200/80 text-xs text-amber-800 flex items-start space-x-3">
                <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                <p className="leading-relaxed">
                  <strong>Clinical Notice: </strong>
                  This application is an academic AI research prototype intended for educational purposes. 
                  It is not a substitute for professional medical diagnosis. The prediction does not confirm or rule out the presence of malignancy.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
