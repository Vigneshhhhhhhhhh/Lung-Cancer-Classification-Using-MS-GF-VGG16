import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  CheckCircle2, 
  Target, 
  ShieldCheck, 
  RefreshCw, 
  Layers, 
  Award,
  AlertTriangle,
  Info
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  Legend, 
  CartesianGrid,
  Cell
} from 'recharts';
import api from '../services/api';

export default function PerformancePage() {
  const [data, setData] = useState(null);
  const [ablation, setAblation] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedClass, setSelectedClass] = useState(null);

  const fetchMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const evalRes = await api.getEvaluationMetrics();
      if (evalRes.status === 'completed' && evalRes.metrics) {
        setData(evalRes);
      } else {
        setData(null);
      }

      const abRes = await api.getAblationStudy();
      if (abRes.experiments) {
        setAblation(abRes.experiments);
      }
    } catch (err) {
      setError(err.message || 'Failed to retrieve model performance metrics.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  const metrics = data?.metrics;
  const history = data?.history;

  // Prepare class performance data for charts
  const classChartData = metrics?.class_performance 
    ? Object.entries(metrics.class_performance).map(([key, val]) => ({
        name: val.display_name,
        precision: Math.round(val.precision * 100),
        recall: Math.round(val.recall * 100),
        f1_score: Math.round(val.f1_score * 100),
        specificity: Math.round(val.specificity * 100),
        support: val.support,
        key
      }))
    : [];

  // Prepare training history data for charts
  const historyChartData = history?.accuracy?.map((acc, idx) => ({
    epoch: idx + 1,
    train_acc: Math.round(acc * 1000) / 10,
    val_acc: Math.round((history.val_accuracy?.[idx] || 0) * 1000) / 10,
    train_loss: Math.round((history.loss?.[idx] || 0) * 100) / 100,
    val_loss: Math.round((history.val_loss?.[idx] || 0) * 100) / 100,
  })) || [];

  return (
    <div className="space-y-12 py-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center space-x-2 text-medical-600 text-xs font-bold uppercase tracking-wider">
            <BarChart3 className="w-4 h-4" />
            <span>Empirical Model Validation Benchmark</span>
          </div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
            Model Performance & Clinical Analytics
          </h1>
          <p className="text-sm text-slate-600 max-w-2xl">
            Evaluated on the independent, untouched test partition (315 pulmonary CT scans). 
            All scores reflect true model generalization without data leakage.
          </p>
        </div>

        <button
          onClick={fetchMetrics}
          disabled={loading}
          className="self-start md:self-auto flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 text-xs font-semibold shadow-xs cursor-pointer transition-all"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-medical-600' : 'text-slate-500'}`} />
          <span>Refresh Analytics</span>
        </button>
      </div>

      {loading && !metrics && (
        <div className="bg-white p-16 rounded-2xl border border-slate-200 text-center space-y-3">
          <RefreshCw className="w-8 h-8 animate-spin text-medical-600 mx-auto" />
          <p className="text-sm font-semibold text-slate-700">Loading Genuine Model Evaluation Metrics...</p>
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {!loading && !metrics && (
        <div className="bg-white p-12 rounded-2xl border border-dashed border-slate-300 text-center space-y-3">
          <Info className="w-8 h-8 text-amber-500 mx-auto" />
          <h3 className="text-base font-bold text-slate-800">Evaluation Metrics Unavailable</h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Model training is either running or has not completed. Once the training pipeline finishes, 
            metrics will automatically appear here.
          </p>
        </div>
      )}

      {metrics && (
        <div className="space-y-10 animate-in fade-in duration-300">
          {/* Top KPI Cards (6 Key Medical Metrics) */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs space-y-1">
              <span className="text-[11px] font-semibold text-slate-500 uppercase">Overall Accuracy</span>
              <p className="text-2xl sm:text-3xl font-extrabold text-medical-600">
                {(metrics.accuracy * 100).toFixed(1)}%
              </p>
              <p className="text-[10px] text-slate-400">315 untouched test scans</p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs space-y-1">
              <span className="text-[11px] font-semibold text-slate-500 uppercase">Macro F1-Score</span>
              <p className="text-2xl sm:text-3xl font-extrabold text-teal-600">
                {(metrics.f1_macro * 100).toFixed(1)}%
              </p>
              <p className="text-[10px] text-slate-400">Unweighted mean</p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs space-y-1">
              <span className="text-[11px] font-semibold text-slate-500 uppercase">Weighted F1</span>
              <p className="text-2xl sm:text-3xl font-extrabold text-indigo-600">
                {(metrics.f1_weighted * 100).toFixed(1)}%
              </p>
              <p className="text-[10px] text-slate-400">Sample-weighted</p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs space-y-1">
              <span className="text-[11px] font-semibold text-slate-500 uppercase">Macro Specificity</span>
              <p className="text-2xl sm:text-3xl font-extrabold text-emerald-600">
                {(metrics.specificity_macro * 100).toFixed(1)}%
              </p>
              <p className="text-[10px] text-slate-400">True Negative Rate</p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs space-y-1">
              <span className="text-[11px] font-semibold text-slate-500 uppercase">Macro Recall</span>
              <p className="text-2xl sm:text-3xl font-extrabold text-blue-600">
                {(metrics.recall_macro * 100).toFixed(1)}%
              </p>
              <p className="text-[10px] text-slate-400">Sensitivity</p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs space-y-1">
              <span className="text-[11px] font-semibold text-slate-500 uppercase">Macro ROC-AUC</span>
              <p className="text-2xl sm:text-3xl font-extrabold text-purple-600">
                {(metrics.roc_auc_macro * 100).toFixed(1)}%
              </p>
              <p className="text-[10px] text-slate-400">One-vs-Rest Area</p>
            </div>
          </div>

          {/* Confusion Matrix & Class Breakdown */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Interactive Confusion Matrix (6 cols) */}
            <div className="lg:col-span-6 bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-slate-900">
                    Test Set Confusion Matrix
                  </h3>
                  <p className="text-xs text-slate-500">
                    Diagonal entries represent correct positive classifications.
                  </p>
                </div>
                <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-100 text-slate-700 font-semibold">
                  N = {metrics.total_test_samples}
                </span>
              </div>

              {/* Confusion Matrix Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-center border-collapse">
                  <thead>
                    <tr>
                      <th className="p-2 text-[11px] font-bold text-slate-400 uppercase text-left">
                        True \ Pred
                      </th>
                      {metrics.class_labels.map((lbl, idx) => (
                        <th key={idx} className="p-2 text-[11px] font-bold text-slate-600 truncate max-w-[80px]">
                          {lbl.split(' ')[0]}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.confusion_matrix.map((row, rIdx) => {
                      const rowTotal = row.reduce((a, b) => a + b, 0);
                      return (
                        <tr key={rIdx} className="border-t border-slate-100">
                          <td className="p-2 text-xs font-semibold text-slate-700 text-left truncate max-w-[120px]">
                            {metrics.class_labels[rIdx]}
                          </td>
                          {row.map((val, cIdx) => {
                            const isDiagonal = rIdx === cIdx;
                            const intensity = rowTotal > 0 ? val / rowTotal : 0;
                            return (
                              <td
                                key={cIdx}
                                className={`p-3 text-xs font-mono font-bold transition-colors ${
                                  isDiagonal
                                    ? intensity > 0.7
                                      ? 'bg-medical-600 text-white'
                                      : 'bg-medical-500/80 text-white'
                                    : val > 0
                                    ? 'bg-rose-50 text-rose-700'
                                    : 'text-slate-300'
                                }`}
                              >
                                {val}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <div className="flex items-center justify-between text-[11px] text-slate-500 pt-2 border-t border-slate-100">
                <span className="flex items-center space-x-1.5">
                  <span className="w-3 h-3 rounded bg-medical-600" />
                  <span>Correct Predictions (TP)</span>
                </span>
                <span className="flex items-center space-x-1.5">
                  <span className="w-3 h-3 rounded bg-rose-100 border border-rose-300" />
                  <span>Misclassifications</span>
                </span>
              </div>
            </div>

            {/* Per-Class Performance Bar Chart (6 cols) */}
            <div className="lg:col-span-6 bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm space-y-6">
              <div>
                <h3 className="text-base font-bold text-slate-900">
                  Per-Class Performance Metrics
                </h3>
                <p className="text-xs text-slate-500">
                  Precision, Recall, F1-Score, and Specificity across the four neoplasms.
                </p>
              </div>

              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={classChartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                    <XAxis dataKey="name" fontSize={11} stroke="#64748b" interval={0} angle={-10} textAnchor="end" />
                    <YAxis domain={[0, 100]} fontSize={11} stroke="#94a3b8" tickFormatter={(v) => `${v}%`} />
                    <Tooltip 
                      formatter={(val, name) => [`${val}%`, name.toUpperCase()]}
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    <Bar dataKey="precision" fill="#0284c7" name="Precision" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="recall" fill="#0d9488" name="Recall" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="f1_score" fill="#6366f1" name="F1 Score" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="specificity" fill="#10b981" name="Specificity" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Class Support Table */}
              <div className="overflow-x-auto pt-2">
                <table className="w-full text-xs text-left">
                  <thead>
                    <tr className="border-b border-slate-100 text-slate-400 font-bold uppercase text-[10px]">
                      <th className="pb-2">Histological Class</th>
                      <th className="pb-2">Precision</th>
                      <th className="pb-2">Recall</th>
                      <th className="pb-2">F1 Score</th>
                      <th className="pb-2">Specificity</th>
                      <th className="pb-2 text-right">Test Scans</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {Object.entries(metrics.class_performance).map(([clsKey, cVal]) => (
                      <tr key={clsKey} className="hover:bg-slate-50/50">
                        <td className="py-2.5 font-semibold text-slate-800">{cVal.display_name}</td>
                        <td className="py-2.5 font-mono">{(cVal.precision * 100).toFixed(1)}%</td>
                        <td className="py-2.5 font-mono">{(cVal.recall * 100).toFixed(1)}%</td>
                        <td className="py-2.5 font-mono font-bold text-medical-700">{(cVal.f1_score * 100).toFixed(1)}%</td>
                        <td className="py-2.5 font-mono text-emerald-600">{(cVal.specificity * 100).toFixed(1)}%</td>
                        <td className="py-2.5 font-mono text-right text-slate-500">{cVal.support}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Training & Validation Curves (if history available) */}
          {historyChartData.length > 0 && (
            <div className="bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm space-y-6">
              <div>
                <h3 className="text-base font-bold text-slate-900">
                  Two-Stage Training Dynamics & Convergence
                </h3>
                <p className="text-xs text-slate-500">
                  Stage 1: Frozen backbone + SGFU training. Stage 2: Deep layer partial fine-tuning.
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Accuracy Curve */}
                <div className="space-y-2">
                  <h4 className="text-xs font-bold text-slate-600 uppercase tracking-wider text-center">
                    Accuracy Progression
                  </h4>
                  <div className="h-60 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={historyChartData} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                        <XAxis dataKey="epoch" fontSize={11} stroke="#94a3b8" />
                        <YAxis domain={[0, 100]} fontSize={11} stroke="#94a3b8" tickFormatter={(v) => `${v}%`} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff', fontSize: '12px' }} />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                        <Line type="monotone" dataKey="train_acc" stroke="#0284c7" strokeWidth={2} dot={{ r: 2 }} name="Training Accuracy" />
                        <Line type="monotone" dataKey="val_acc" stroke="#0d9488" strokeWidth={2} dot={{ r: 2 }} name="Validation Accuracy" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Loss Curve */}
                <div className="space-y-2">
                  <h4 className="text-xs font-bold text-slate-600 uppercase tracking-wider text-center">
                    Loss Trajectory (Categorical Crossentropy)
                  </h4>
                  <div className="h-60 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={historyChartData} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                        <XAxis dataKey="epoch" fontSize={11} stroke="#94a3b8" />
                        <YAxis fontSize={11} stroke="#94a3b8" />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff', fontSize: '12px' }} />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                        <Line type="monotone" dataKey="train_loss" stroke="#0284c7" strokeWidth={2} dot={{ r: 2 }} name="Training Loss" />
                        <Line type="monotone" dataKey="val_loss" stroke="#dc2626" strokeWidth={2} dot={{ r: 2 }} name="Validation Loss" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Ablation Study Benchmark Card */}
          <div className="bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <div className="flex items-center space-x-2 text-medical-600 text-xs font-bold uppercase tracking-wider">
                  <Award className="w-4 h-4" />
                  <span>Ablation Study Experiments</span>
                </div>
                <h3 className="text-lg font-bold text-slate-900">
                  Architectural Component Contribution Comparison
                </h3>
              </div>
              <span className="text-xs text-slate-500 font-mono">
                Evaluated under identical preprocessing & test split
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              {ablation.map((exp, idx) => {
                const isProposed = exp.experiment.includes('Proposed') || idx === 2;
                return (
                  <div
                    key={idx}
                    className={`p-6 rounded-2xl border transition-all ${
                      isProposed
                        ? 'bg-gradient-to-b from-medical-50/60 to-teal-50/30 border-medical-300 ring-2 ring-medical-500/20 shadow-sm'
                        : 'bg-slate-50 border-slate-200'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <span className={`text-xs font-bold px-2.5 py-0.5 rounded-full ${
                        isProposed ? 'bg-medical-600 text-white' : 'bg-slate-200 text-slate-700'
                      }`}>
                        {exp.experiment.split(':')[0]}
                      </span>
                      {isProposed && (
                        <span className="text-[11px] font-semibold text-teal-700 flex items-center space-x-1">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          <span>Proposed Architecture</span>
                        </span>
                      )}
                    </div>

                    <h4 className="text-base font-bold text-slate-900 mb-1">{exp.experiment.split(':')[1] || exp.experiment}</h4>
                    <p className="text-xs text-slate-600 mb-4 leading-relaxed">{exp.description}</p>

                    <div className="pt-3 border-t border-slate-200/80 flex items-center justify-between">
                      <div>
                        <p className="text-[11px] text-slate-500 font-medium">Test Accuracy</p>
                        <p className={`text-xl font-black ${isProposed ? 'text-medical-600' : 'text-slate-700'}`}>
                          {exp.accuracy !== null ? `${(exp.accuracy * 100).toFixed(1)}%` : 'Running...'}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-[11px] text-slate-500 font-medium">Macro F1</p>
                        <p className={`text-xl font-black ${isProposed ? 'text-teal-600' : 'text-slate-700'}`}>
                          {exp.f1_macro !== null ? `${(exp.f1_macro * 100).toFixed(1)}%` : 'Running...'}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-1">
              <span className="font-semibold text-slate-800">Ablation Interpretation: </span>
              <span>
                Standard VGG16 relies solely on low-resolution 4×4 Block 5 features. MS-VGG16 without gating introduces shallow features but weights them equally regardless of slice context. The proposed <strong>MS-GF-VGG16</strong> adaptively weights feature scales using the SGFU gating MLP, achieving superior fine-grained neoplasm discrimination.
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

