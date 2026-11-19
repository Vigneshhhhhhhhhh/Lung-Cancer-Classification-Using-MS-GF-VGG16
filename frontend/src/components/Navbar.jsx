import React from 'react';
import { Activity, Cpu, BarChart3, Info, Stethoscope } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, backendStatus }) {
  const navItems = [
    { id: 'home', label: 'Overview', icon: Stethoscope },
    { id: 'predict', label: 'CT Analysis', icon: Activity },
    { id: 'performance', label: 'Evaluation Metrics', icon: BarChart3 },
    { id: 'architecture', label: 'MS-GF-VGG16 Architecture', icon: Cpu },
    { id: 'about', label: 'Research & Methodology', icon: Info },
  ];

  return (
    <header className="sticky top-0 z-50 bg-white/95 backdrop-blur-md border-b border-slate-200 shadow-sm transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Brand */}
          <div 
            onClick={() => setActiveTab('home')}
            className="flex items-center space-x-3 cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-medical-700 to-teal-500 flex items-center justify-center text-white shadow-md shadow-medical-500/20 group-hover:scale-105 transition-transform">
              <Activity className="w-6 h-6 stroke-[2.5]" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-extrabold text-slate-900 tracking-tight text-lg">
                  MS-GF-VGG16
                </span>
                <span className="text-xs px-2 py-0.5 font-semibold bg-medical-50 text-medical-700 rounded-full border border-medical-200">
                  AI CAD
                </span>
              </div>
              <p className="text-xs text-slate-500 font-medium">Lung CT Neoplasm Classifier</p>
            </div>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-medical-50 text-medical-700 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/70'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-medical-600' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Backend Status Indicator */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-xs">
              <span className={`w-2 h-2 rounded-full ${backendStatus?.healthy ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`} />
              <span className="text-slate-600 font-medium">
                {backendStatus?.healthy ? 'Model Online' : 'Connecting API...'}
              </span>
            </div>
            <button
              onClick={() => setActiveTab('predict')}
              className="bg-medical-600 hover:bg-medical-700 text-white px-4 py-2 rounded-lg text-sm font-semibold shadow-sm shadow-medical-600/30 transition-all hover:shadow-md"
            >
              Analyze Scan
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}

