import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import LandingPage from './pages/LandingPage';
import PredictionPage from './pages/PredictionPage';
import PerformancePage from './pages/PerformancePage';
import ArchitecturePage from './pages/ArchitecturePage';
import AboutPage from './pages/AboutPage';
import api from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [backendStatus, setBackendStatus] = useState({ healthy: false });
  const [datasetStats, setDatasetStats] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const health = await api.checkHealth();
        setBackendStatus({ healthy: health.status === 'healthy' && health.model_loaded });

        const info = await api.getModelInfo();
        setModelInfo(info);

        const stats = await api.getDatasetStats();
        if (stats.dataset) {
          setDatasetStats(stats.dataset);
        }
      } catch (err) {
        setBackendStatus({ healthy: false });
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col min-h-screen bg-slate-50 selection:bg-medical-100 selection:text-medical-800">
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        backendStatus={backendStatus} 
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'home' && (
          <LandingPage 
            setActiveTab={setActiveTab} 
            datasetStats={datasetStats} 
            modelInfo={modelInfo} 
          />
        )}
        {activeTab === 'predict' && (
          <PredictionPage />
        )}
        {activeTab === 'performance' && (
          <PerformancePage />
        )}
        {activeTab === 'architecture' && (
          <ArchitecturePage />
        )}
        {activeTab === 'about' && (
          <AboutPage />
        )}
      </main>

      <Footer />
    </div>
  );
}

