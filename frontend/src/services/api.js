import axios from 'axios';

// Base API configuration: uses Vite's reverse proxy (/api) so that tunnels and remote devices work seamlessly
const API_BASE = import.meta.env.VITE_API_BASE || '/api';

const client = axios.create({
  baseURL: API_BASE,
  timeout: 45000,
});

export const api = {
  checkHealth: async () => {
    const res = await client.get('/health');
    return res.data;
  },

  getModelInfo: async () => {
    const res = await client.get('/model-info');
    return res.data;
  },

  getDatasetStats: async () => {
    const res = await client.get('/dataset-stats');
    return res.data;
  },

  getSampleImages: async () => {
    const res = await client.get('/samples');
    return res.data;
  },

  getEvaluationMetrics: async () => {
    const res = await client.get('/evaluation');
    return res.data;
  },

  getAblationStudy: async () => {
    const res = await client.get('/ablation');
    return res.data;
  },

  predictImage: async (file, gradcam = true, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);

    const res = await client.post(`/predict?gradcam=${gradcam}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percent);
        }
      },
    });
    return res.data;
  },
};

export default api;
