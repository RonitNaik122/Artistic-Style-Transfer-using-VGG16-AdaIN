import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'https://d2b7-2409-4080-d14-4036-bd30-44eb-d51-488a.ngrok-free.app';

const apiService = {
  async processFile(file, model) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('model', model);
      
      const response = await axios.post(`${API_URL}/api/process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      return {
        success: true,
        data: response.data,
        outputUrl: `${API_URL}${response.data.output_url}`
      };
    } catch (error) {
      console.error('Error processing file:', error);
      return {
        success: false,
        error: error.response?.data?.error || 'An unexpected error occurred'
      };
    }
  }
};

export default apiService;