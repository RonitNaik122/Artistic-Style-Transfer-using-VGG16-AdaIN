import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileImage, FileVideo, Palette, Loader, Check, AlertTriangle } from 'lucide-react';
import apiService from './services/api'; // Import the API service

export default function App() {
  const [file, setFile] = useState(null);
  const [filePreview, setFilePreview] = useState('');
  const [fileType, setFileType] = useState('');
  const [selectedModel, setSelectedModel] = useState('geometric_painting.pth');
  const [outputPath, setOutputPath] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const models = [
    { id: 'geometric_painting.pth', name: 'Geometric Painting' },
    { id: 'oil_painting.pth', name: 'Oil Painting' },
    { id: 'van_gogh.pth', name: 'Van Gogh' }
  ];

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    
    if (!selectedFile) return;
    
    const fileExtension = selectedFile.name.split('.').pop().toLowerCase();
    
    // Validate file type
    if (fileExtension === 'jpg' || fileExtension === 'jpeg' || fileExtension === 'png') {
      setFileType('image');
    } else if (fileExtension === 'mp4') {
      setFileType('video');
    } else {
      setError('Please upload a JPG/PNG image or MP4 video file.');
      return;
    }
    
    setFile(selectedFile);
    setError('');
    
    // Create preview URL
    const previewUrl = URL.createObjectURL(selectedFile);
    setFilePreview(previewUrl);
  };

  const processFile = async () => {
    if (!file) {
      setError('Please upload a file first.');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      // Call the API service to process the file
      const result = await apiService.processFile(file, selectedModel);
      
      if (result.success) {
        setOutputPath(result.outputUrl);
      } else {
        setError(result.error || 'An error occurred during processing.');
      }
    } catch (err) {
      setError('An error occurred during processing. Please try again.');
      console.error('Processing error:', err);
    } finally {
      setLoading(false);
    }
  };

  const clearFile = () => {
    setFile(null);
    setFilePreview('');
    setFileType('');
    setOutputPath('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Clean up URLs on unmount
  useEffect(() => {
    return () => {
      if (filePreview) {
        URL.revokeObjectURL(filePreview);
      }
    };
  }, [filePreview]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-pink-500 to-red-500">
            Neural Style Transfer
          </h1>
          <p className="text-gray-300 mt-2">
            Transform your images and videos with AI-powered style transfer
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-gray-800 rounded-xl p-6 shadow-xl"
          >
            <h2 className="text-xl font-semibold mb-4">Upload Media</h2>
            
            <div className="mb-6">
              <div 
                onClick={() => fileInputRef.current.click()}
                className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center cursor-pointer hover:border-purple-500 transition-colors"
              >
                {filePreview ? (
                  <div className="relative">
                    {fileType === 'image' ? (
                      <img 
                        src={filePreview} 
                        alt="Preview" 
                        className="max-h-48 mx-auto rounded" 
                      />
                    ) : (
                      <video 
                        src={filePreview} 
                        controls 
                        className="max-h-48 mx-auto rounded"
                      />
                    )}
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        clearFile();
                      }}
                      className="absolute top-2 right-2 bg-red-500 text-white p-1 rounded-full"
                    >
                      ✕
                    </button>
                  </div>
                ) : (
                  <div className="flex flex-col items-center">
                    <Upload className="w-12 h-12 text-gray-400 mb-2" />
                    <p className="text-gray-400">Click to upload an image (JPG) or video (MP4)</p>
                  </div>
                )}
                <input 
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept=".jpg,.jpeg,.png,.mp4"
                  className="hidden"
                />
              </div>
              
              {error && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-4 p-3 bg-red-900/50 border border-red-700 rounded-lg flex items-center"
                >
                  <AlertTriangle className="w-5 h-5 text-red-400 mr-2" />
                  <p className="text-red-200 text-sm">{error}</p>
                </motion.div>
              )}
            </div>
            
            <div className="mb-6">
              <h3 className="text-gray-300 mb-2 flex items-center">
                <Palette className="w-4 h-4 mr-2" />
                Select Style
              </h3>
              <div className="grid grid-cols-3 gap-2">
                {models.map((model) => (
                  <motion.button
                    key={model.id}
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setSelectedModel(model.id)}
                    className={`p-3 rounded-lg text-sm ${
                      selectedModel === model.id
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    {model.name}
                  </motion.button>
                ))}
              </div>
            </div>
            
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              disabled={!file || loading}
              onClick={processFile}
              className={`w-full py-3 rounded-lg font-medium flex items-center justify-center
                ${!file || loading ? 'bg-gray-600 cursor-not-allowed' : 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600'}`}
            >
              {loading ? (
                <>
                  <Loader className="w-5 h-5 mr-2 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  {fileType === 'image' ? <FileImage className="w-5 h-5 mr-2" /> : <FileVideo className="w-5 h-5 mr-2" />}
                  Apply Style Transfer
                </>
              )}
            </motion.button>
          </motion.div>
          
          {/* Output Section */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-gray-800 rounded-xl p-6 shadow-xl"
          >
            <h2 className="text-xl font-semibold mb-4">Stylized Output</h2>
            
            <div className="border-2 border-gray-700 rounded-lg p-4 flex items-center justify-center min-h-64">
              <AnimatePresence mode="wait">
                {outputPath ? (
                  <motion.div
                    key="output"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="w-full"
                  >
                    {fileType === 'image' ? (
                      <div className="relative">
                        <img src={outputPath} alt="Stylized output" className="max-h-64 mx-auto rounded" />
                        <div className="absolute top-2 right-2 bg-green-500 text-white p-1 rounded-full">
                          <Check className="w-4 h-4" />
                        </div>
                      </div>
                    ) : (
                      <div className="relative">
                        <video 
                          src={outputPath} 
                          controls 
                          className="max-h-64 mx-auto rounded"
                        />
                        <div className="absolute top-2 right-2 bg-green-500 text-white p-1 rounded-full">
                          <Check className="w-4 h-4" />
                        </div>
                      </div>
                    )}
                    
                    <div className="mt-4 text-center">
                      <motion.a
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        href={outputPath}
                        download
                        className="inline-block px-4 py-2 bg-purple-600 rounded-lg text-sm font-medium hover:bg-purple-700"
                      >
                        Download Stylized {fileType === 'image' ? 'Image' : 'Video'}
                      </motion.a>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="placeholder"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="text-center text-gray-400"
                  >
                    <div className="p-8">
                      <div className="mb-4 mx-auto w-24 h-24 border-2 border-dashed border-gray-600 rounded-lg flex items-center justify-center">
                        {fileType === 'image' ? (
                          <FileImage className="w-12 h-12 text-gray-600" />
                        ) : fileType === 'video' ? (
                          <FileVideo className="w-12 h-12 text-gray-600" />
                        ) : (
                          <Palette className="w-12 h-12 text-gray-600" />
                        )}
                      </div>
                      <p>Your stylized output will appear here</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
            
            {outputPath && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4 p-3 bg-green-900/20 border border-green-700/50 rounded-lg"
              >
                <h3 className="text-green-400 text-sm font-medium mb-1">Processing Complete!</h3>
                <p className="text-gray-300 text-xs">
                  Applied <span className="font-medium">{models.find(m => m.id === selectedModel).name}</span> style 
                  to your {fileType}
                </p>
              </motion.div>
            )}
          </motion.div>
        </div>
        
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0, transition: { delay: 0.3 } }}
          className="mt-8 text-center text-gray-400 text-sm"
        >
          <p>Powered by PyTorch Neural Style Transfer</p>
        </motion.div>
      </div>
    </div>
  );
}