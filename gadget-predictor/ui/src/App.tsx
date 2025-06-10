import React, { useState } from "react";
import { Brain, Zap, Shield, Cpu } from "lucide-react";
import DropZone from "./components/DropZone";
import ImagePreview from "./components/ImagePreview";
import PredictionResults from "./components/PredictionResults";

interface Prediction {
  device: string;
  confidence: number;
}

function App() {
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const generateMockPredictions = (): Prediction[] => {
    const devices = ["smartphone", "smartwatch", "tablet", "camera", "laptop"];
    const randomIndex = Math.floor(Math.random() * devices.length);
    const topDevice = devices[randomIndex];

    // Generate realistic confidence scores
    const predictions: Prediction[] = devices.map((device, index) => {
      let confidence: number;
      if (device === topDevice) {
        confidence = 0.75 + Math.random() * 0.24; // 75-99%
      } else {
        confidence = Math.random() * 0.25; // 0-25%
      }
      return { device, confidence };
    });

    // Sort by confidence (highest first)
    return predictions.sort((a, b) => b.confidence - a.confidence);
  };

  const handleImageUpload = async (file: File) => {
    setIsUploading(true);
    setUploadedImage(file);

    // Create preview URL
    const url = URL.createObjectURL(file);
    setImageUrl(url);

    // Simulate upload delay
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setIsUploading(false);

    // Start analysis
    setIsAnalyzing(true);

    // Simulate ML model prediction delay
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const mockPredictions = generateMockPredictions();
    setPredictions(mockPredictions);
    setIsAnalyzing(false);
  };

  const handleRemoveImage = () => {
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
    }
    setUploadedImage(null);
    setImageUrl("");
    setPredictions([]);
  };

  const handleStartOver = () => {
    handleRemoveImage();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="py-8 px-4">
        <div className="max-w-6xl mx-auto text-center">
          <div className="flex items-center justify-center space-x-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl">
              <Brain className="h-8 w-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Device Classifier
            </h1>
          </div>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Upload an image and let our AI identify whether it's a smartphone,
            smartwatch, tablet, camera, or laptop
          </p>
        </div>
      </header>

      {/* Features */}
      {!uploadedImage && (
        <section className="py-12 px-4">
          <div className="max-w-4xl mx-auto">
            <div className="grid md:grid-cols-3 gap-8 mb-12">
              <div className="text-center p-6">
                <div className="p-3 bg-blue-100 rounded-2xl w-fit mx-auto mb-4">
                  <Zap className="h-8 w-8 text-blue-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Lightning Fast
                </h3>
                <p className="text-gray-600">
                  Get results in seconds with our optimized AI model
                </p>
              </div>
              <div className="text-center p-6">
                <div className="p-3 bg-purple-100 rounded-2xl w-fit mx-auto mb-4">
                  <Shield className="h-8 w-8 text-purple-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Secure & Private
                </h3>
                <p className="text-gray-600">
                  Your images are processed securely and never stored
                </p>
              </div>
              <div className="text-center p-6">
                <div className="p-3 bg-green-100 rounded-2xl w-fit mx-auto mb-4">
                  <Cpu className="h-8 w-8 text-green-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  High Accuracy
                </h3>
                <p className="text-gray-600">
                  Trained on thousands of device images for precision
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Main Content */}
      <main className="py-8 px-4">
        <div className="max-w-6xl mx-auto">
          {!uploadedImage ? (
            <DropZone
              onImageUpload={handleImageUpload}
              isUploading={isUploading}
            />
          ) : (
            <div className="space-y-8">
              {/* Image Preview */}
              <div className="text-center">
                <ImagePreview
                  imageUrl={imageUrl}
                  fileName={uploadedImage.name}
                  onRemove={handleRemoveImage}
                />
              </div>

              {/* Results */}
              {(isAnalyzing || predictions.length > 0) && (
                <PredictionResults
                  predictions={predictions}
                  isLoading={isAnalyzing}
                />
              )}

              {/* Action Buttons */}
              {predictions.length > 0 && (
                <div className="text-center">
                  <button
                    onClick={handleStartOver}
                    className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
                  >
                    Classify Another Image
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="py-8 px-4 text-center text-gray-500 text-sm">
        <p>
          Powered by advanced machine learning • Built with React & TypeScript
        </p>
      </footer>
    </div>
  );
}

export default App;
