import React from 'react';
import { Smartphone, Watch, Tablet, Camera, Laptop, CheckCircle } from 'lucide-react';

interface Prediction {
  device: string;
  confidence: number;
}

interface PredictionResultsProps {
  predictions: Prediction[];
  isLoading: boolean;
}

const deviceIcons = {
  smartphone: Smartphone,
  smartwatch: Watch,
  tablet: Tablet,
  camera: Camera,
  laptop: Laptop,
};

const deviceColors = {
  smartphone: 'text-blue-600 bg-blue-100',
  smartwatch: 'text-purple-600 bg-purple-100',
  tablet: 'text-green-600 bg-green-100',
  camera: 'text-orange-600 bg-orange-100',
  laptop: 'text-indigo-600 bg-indigo-100',
};

export default function PredictionResults({ predictions, isLoading }: PredictionResultsProps) {
  if (isLoading) {
    return (
      <div className="w-full max-w-2xl mx-auto">
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              Analyzing your image...
            </h3>
            <p className="text-gray-600">
              Our AI is identifying the device in your image
            </p>
          </div>
        </div>
      </div>
    );
  }

  const topPrediction = predictions[0];
  const Icon = deviceIcons[topPrediction.device as keyof typeof deviceIcons];
  const colorClass = deviceColors[topPrediction.device as keyof typeof deviceColors];

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6">
      {/* Top Prediction */}
      <div className="bg-white rounded-2xl shadow-lg p-8">
        <div className="text-center mb-6">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <CheckCircle className="h-6 w-6 text-green-500" />
            <h3 className="text-xl font-semibold text-gray-900">
              Classification Complete
            </h3>
          </div>
          <div className={`inline-flex items-center space-x-3 p-4 rounded-2xl ${colorClass.split(' ')[1]}`}>
            <Icon className={`h-8 w-8 ${colorClass.split(' ')[0]}`} />
            <div>
              <h4 className="text-2xl font-bold text-gray-900 capitalize">
                {topPrediction.device}
              </h4>
              <p className={`text-sm font-medium ${colorClass.split(' ')[0]}`}>
                {(topPrediction.confidence * 100).toFixed(1)}% confidence
              </p>
            </div>
          </div>
        </div>

        {/* Confidence Bar */}
        <div className="mb-6">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>Confidence Level</span>
            <span>{(topPrediction.confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-1000 ease-out"
              style={{ width: `${topPrediction.confidence * 100}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* All Predictions */}
      <div className="bg-white rounded-2xl shadow-lg p-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-4">
          All Predictions
        </h4>
        <div className="space-y-3">
          {predictions.map((prediction, index) => {
            const PredIcon = deviceIcons[prediction.device as keyof typeof deviceIcons];
            const predColorClass = deviceColors[prediction.device as keyof typeof deviceColors];
            
            return (
              <div key={index} className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 transition-colors duration-200">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg ${predColorClass.split(' ')[1]}`}>
                    <PredIcon className={`h-5 w-5 ${predColorClass.split(' ')[0]}`} />
                  </div>
                  <span className="font-medium text-gray-900 capitalize">
                    {prediction.device}
                  </span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${predColorClass.split(' ')[0].replace('text-', 'bg-')}`}
                      style={{ width: `${prediction.confidence * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-gray-600 w-12 text-right">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}