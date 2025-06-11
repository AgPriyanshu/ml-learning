import React from "react";
import { X } from "lucide-react";

interface ImagePreviewProps {
  imageUrl: string;
  fileName: string;
  onRemove: () => void;
}

export default function ImagePreview({
  imageUrl,
  fileName,
  onRemove,
}: ImagePreviewProps) {
  return (
    <div className="relative max-w-md mx-auto">
      <div className="relative bg-white rounded-2xl shadow-lg overflow-hidden">
        <img
          src={imageUrl}
          alt="Uploaded device"
          className="w-full h-64 object-cover"
        />
        <button
          onClick={onRemove}
          className="absolute top-3 right-3 p-2 bg-white rounded-full shadow-md hover:shadow-lg transition-shadow duration-200 hover:bg-gray-50"
        >
          <X className="h-4 w-4 text-gray-600" />
        </button>
      </div>
      <p className="mt-3 text-sm text-gray-600 text-center truncate px-4">
        {fileName}
      </p>
    </div>
  );
}
