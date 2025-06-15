# Django Integration for Gadget Classifier

This guide shows how to integrate the FastAI gadget classifier into your Django application.

## Files Overview

- `inference.py` - Main inference script with GadgetClassifier class
- `django_integration_example.py` - Example Django views and integration code
- `fastai.ipynb` - Original training notebook

## Setup Instructions

### 1. Install Dependencies

```bash
pip install fastai torch torchvision pillow django
```

### 2. Model File Location

Make sure your trained model file is available at:
```
your_django_app/models/gadget_classifier_model.pkl
```

Or update the model path in `inference.py` if needed.

### 3. Django App Integration

#### Step 1: Add files to your Django app

```
your_django_app/
├── __init__.py
├── models.py
├── views.py          # Copy content from django_integration_example.py
├── urls.py
├── inference.py      # Copy the inference.py file
└── models/
    └── gadget_classifier_model.pkl  # Your trained model
```

#### Step 2: Update your app's urls.py

```python
from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_gadget, name='predict_gadget'),
    path('model-info/', views.model_info, name='model_info'),
    path('predict-batch/', views.predict_batch, name='predict_batch'),
]
```

#### Step 3: Include in main URLs

In your project's main `urls.py`:

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('classifier/', include('your_app.urls')),  # Replace 'your_app' with actual app name
]
```

## API Endpoints

### 1. Single Image Prediction

**POST** `/classifier/predict/`

Upload a single image file:

```bash
curl -X POST \
  -F "image=@/path/to/your/image.jpg" \
  http://localhost:8000/classifier/predict/
```

**Response:**
```json
{
  "predicted_class": "smartphone",
  "confidence": 0.95,
  "all_predictions": [
    {
      "class": "smartphone",
      "confidence": 0.95,
      "rank": 1
    },
    {
      "class": "tablet",
      "confidence": 0.03,
      "rank": 2
    },
    // ... other classes
  ],
  "success": true,
  "error": null,
  "filename": "image.jpg",
  "file_size": 1024,
  "content_type": "image/jpeg"
}
```

### 2. Model Information

**GET** `/classifier/model-info/`

```bash
curl http://localhost:8000/classifier/model-info/
```

**Response:**
```json
{
  "success": true,
  "model_info": {
    "loaded": true,
    "classes": ["smartphone", "tablet", "smartwatch", "headphones", "camera"],
    "num_classes": 5,
    "model_type": "FastAI Vision Learner",
    "architecture": "ResNet18"
  }
}
```

### 3. Batch Prediction

**POST** `/classifier/predict-batch/`

Upload multiple images:

```bash
curl -X POST \
  -F "images=@/path/to/image1.jpg" \
  -F "images=@/path/to/image2.jpg" \
  http://localhost:8000/classifier/predict-batch/
```

## Usage Examples

### In Django Views

```python
from .inference import create_classifier

def my_view(request):
    # Create classifier instance
    classifier = create_classifier()
    
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        
        # Make prediction
        result = classifier.predict(image_file.read())
        
        if result['success']:
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            # Use the prediction...
        else:
            error = result['error']
            # Handle error...
```

### With Django REST Framework

```python
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .inference import create_classifier

@api_view(['POST'])
def classify_gadget(request):
    if 'image' not in request.FILES:
        return Response({'error': 'No image provided'}, status=400)
    
    classifier = create_classifier()
    image_file = request.FILES['image']
    
    result = classifier.predict(image_file.read())
    return Response(result)
```

## Frontend Integration

### JavaScript (Fetch API)

```javascript
function classifyImage(imageFile) {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    fetch('/classifier/predict/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Prediction:', data.predicted_class);
            console.log('Confidence:', data.confidence);
        } else {
            console.error('Error:', data.error);
        }
    })
    .catch(error => console.error('Error:', error));
}
```

### React Component Example

```jsx
import React, { useState } from 'react';

function GadgetClassifier() {
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        setLoading(true);
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/classifier/predict/', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            setResult(data);
        } catch (error) {
            console.error('Error:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <input type="file" accept="image/*" onChange={handleFileUpload} />
            {loading && <p>Classifying...</p>}
            {result && result.success && (
                <div>
                    <h3>Prediction: {result.predicted_class}</h3>
                    <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
                </div>
            )}
        </div>
    );
}
```

## Performance Considerations

1. **Model Loading**: The model is loaded once when the Django app starts (singleton pattern)
2. **Memory Usage**: Keep only one model instance in memory
3. **Async Processing**: For high-traffic applications, consider using Celery for background processing
4. **Caching**: Cache predictions for identical images if needed

## Error Handling

The inference script includes comprehensive error handling:

- Invalid file types
- File size limits
- Model loading errors
- Preprocessing errors
- Prediction errors

## Security Considerations

1. **File Validation**: Always validate uploaded files
2. **Size Limits**: Implement file size limits
3. **CSRF Protection**: Use CSRF tokens for forms
4. **Rate Limiting**: Implement rate limiting for API endpoints

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model file exists at the specified path
2. **Import errors**: Make sure all dependencies are installed
3. **Memory issues**: The model requires sufficient RAM (typically 1-2GB)
4. **Permission errors**: Ensure Django has read access to the model file

### Debugging

Enable logging to see detailed error messages:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Production Deployment

For production deployment:

1. Use a proper web server (nginx + gunicorn)
2. Set up proper logging
3. Use a CDN for static files
4. Consider using a GPU-enabled server for faster inference
5. Implement proper monitoring and health checks

## Testing

Test the integration:

```bash
# Test model info endpoint
curl http://localhost:8000/classifier/model-info/

# Test prediction with a sample image
curl -X POST -F "image=@sample_image.jpg" http://localhost:8000/classifier/predict/
``` 