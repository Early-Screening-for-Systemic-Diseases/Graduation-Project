from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from io import BytesIO
import os
from flask_cors import CORS

CORS(app)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "api": "Anemia Detection",
        "endpoints": {
            "POST /predict": "Upload image and get anemia prediction",
            "GET /health": "Check API status"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict anemia percentage from eye image"""
    try:
        # Get file
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read and process image
        img = Image.open(BytesIO(file.read())).convert('RGB')
        
        # Resize to standard size
        img_resized = img.resize((224, 224))
        
        # Calculate image statistics
        img_array = np.array(img_resized)
        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)
        
        # Convert to HSV for better color analysis
        # Using PIL HSV conversion
        img_hsv = img_resized.convert('HSV')
        hsv_array = np.array(img_hsv)
        saturation = np.mean(hsv_array[:,:,1])
        value = np.mean(hsv_array[:,:,2])
        
        # Calculate anemia percentage based on image properties
        # Lower saturation and value = more pale/anemic appearance
        normalized_saturation = saturation / 255.0
        normalized_value = value / 255.0
        
        # Combine metrics to get anemia percentage
        anemia_percentage = ((1.0 - normalized_saturation) * 50 + 
                           (1.0 - normalized_value) * 50)
        anemia_percentage = min(100, max(0, anemia_percentage))
        
        # Calculate confidence based on image quality
        # Higher variance = clearer image = higher confidence
        confidence = min(95, max(45, (std_intensity / 85.0) * 100))
        
        # Interpretation
        interpretation = "Anemic (requires medical attention)" if anemia_percentage > 50 else "Non-anemic"
        
        return jsonify({
            "status": "success",
            "anemia_percentage": round(float(anemia_percentage), 2),
            "confidence": round(float(confidence), 2),
            "interpretation": interpretation,
            "image_analysis": {
                "mean_intensity": round(float(mean_intensity), 2),
                "std_deviation": round(float(std_intensity), 2),
                "saturation": round(float(saturation), 2),
                "brightness": round(float(value), 2)
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Anemia Detection API",
        "version": "1.0.0"
    }), 200

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  ANEMIA DETECTION API")
    print("="*60)
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Starting server on port {port}")
    print("📤 POST /predict - Upload eye image")
    print("🏥 GET /health - API status")
    print("="*60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
