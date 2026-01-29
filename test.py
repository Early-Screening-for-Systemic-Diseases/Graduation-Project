import requests
import json
from pathlib import Path

API_URL = "http://localhost:5000"

def test_api():
    """Test the API"""
    print("\n" + "="*60)
    print("  ANEMIA DETECTION API - TEST CLIENT")
    print("="*60 + "\n")
    
    # Test health check
    print("1️⃣  Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"✓ Health Check: {response.json()}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return
    
    # Test prediction with sample image
    print("2️⃣  Testing /predict endpoint...")
    print("   Looking for test image in data/output_train/...")
    
    # Find a sample image
    data_path = Path("data/output_train")
    if data_path.exists():
        # Try to find any image
        image_files = list(data_path.glob("*/*.png"))
        if image_files:
            test_image = image_files[0]
            print(f"   Using: {test_image}")
            
            with open(test_image, "rb") as f:
                files = {"file": f}
                try:
                    response = requests.post(f"{API_URL}/predict", files=files)
                    result = response.json()
                    
                    print("\n✓ Prediction Result:")
                    print("="*60)
                    print(json.dumps(result, indent=2))
                    print("="*60 + "\n")
                except Exception as e:
                    print(f"✗ Error: {e}\n")
        else:
            print("   No images found in data folder")
    else:
        print("   data/ folder not found")

if __name__ == "__main__":
    test_api()
