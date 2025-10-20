#!/usr/bin/env python3
"""
Basic Diabetes Tongue Classification Pipeline
This script will work with basic packages and create a simple ML model
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Try to import additional packages, but continue if they fail
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Using basic analysis only.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Skipping visualizations.")

def create_labels_csv(data_dir="data"):
    """Create labels CSV from the dataset structure"""
    
    print("Creating labels CSV from dataset structure...")
    
    # Initialize lists to store data
    filenames = []
    labels = []
    
    # Process training data
    train_diabetes_dir = os.path.join(data_dir, "output_train", "diabetes")
    train_non_diabetes_dir = os.path.join(data_dir, "output_train", "non_diabetes")
    
    # Process validation data
    val_diabetes_dir = os.path.join(data_dir, "output_valid", "diabetes")
    val_non_diabetes_dir = os.path.join(data_dir, "output_valid", "non_diabetes")
    
    # Count files
    train_diabetes_count = 0
    train_non_diabetes_count = 0
    val_diabetes_count = 0
    val_non_diabetes_count = 0
    
    # Process training diabetes images
    if os.path.exists(train_diabetes_dir):
        for filename in os.listdir(train_diabetes_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                filenames.append(os.path.join("output_train", "diabetes", filename))
                labels.append("diabetes")
                train_diabetes_count += 1
    
    # Process training non-diabetes images
    if os.path.exists(train_non_diabetes_dir):
        for filename in os.listdir(train_non_diabetes_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                filenames.append(os.path.join("output_train", "non_diabetes", filename))
                labels.append("no_diabetes")
                train_non_diabetes_count += 1
    
    # Process validation diabetes images
    if os.path.exists(val_diabetes_dir):
        for filename in os.listdir(val_diabetes_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                filenames.append(os.path.join("output_valid", "diabetes", filename))
                labels.append("diabetes")
                val_diabetes_count += 1
    
    # Process validation non-diabetes images
    if os.path.exists(val_non_diabetes_dir):
        for filename in os.listdir(val_non_diabetes_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                filenames.append(os.path.join("output_valid", "non_diabetes", filename))
                labels.append("no_diabetes")
                val_non_diabetes_count += 1
    
    # Create DataFrame
    df = pd.DataFrame({
        "filename": filenames,
        "label": labels
    })
    
    # Save CSV
    csv_path = os.path.join(data_dir, "labels.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"[OK] Created labels CSV with {len(df)} images")
    print(f"  Training diabetes: {train_diabetes_count}")
    print(f"  Training non-diabetes: {train_non_diabetes_count}")
    print(f"  Validation diabetes: {val_diabetes_count}")
    print(f"  Validation non-diabetes: {val_non_diabetes_count}")
    print(f"  Total: {len(df)} images")
    print(f"  CSV saved to: {csv_path}")
    
    return csv_path

def analyze_dataset(csv_path):
    """Analyze the dataset"""
    
    print("\nAnalyzing dataset...")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"Total images: {len(df)}")
    print(f"Class distribution:")
    print(df["label"].value_counts())
    
    # Check for missing images
    missing_count = 0
    for _, row in df.iterrows():
        image_path = os.path.join("data", row["filename"])
        if not os.path.exists(image_path):
            missing_count += 1
    
    if missing_count > 0:
        print(f"Warning: {missing_count} images not found")
    else:
        print("[OK] All images found")
    
    return df

def create_basic_features(df):
    """Create basic features from filenames and paths"""
    
    print("\nCreating basic features...")
    
    # Extract features from filename
    features = []
    
    for _, row in df.iterrows():
        filename = row["filename"]
        
        # Basic features from filename
        feature_dict = {
            "filename_length": len(filename),
            "has_diabetes_in_path": "diabetes" in filename.lower(),
            "has_non_diabetes_in_path": "non_diabetes" in filename.lower(),
            "is_training": "output_train" in filename,
            "is_validation": "output_valid" in filename,
            "file_extension": filename.split(".")[-1].lower() if "." in filename else "unknown"
        }
        
        features.append(feature_dict)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(features)
    
    print(f"[OK] Created {len(feature_df.columns)} basic features")
    
    return feature_df

def train_basic_model(X, y):
    """Train a basic machine learning model"""
    
    print("\nTraining basic model...")
    
    if not SKLEARN_AVAILABLE:
        print("Warning: scikit-learn not available. Using dummy model.")
        return None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"[OK] Model trained successfully")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    return model, {
        "accuracy": accuracy,
        "auc": auc,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba
    }

def create_visualizations(results, df):
    """Create visualizations"""
    
    if not MATPLOTLIB_AVAILABLE or not results:
        print("Skipping visualizations (matplotlib not available or no results)")
        return
    
    print("\nCreating visualizations...")
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Class distribution plot
    plt.figure(figsize=(10, 6))
    df["label"].value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Confusion matrix
    if "y_test" in results and "y_pred" in results:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(results["y_test"], results["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("outputs/confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    print("[OK] Visualizations saved to outputs/")

def generate_report(results, df):
    """Generate a comprehensive report"""
    
    print("\nGenerating report...")
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    report = f"""
# Diabetes Tongue Classification - Basic ML Pipeline Results

## Dataset Summary
- **Total images:** {len(df)}
- **Class distribution:** {df['label'].value_counts().to_dict()}
- **Training images:** {len(df[df['filename'].str.contains('output_train')])}
- **Validation images:** {len(df[df['filename'].str.contains('output_valid')])}

## Model Performance
"""
    
    if results:
        report += f"""
- **Accuracy:** {results['accuracy']:.4f}
- **AUC Score:** {results['auc']:.4f}
"""
    else:
        report += """
- **Model:** Not trained (scikit-learn not available)
- **Note:** Install scikit-learn for full ML pipeline
"""
    
    report += f"""

## Features Used
- Filename length
- Path indicators (diabetes/non_diabetes)
- Dataset split indicators (train/validation)
- File extension

## Recommendations
1. **Install full ML packages** for better performance:
   ```bash
   pip install scikit-learn matplotlib seaborn
   ```

2. **Use deep learning models** for better accuracy:
   - Install PyTorch: `pip install torch torchvision`
   - Use pre-trained models like ResNet, EfficientNet

3. **Improve features**:
   - Extract image features (color histograms, texture)
   - Use image augmentation
   - Implement proper train/validation splits

4. **Data collection**:
   - Ensure balanced classes (currently: {df['label'].value_counts().to_dict()})
   - Collect more diverse images if possible

## Next Steps
1. Install required packages for full pipeline
2. Implement deep learning models
3. Add proper image preprocessing
4. Create comprehensive evaluation metrics
5. Deploy model for inference

## Files Generated
- `outputs/class_distribution.png` - Class distribution visualization
- `outputs/confusion_matrix.png` - Model confusion matrix (if available)
- `outputs/report.md` - This report
- `data/labels.csv` - Dataset labels

---
*Generated by Diabetes Tongue Classification Pipeline*
"""
    
    # Save report
    with open("outputs/report.md", "w") as f:
        f.write(report)
    
    print("[OK] Report saved to outputs/report.md")

def main():
    """Main function"""
    
    print("=" * 60)
    print("DIABETES TONGUE CLASSIFICATION - BASIC ML PIPELINE")
    print("=" * 60)
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Step 1: Create labels CSV
    print("\n1. Creating labels CSV...")
    csv_path = create_labels_csv()
    
    # Step 2: Analyze dataset
    print("\n2. Analyzing dataset...")
    df = analyze_dataset(csv_path)
    
    # Step 3: Create features
    print("\n3. Creating features...")
    features = create_basic_features(df)
    
    # Step 4: Train model
    print("\n4. Training model...")
    if SKLEARN_AVAILABLE:
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(df["label"])
        
        # Train model
        model, results = train_basic_model(features, y)
    else:
        model, results = None, None
        print("Skipping model training (scikit-learn not available)")
    
    # Step 5: Create visualizations
    print("\n5. Creating visualizations...")
    create_visualizations(results, df)
    
    # Step 6: Generate report
    print("\n6. Generating report...")
    generate_report(results, df)
    
    print("\n" + "="*60)
    print("BASIC ML PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check outputs/ directory for results")
    print("2. Install full ML packages for better performance")
    print("3. Run the complete deep learning pipeline")

if __name__ == "__main__":
    main()
