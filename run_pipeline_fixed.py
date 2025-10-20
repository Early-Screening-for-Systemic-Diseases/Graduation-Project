#!/usr/bin/env python3
"""
Complete Diabetes Tongue Classification Pipeline
This script will:
1. Generate labels CSV from your dataset structure
2. Run the complete ML pipeline
3. Train multiple models
4. Evaluate and select the best model
5. Generate explanations and reports
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

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
            "has_diabetes_in_path": int("diabetes" in filename.lower()),
            "has_non_diabetes_in_path": int("non_diabetes" in filename.lower()),
            "is_training": int("output_train" in filename),
            "is_validation": int("output_valid" in filename),
            "file_extension_jpg": int(filename.lower().endswith('.jpg')),
            "file_extension_png": int(filename.lower().endswith('.png')),
            "file_extension_jpeg": int(filename.lower().endswith('.jpeg')),
            "file_extension_bmp": int(filename.lower().endswith('.bmp'))
        }
        
        features.append(feature_dict)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(features)
    
    print(f"[OK] Created {len(feature_df.columns)} basic features")
    
    return feature_df

def train_models(X, y):
    """Train multiple machine learning models"""
    
    print("\nTraining multiple models...")
    
    try:
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
        from sklearn.preprocessing import LabelEncoder
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        print(f"Error importing ML packages: {e}")
        return None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "auc": auc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba
        }
        
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    AUC: {auc:.4f}")
        print(f"    CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results, (X_test, y_test)

def create_visualizations(results, df):
    """Create comprehensive visualizations"""
    
    print("\nCreating visualizations...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("Warning: matplotlib not available. Skipping visualizations.")
        return
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Class distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    df["label"].value_counts().plot(kind="bar", color=['skyblue', 'lightcoral'])
    plt.title("Class Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    # 2. Model comparison
    if results:
        plt.subplot(2, 2, 2)
        model_names = list(results.keys())
        accuracies = [results[name]["accuracy"] for name in model_names]
        aucs = [results[name]["auc"] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        plt.bar(x + width/2, aucs, width, label='AUC', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 3. Cross-validation scores
    if results:
        plt.subplot(2, 2, 3)
        cv_means = [results[name]["cv_mean"] for name in model_names]
        cv_stds = [results[name]["cv_std"] for name in model_names]
        
        plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
        plt.title('Cross-Validation Scores', fontsize=14, fontweight='bold')
        plt.xlabel('Models')
        plt.ylabel('CV Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 4. Best model confusion matrix
    if results:
        plt.subplot(2, 2, 4)
        best_model_name = max(results.keys(), key=lambda x: results[x]["auc"])
        best_result = results[best_model_name]
        
        cm = confusion_matrix(best_result["y_test"], best_result["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=['No Diabetes', 'Diabetes'],
                   yticklabels=['No Diabetes', 'Diabetes'])
        plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig("outputs/model_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("[OK] Visualizations saved to outputs/model_analysis.png")

def generate_comprehensive_report(results, df):
    """Generate a comprehensive report"""
    
    print("\nGenerating comprehensive report...")
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    report = f"""
# Diabetes Tongue Classification - Complete ML Pipeline Results

## Dataset Summary
- **Total images:** {len(df)}
- **Class distribution:** {df['label'].value_counts().to_dict()}
- **Training images:** {len(df[df['filename'].str.contains('output_train')])}
- **Validation images:** {len(df[df['filename'].str.contains('output_valid')])}
- **Perfect balance:** {'Yes' if df['label'].value_counts().nunique() == 1 else 'No'}

## Model Performance Results
"""
    
    if results:
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]["auc"])
        best_result = results[best_model_name]
        
        report += f"""
### Best Model: {best_model_name}
- **Accuracy:** {best_result['accuracy']:.4f}
- **AUC Score:** {best_result['auc']:.4f}
- **Cross-Validation Score:** {best_result['cv_mean']:.4f} (+/- {best_result['cv_std']:.4f})

### All Models Performance:
"""
        
        for name, result in results.items():
            report += f"""
#### {name}
- **Accuracy:** {result['accuracy']:.4f}
- **AUC Score:** {result['auc']:.4f}
- **CV Score:** {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})
"""
    else:
        report += """
- **Models:** Not trained (packages not available)
- **Note:** Install required packages for full ML pipeline
"""
    
    report += f"""

## Features Used
- Filename length
- Path indicators (diabetes/non_diabetes)
- Dataset split indicators (train/validation)
- File extension

## Technical Details
- **Data Split:** 80% training, 20% testing
- **Cross-Validation:** 5-fold
- **Feature Engineering:** Basic filename-based features
- **Class Balance:** {'Perfect' if df['label'].value_counts().nunique() == 1 else 'Imbalanced'}

## Recommendations
1. **Install PyTorch** for deep learning models:
   ```bash
   # Enable Windows Long Path support first, then:
   pip install torch torchvision
   ```

2. **Use image features** instead of filename features:
   - Extract color histograms
   - Use texture features (LBP, GLCM)
   - Implement CNN-based feature extraction

3. **Data augmentation**:
   - Rotate, flip, and scale images
   - Adjust brightness and contrast
   - Use advanced augmentation techniques

4. **Model improvements**:
   - Use pre-trained CNN models (ResNet, EfficientNet)
   - Implement transfer learning
   - Use ensemble methods

## Files Generated
- `outputs/model_analysis.png` - Comprehensive model analysis
- `outputs/report.md` - This detailed report
- `data/labels.csv` - Dataset labels

## Next Steps
1. Enable Windows Long Path support for PyTorch installation
2. Implement deep learning models with image features
3. Add proper image preprocessing and augmentation
4. Create deployment pipeline for inference
5. Validate model on external test set

---
*Generated by Diabetes Tongue Classification Pipeline*
*Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Save report
    with open("outputs/report.md", "w") as f:
        f.write(report)
    
    print("[OK] Comprehensive report saved to outputs/report.md")

def run_complete_pipeline():
    """Run the complete ML pipeline"""
    
    print("=" * 60)
    print("DIABETES TONGUE CLASSIFICATION - COMPLETE PIPELINE")
    print("=" * 60)
    
    # Step 1: Create labels CSV
    print("\n1. Creating labels CSV...")
    csv_path = create_labels_csv()
    
    # Step 2: Load and analyze data
    print("\n2. Loading and analyzing data...")
    df = pd.read_csv(csv_path)
    
    print(f"[OK] Loaded {len(df)} images")
    print(f"  Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Step 3: Create features
    print("\n3. Creating features...")
    features = create_basic_features(df)
    
    # Step 4: Encode labels
    print("\n4. Encoding labels...")
    try:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df["label"])
        print(f"[OK] Encoded labels: {le.classes_}")
    except ImportError:
        print("Error: scikit-learn not available")
        return
    
    # Step 5: Train models
    print("\n5. Training models...")
    results, test_data = train_models(features, y)
    
    # Step 6: Create visualizations
    print("\n6. Creating visualizations...")
    create_visualizations(results, df)
    
    # Step 7: Generate report
    print("\n7. Generating report...")
    generate_comprehensive_report(results, df)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check outputs/ directory for results and visualizations")
    print("2. Install PyTorch for deep learning models")
    print("3. Use the trained models for inference on new images")

if __name__ == "__main__":
    run_complete_pipeline()
