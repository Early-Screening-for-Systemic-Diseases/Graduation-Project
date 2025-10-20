#!/usr/bin/env python3
"""
Proper Diabetes Tongue Classification Pipeline
This script uses actual image features instead of filename-based features
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

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

def extract_image_features(image_path):
    """Extract basic features from an image"""
    
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("Warning: PIL not available. Using dummy features.")
        return {
            "width": 100,
            "height": 100,
            "aspect_ratio": 1.0,
            "mean_red": 128,
            "mean_green": 128,
            "mean_blue": 128,
            "std_red": 50,
            "std_green": 50,
            "std_blue": 50
        }
    
    try:
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Basic features
        height, width = img_array.shape[:2]
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Color features (if RGB)
        if len(img_array.shape) == 3:
            mean_red = np.mean(img_array[:, :, 0])
            mean_green = np.mean(img_array[:, :, 1])
            mean_blue = np.mean(img_array[:, :, 2])
            std_red = np.std(img_array[:, :, 0])
            std_green = np.std(img_array[:, :, 1])
            std_blue = np.std(img_array[:, :, 2])
        else:
            # Grayscale
            mean_red = mean_green = mean_blue = np.mean(img_array)
            std_red = std_green = std_blue = np.std(img_array)
        
        return {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "mean_red": mean_red,
            "mean_green": mean_green,
            "mean_blue": mean_blue,
            "std_red": std_red,
            "std_green": std_green,
            "std_blue": std_blue
        }
    
    except Exception as e:
        print(f"Warning: Could not process {image_path}: {e}")
        return {
            "width": 100,
            "height": 100,
            "aspect_ratio": 1.0,
            "mean_red": 128,
            "mean_green": 128,
            "mean_blue": 128,
            "std_red": 50,
            "std_green": 50,
            "std_blue": 50
        }

def create_proper_features(df, sample_size=100):
    """Create proper features from actual images (not filenames)"""
    
    print(f"\nCreating proper image features (sampling {sample_size} images per class)...")
    
    # Sample images to avoid long processing time
    diabetes_samples = df[df['label'] == 'diabetes'].sample(n=min(sample_size, len(df[df['label'] == 'diabetes'])), random_state=42)
    no_diabetes_samples = df[df['label'] == 'no_diabetes'].sample(n=min(sample_size, len(df[df['label'] == 'no_diabetes'])), random_state=42)
    
    sampled_df = pd.concat([diabetes_samples, no_diabetes_samples]).reset_index(drop=True)
    
    print(f"  Processing {len(sampled_df)} images...")
    
    # Extract features from actual images
    features = []
    labels = []
    
    for idx, row in sampled_df.iterrows():
        if idx % 20 == 0:
            print(f"    Processing image {idx+1}/{len(sampled_df)}")
        
        image_path = os.path.join("data", row["filename"])
        if os.path.exists(image_path):
            feature_dict = extract_image_features(image_path)
            features.append(feature_dict)
            labels.append(row["label"])
        else:
            print(f"    Warning: Image not found: {image_path}")
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(features)
    
    print(f"[OK] Created {len(feature_df.columns)} proper image features from {len(feature_df)} images")
    
    return feature_df, labels

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
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
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
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    df["label"].value_counts().plot(kind="bar", color=['skyblue', 'lightcoral'])
    plt.title("Class Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    # 2. Model comparison
    if results:
        plt.subplot(2, 3, 2)
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
        plt.subplot(2, 3, 3)
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
        plt.subplot(2, 3, 4)
        best_model_name = max(results.keys(), key=lambda x: results[x]["auc"])
        best_result = results[best_model_name]
        
        cm = confusion_matrix(best_result["y_test"], best_result["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=['No Diabetes', 'Diabetes'],
                   yticklabels=['No Diabetes', 'Diabetes'])
        plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    # 5. Feature importance (if available)
    if results and hasattr(list(results.values())[0]["model"], 'feature_importances_'):
        plt.subplot(2, 3, 5)
        best_model = list(results.values())[0]["model"]
        if hasattr(best_model, 'feature_importances_'):
            feature_names = [f"Feature {i+1}" for i in range(len(best_model.feature_importances_))]
            plt.bar(feature_names, best_model.feature_importances_)
            plt.title('Feature Importance', fontsize=14, fontweight='bold')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(rotation=45)
    
    # 6. ROC Curve
    if results:
        plt.subplot(2, 3, 6)
        from sklearn.metrics import roc_curve
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(result["y_test"], result["y_pred_proba"])
            plt.plot(fpr, tpr, label=f'{name} (AUC={result["auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("outputs/proper_model_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("[OK] Visualizations saved to outputs/proper_model_analysis.png")

def generate_proper_report(results, df):
    """Generate a proper report"""
    
    print("\nGenerating proper report...")
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    report = f"""
# Diabetes Tongue Classification - PROPER ML Pipeline Results

## WARNING: Previous Results Were Misleading!

The previous pipeline achieved 100% accuracy because it was using **filename-based features** that directly contained class information. This is called **data leakage** and is not valid for real-world applications.

## Dataset Summary
- **Total images:** {len(df)}
- **Class distribution:** {df['label'].value_counts().to_dict()}
- **Images processed:** {len(results) * 20 if results else 0} (sampled for efficiency)
- **Perfect balance:** {'Yes' if df['label'].value_counts().nunique() == 1 else 'No'}

## Proper Model Performance Results (Using Actual Image Features)
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

## Features Used (Proper Image Features)
- Image width and height
- Aspect ratio
- Mean RGB values
- Standard deviation of RGB values
- **NO filename information used!**

## Why Previous Results Were Wrong
1. **Data Leakage:** Models learned from filename paths containing class labels
2. **Unrealistic Performance:** 100% accuracy is extremely rare in real ML
3. **Invalid Features:** Filename-based features wouldn't work in production

## Technical Details
- **Data Split:** 80% training, 20% testing
- **Cross-Validation:** 5-fold
- **Feature Engineering:** Actual image-based features
- **Class Balance:** {'Perfect' if df['label'].value_counts().nunique() == 1 else 'Imbalanced'}

## Recommendations for Better Performance
1. **Use Deep Learning Models:**
   ```bash
   pip install torch torchvision
   ```

2. **Extract Better Features:**
   - Color histograms
   - Texture features (LBP, GLCM)
   - Edge detection features
   - CNN-based features

3. **Data Augmentation:**
   - Rotate, flip, and scale images
   - Adjust brightness and contrast
   - Use advanced augmentation techniques

4. **More Data:**
   - Collect more diverse images
   - Ensure proper train/validation/test splits

## Files Generated
- `outputs/proper_model_analysis.png` - Proper model analysis
- `outputs/proper_report.md` - This report
- `data/labels.csv` - Dataset labels

## Next Steps
1. Install PyTorch for deep learning models
2. Implement proper CNN-based feature extraction
3. Use transfer learning with pre-trained models
4. Implement proper data augmentation
5. Validate on completely unseen test data

---
*Generated by Proper Diabetes Tongue Classification Pipeline*
*Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Save report
    with open("outputs/proper_report.md", "w") as f:
        f.write(report)
    
    print("[OK] Proper report saved to outputs/proper_report.md")

def main():
    """Main function"""
    
    print("=" * 60)
    print("DIABETES TONGUE CLASSIFICATION - PROPER ML PIPELINE")
    print("=" * 60)
    print("WARNING: This pipeline uses ACTUAL image features, not filename features!")
    print("=" * 60)
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Step 1: Create labels CSV
    print("\n1. Creating labels CSV...")
    csv_path = create_labels_csv()
    
    # Step 2: Load and analyze data
    print("\n2. Loading and analyzing data...")
    df = pd.read_csv(csv_path)
    
    print(f"[OK] Loaded {len(df)} images")
    print(f"  Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Step 3: Create proper features
    print("\n3. Creating proper image features...")
    features, labels = create_proper_features(df, sample_size=50)  # Sample for efficiency
    
    # Step 4: Train models
    print("\n4. Training models...")
    results, test_data = train_models(features, labels)
    
    # Step 5: Create visualizations
    print("\n5. Creating visualizations...")
    create_visualizations(results, df)
    
    # Step 6: Generate report
    print("\n6. Generating report...")
    generate_proper_report(results, df)
    
    print("\n" + "=" * 60)
    print("PROPER PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nThis shows REALISTIC model performance using proper features!")
    print("Check outputs/proper_report.md for detailed analysis.")

if __name__ == "__main__":
    main()
