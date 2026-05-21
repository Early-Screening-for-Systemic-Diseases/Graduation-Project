FROM python:3.11-slim

# System deps required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY api/requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Copy API source
COPY api/ ./api/

# Copy only the model files needed for inference
COPY Segmentation_Dataset/Segmentation_Branch_Outputs/05_training_runs/checkpoints/best_efficientnetb0_unet.pth \
     Segmentation_Dataset/Segmentation_Branch_Outputs/05_training_runs/checkpoints/best_efficientnetb0_unet.pth

COPY diabetes_pipeline_outputs/05_segmented_training_lighting_robust/best_segmented_lighting_robust_model.pth \
     diabetes_pipeline_outputs/05_segmented_training_lighting_robust/best_segmented_lighting_robust_model.pth

COPY diabetes_pipeline_outputs/11_hybrid_probability_fusion/11_hybrid_model.joblib \
     diabetes_pipeline_outputs/11_hybrid_probability_fusion/11_hybrid_model.joblib

COPY diabetes_pipeline_outputs/11_hybrid_probability_fusion/11_hybrid_scaler.joblib \
     diabetes_pipeline_outputs/11_hybrid_probability_fusion/11_hybrid_scaler.joblib

COPY diabetes_pipeline_outputs/10_hybrid_image_features/10_model_feature_columns.json \
     diabetes_pipeline_outputs/10_hybrid_image_features/10_model_feature_columns.json

COPY diabetes_pipeline_outputs/11_hybrid_probability_fusion/11_hybrid_best_model_summary.json \
     diabetes_pipeline_outputs/11_hybrid_probability_fusion/11_hybrid_best_model_summary.json

WORKDIR /app/api

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
