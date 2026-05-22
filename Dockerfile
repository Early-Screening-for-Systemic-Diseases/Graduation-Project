FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY skin_cancer_api/requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Copy API source
COPY skin_cancer_api/ ./skin_cancer_api/

# Copy only the model file needed for inference
# JSON array syntax is required because the path contains a space
COPY ["pipe output/pytorch_training_binary_2phase/best_model_binary_2phase.pt", \
      "pipe output/pytorch_training_binary_2phase/best_model_binary_2phase.pt"]

WORKDIR /app/skin_cancer_api

RUN chmod +x start.sh

EXPOSE 8000

CMD ["./start.sh"]
