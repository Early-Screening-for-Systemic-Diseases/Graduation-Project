# 1. Lightweight Python base
FROM python:3.9-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Install curl (to download the model)
RUN apt-get update && apt-get install -y curl

# 4. Download the model from HuggingFace
RUN curl -L "https://huggingface.co/Hassan-101/diabetes-survey-model/resolve/main/diabetes_survey_model.pkl?download=true" -o /app/diabetes_survey_model.pkl

# 5. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the project code
COPY . .

# 7. Start the API (adjust main:app if your FastAPI file is different)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
