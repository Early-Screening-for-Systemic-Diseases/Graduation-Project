# Use Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (including your .pkl model)
COPY . .

# Start FastAPI
CMD ["python", "-c", "import os,uvicorn; uvicorn.run('app:app', host='0.0.0.0', port=int(os.environ.get('PORT',8000)))"]