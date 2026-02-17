# Lightweight Python base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything (including the .pkl file)
COPY . .

# Start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]