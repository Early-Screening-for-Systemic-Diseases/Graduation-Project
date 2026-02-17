# 1️⃣ Base image
FROM python:3.9-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ Copy all project files (including .pkl and app.py)
COPY . .

# 5️⃣ Start FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]