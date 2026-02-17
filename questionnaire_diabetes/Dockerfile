FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files and model
COPY . .

ENV PORT 8000

CMD ["bash","-lc","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT 8000

CMD ["bash","-lc","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]