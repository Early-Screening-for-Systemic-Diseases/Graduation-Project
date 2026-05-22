#!/bin/bash
echo "Downloading SpaCy English model..."
python -m spacy download en_core_web_sm

echo "Starting Uvicorn web server..."
# Railway automatically provides the $PORT environment variable
uvicorn main:app --host 0.0.0.0 --port $PORT
