FROM python:3.11-slim

WORKDIR /app

COPY Questionnaire_skin/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Questionnaire_skin/ .

CMD ["python", "-c", "import uvicorn, os; uvicorn.run('app:app', host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))"]
