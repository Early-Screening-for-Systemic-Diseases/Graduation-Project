#!/usr/bin/env bash
# Start the service from the questionnaire_diabetes subfolder (Railpack fallback)
cd questionnaire_diabetes || exit 1
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
