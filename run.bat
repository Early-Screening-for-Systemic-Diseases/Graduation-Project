@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting API on http://localhost:5000
echo.
python app.py
