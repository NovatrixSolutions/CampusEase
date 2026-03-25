# Campus Ease - Render Deployment

## Deploy Steps

1. Push this folder to GitHub
2. Go to render.com and sign in
3. Click "New +" → "Web Service"
4. Connect your GitHub repo
5. Configure:
   - **Name**: campus-ease
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
6. Add Environment Variable:
   - **Key**: GROQ_API_KEY
   - **Value**: (paste your Groq API key here)
7. Click "Create Web Service"

Deployment will take 5-10 minutes.

## Files Included
- app.py (Flask backend)
- placement_model.keras (LSTM model)
- placement_model.pkl (Scaler)
- requirements.txt (Dependencies)
- Procfile (Start command)
- templates/index.html
- static/style.css
- static/script.js
