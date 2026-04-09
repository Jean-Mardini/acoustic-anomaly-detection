# First-Shot Acoustic Anomaly Detection Under Domain Shift

Starter repository for an end-to-end acoustic anomaly detection product:
- ML pipeline for anomaly scoring under domain shift
- FastAPI backend for inference and explanations
- Next.js frontend for modern visualization UI

## Step 0 Scope

This commit initializes a clean structure only:
- backend service skeleton
- frontend app skeleton
- root dependency and ignore files

No complex ML logic yet.

## Project Structure

```text
backend/     # API + ML modules
frontend/    # Next.js + Tailwind UI
```

## Quick Start

### Backend

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` for UI and `http://localhost:8000/docs` for API docs.
