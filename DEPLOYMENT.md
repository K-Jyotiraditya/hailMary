# Deployment

## Frontend

The dashboard is static and can stay on GitHub Pages.

Before building for production, point it at a hosted API:

```powershell
$env:VITE_API_URL="https://your-backend-host.example.com"
cd dashboard
npm run deploy
```

If `VITE_API_URL` is not set:

- local browser sessions use `http://localhost:8000`
- non-local builds fall back to the current site origin

## Backend

The FastAPI backend cannot run on GitHub Pages. Host it separately.

### Render

This repo now includes [render.yaml](/C:/Users/yashj/Desktop/HailMary/render.yaml).

Steps:

1. Push the repo to GitHub.
2. In Render, create a new Blueprint or Web Service from the repo.
3. Set any required secrets like API keys in Render environment variables.
4. Set `CORS_ORIGINS` to the frontend origin(s), for example:

```text
https://k-jyotiraditya.github.io,http://localhost:5173
```

5. Deploy and copy the public backend URL.
6. Rebuild the frontend with `VITE_API_URL` set to that backend URL.

### Start command

The backend start command is:

```bash
uvicorn backend.server:app --host 0.0.0.0 --port $PORT
```
