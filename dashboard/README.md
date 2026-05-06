# HailMary Dashboard

This frontend is production-aware.

## Local development

Run the API from the project root:

```bash
uvicorn backend.server:app --reload --port 8000
```

Then run the dashboard:

```bash
npm run dev
```

When running on `localhost`, the frontend automatically uses `http://localhost:8000`.

## Production backend URL

For hosted builds, set `VITE_API_URL` before building:

```bash
VITE_API_URL=https://your-backend-host.example.com npm run build
```

On Windows PowerShell:

```powershell
$env:VITE_API_URL="https://your-backend-host.example.com"
npm run build
```

An example file is included at [dashboard/.env.production.example](/C:/Users/yashj/Desktop/HailMary/dashboard/.env.production.example).

## GitHub Pages

If you deploy the frontend to GitHub Pages, you still need a separately hosted backend.

After setting `VITE_API_URL`, publish with:

```bash
npm run deploy
```
