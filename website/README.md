# MachinaSense website

This folder contains the React + Vite interface. The FastAPI application and
the existing saved-model backend live in `../website_backend_package`.

## Run locally

Use Python 3.11 or another environment compatible with the saved Keras model.
In PowerShell, from the project root:

```powershell
$python311 = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
& $python311 -m pip install -r website_backend_package\requirements_backend.txt
& $python311 -m uvicorn website_backend_package.main:app --reload --port 8000
```

If Python 3.11 is installed elsewhere, update `$python311` to that interpreter.

In a second terminal:

```powershell
cd website
npm install
npm run dev
```

Open the Vite URL printed in the terminal (normally `http://localhost:5173`).
The API defaults to `http://localhost:8000`. Set `VITE_API_URL` in a
`website/.env.local` file if the API runs elsewhere. Set `FRONTEND_ORIGIN` for
the FastAPI service if the frontend origin differs from the default.

## Included flows

- Single timestamped reading with the 17 model inputs.
- Timestamped CSV upload with up to 5,000 rows.
- Rule-based ranking with a strict `> 0.5` prototype attention cutoff.
- Saved local assessment history in SQLite. Set `MACHINASENSE_DB_PATH` to
  select another database file.
- Global feature importance and beeswarm plots from the saved 500-row SHAP
  archive, plus on-demand local SHAP for one selected output.
- Input validation using the data dictionary's extended sampling ranges.
  Values outside the normal range but within the accepted range are allowed
  and shown as warnings. No units are assumed because none are documented.

## Interpretation and deployment notes

The model scores reproduce synthetic fuzzy expert-system targets; they are
not calibrated real-world fault probabilities. Each timestamped row is scored
independently. The prototype does not forecast faults or estimate lead time or
remaining useful life. A score outside `[0, 1]` makes that row's ranking
incomplete, and scores are not clipped. Alerts recommend inspection and
qualified maintenance review, not a specific repair. The 0.5 cutoff is a
project-selected rule and should not be described as field-validated.

The API currently has no authentication or production deployment hardening.
Run it on a trusted local network for demonstrations; production deployment
requires authentication, access controls, and a data-retention policy.
