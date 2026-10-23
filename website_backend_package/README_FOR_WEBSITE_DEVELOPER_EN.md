# Chiller prediction backend package

This package contains the saved-model backend, the FastAPI application, and
the model artifacts. The React + Vite UI is in the sibling `website/` folder.

## Package contents

- `chiller_backend.py`: saved-model loading, timestamped batch inference, saved
  global SHAP data access, and on-demand local SHAP.
- `rule_engine.py`: score validation, the project-selected strict `> 0.5`
  attention rule, parent/subsystem gating, duplicate-output grouping, and ranking.
- `main.py`: FastAPI endpoints and SQLite assessment-history persistence.
- `model_exports/`: the trained MLP, fitted StandardScaler, and metadata. These
  are copied from the original project and are not retrained or edited.
- `shap_results_500_rows.npz`, `shap_explanation_inputs_500.csv`, and
  `shap_background_inputs_100.csv`: saved SHAP values plus the matching raw
  rows needed to plot global explanations or compute a local explanation.
- `shap_global_importance.csv`: ready-to-display global importance summary.
- `requirements_backend.txt`: Python packages used by the backend.

## Run the API

Use Python 3.11 or a compatible environment for TensorFlow/Keras and the saved
`.keras` model. From the project root, install the backend requirements and
start the package API:

```powershell
$python311 = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
& $python311 -m pip install -r website_backend_package\requirements_backend.txt
& $python311 -m uvicorn website_backend_package.main:app --reload --port 8000
```

If you extracted `website_backend_bundle.zip` into its own folder, run the same
commands there, changing the requirements path to `requirements_backend.txt`
and running `uvicorn main:app`.

The API documentation is available at `http://localhost:8000/docs` while the
service is running. The React/Vite frontend runs separately; follow
`website/README.md`.

The CSV must contain a `timestamp` column and all 17 input columns in
`model_exports/model_metadata.json`. Input columns may arrive in any order;
the service reorders them to the saved model order. Extra columns are ignored.
The service sorts rows by timestamp and accepts up to 5,000 rows per call.
Missing, non-numeric, or non-finite inputs reject the batch. It does not
interpolate or impute values.

Each result contains:

- `timestamp`
- `status` (`complete` or `incomplete`)
- `sensor_readings`
- `model_scores` (the 26 named model outputs)
- `attention_candidates` (rank, output name(s), meaning, score, and cautious
  inspection guidance)
- `errors` when an output score is invalid

An output outside `[0, 1]` makes that timestamp's ranking incomplete. Scores
are estimates of synthetic fuzzy-system targets; they are not calibrated
real-world failure probabilities. The four duplicate output pairs are grouped
because the supplied rules produce identical labels for both members. The
source rules do not define specific repair actions, so the engine only says
inspection is required.

## SHAP methods

For global values for one output:

```python
global_result = backend.get_global_shap("Chiller_Condenser_Fault")
# Includes feature_names, values (500 x 17), base_values, and raw data (500 x 17).
```

For an on-demand local explanation of one timestamp and output:

```python
local_result = backend.explain_local(
    results[0]["sensor_readings"],
    "Chiller_Condenser_Fault",
)
# Includes per-feature SHAP values, baseline, and predicted_score.
```

Local SHAP is slower than prediction. Run it only when the user opens an alert
or explicitly requests an explanation. Positive values push that output above
its baseline; negative values push it below.

## API routes

- `GET /api/health`, `GET /api/config`
- `POST /api/predict` with a JSON timestamp and `sensor_readings` object
- `POST /api/predict/csv` with a multipart CSV file
- `GET /api/shap/global/{output_name}` for saved global importance and beeswarm
  data
- `POST /api/shap/local` with one output name and one sensor-reading object
- `GET /api/history?limit=50` and `DELETE /api/history/{run_id}`

FastAPI also exposes interactive endpoint documentation at `/docs`. The
frontend should display incomplete rows and their errors without showing a
fault ranking, show normal-range deviations as warnings, and label outputs as
model scores rather than failure probabilities.

The API stores submitted assessments locally in SQLite by default at
`website_backend_package/data/machinasense.sqlite3`. Set `MACHINASENSE_DB_PATH`
to choose another file. The prototype has no authentication; use it only on a
trusted local network. This evaluates each timestamp independently. It does
not forecast faults, estimate lead time, or calculate remaining useful life.
