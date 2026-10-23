"""FastAPI application for the MachinaSense college prototype."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from .chiller_backend import ChillerBackend
    from .rule_engine import ATTENTION_THRESHOLD
except ImportError:  # Also support the extracted backend-only ZIP layout.
    from chiller_backend import ChillerBackend
    from rule_engine import ATTENTION_THRESHOLD


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = PACKAGE_DIR / "data" / "machinasense.sqlite3"
DB_PATH = Path(os.environ.get("MACHINASENSE_DB_PATH", str(DEFAULT_DB)))
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_CSV_ROWS = 5000

# The dictionary distinguishes its normal operating range from the wider
# extended range used when generating training samples. Values beyond the
# extended range are rejected; normal-range deviations are warnings.
INPUT_RANGES: dict[str, dict[str, float]] = {
    "F36": {"normal_min": 0, "normal_max": 40, "min": -10, "max": 50},
    "F15": {"normal_min": 1, "normal_max": 10, "min": 0, "max": 10},
    "H25": {"normal_min": 28, "normal_max": 35, "min": 18, "max": 45},
    "H24": {"normal_min": 40, "normal_max": 55, "min": 30, "max": 65},
    "H64": {"normal_min": 28, "normal_max": 38, "min": 16, "max": 50},
    "H15": {"normal_min": 50, "normal_max": 75, "min": 35, "max": 90},
    "F14": {"normal_min": 1, "normal_max": 160, "min": 0, "max": 160},
    "H42": {"normal_min": 4.5, "normal_max": 11, "min": 1.6875, "max": 13.8125},
    "Mode_1Cool_2Heat": {"normal_min": 1, "normal_max": 2, "min": 1, "max": 2},
    "H23F": {"normal_min": 3, "normal_max": 10, "min": 0, "max": 13},
    "H23C": {"normal_min": 8, "normal_max": 12, "min": 6, "max": 14},
    "P02": {"normal_min": 0.1, "normal_max": 0.8, "min": 0, "max": 1.5},
    "H20": {"normal_min": 0.1, "normal_max": 5, "min": 0, "max": 5},
    "P03": {"normal_min": 1.5, "normal_max": 3, "min": 0, "max": 4.5},
    "F37": {"normal_min": 8, "normal_max": 20, "min": 3, "max": 24},
    "F39": {"normal_min": 3, "normal_max": 15, "min": -1, "max": 22},
    "H62": {"normal_min": 0, "normal_max": 1, "min": 0, "max": 1},
}


class SinglePredictionRequest(BaseModel):
    timestamp: datetime | None = None
    sensor_readings: dict[str, Any]


class LocalExplanationRequest(BaseModel):
    output_name: str
    sensor_readings: dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    initialize_database()
    app.state.backend = ChillerBackend()
    app.state.model_version = artifact_fingerprint(app.state.backend)
    yield


app = FastAPI(
    title="MachinaSense API",
    version="1.0.0",
    description="Chiller screening prototype using saved MLP and SHAP artifacts.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


def initialize_database() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS prediction_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                row_count INTEGER NOT NULL
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS prediction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES prediction_runs(id) ON DELETE CASCADE,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                sensor_readings_json TEXT NOT NULL,
                model_scores_json TEXT NOT NULL,
                candidates_json TEXT NOT NULL,
                errors_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                threshold REAL NOT NULL,
                model_version TEXT NOT NULL
            )"""
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_results_run ON prediction_results(run_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_results_time ON prediction_results(timestamp)")


def backend() -> ChillerBackend:
    return app.state.backend


def artifact_fingerprint(model_backend: ChillerBackend) -> str:
    """Fingerprint the exact model/scaler/metadata bundle used for inference."""
    digest = hashlib.sha256()
    for filename in ("mlp_model.keras", "input_scaler.pkl", "model_metadata.json"):
        path = model_backend.artifact_dir / filename
        digest.update(filename.encode("utf-8"))
        with path.open("rb") as artifact:
            for block in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()[:16]


def validate_sensor_row(values: dict[str, Any], *, row_label: str = "Input") -> tuple[dict[str, float], list[str]]:
    model = backend()
    missing = [name for name in model.input_columns if name not in values]
    if missing:
        raise HTTPException(422, f"{row_label} is missing required sensor columns: {', '.join(missing)}")

    parsed: dict[str, float] = {}
    for name in model.input_columns:
        try:
            value = float(values[name])
        except (TypeError, ValueError, OverflowError):
            raise HTTPException(422, f"{row_label}: {name} must be a numeric value.") from None
        if not np.isfinite(value):
            raise HTTPException(422, f"{row_label}: {name} must be finite.")
        limits = INPUT_RANGES[name]
        if value < limits["min"] or value > limits["max"]:
            raise HTTPException(
                422,
                f"{row_label}: {name}={value:g} is outside the allowed dataset range "
                f"[{limits['min']:g}, {limits['max']:g}].",
            )
        parsed[name] = value

    if parsed["Mode_1Cool_2Heat"] not in (1.0, 2.0):
        raise HTTPException(422, f"{row_label}: Mode_1Cool_2Heat must be 1 (Cool) or 2 (Heat).")
    if parsed["H62"] not in (0.0, 1.0):
        raise HTTPException(422, f"{row_label}: H62 must be 0 or 1.")

    if parsed["Mode_1Cool_2Heat"] == 1.0:
        parsed["H23C"] = 10.0
    else:
        parsed["H23F"] = 6.5

    warnings = [
        f"{name}={parsed[name]:g} is outside the dictionary's normal range "
        f"[{INPUT_RANGES[name]['normal_min']:g}, {INPUT_RANGES[name]['normal_max']:g}]."
        for name in model.input_columns
        if name != "Mode_1Cool_2Heat"
        and not (INPUT_RANGES[name]["normal_min"] <= parsed[name] <= INPUT_RANGES[name]["normal_max"])
    ]
    return parsed, warnings


def persist_results(source: str, results: list[dict[str, Any]]) -> int:
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        cursor = connection.execute(
            "INSERT INTO prediction_runs(source, created_at, row_count) VALUES (?, ?, ?)",
            (source, created_at, len(results)),
        )
        run_id = int(cursor.lastrowid)
        connection.executemany(
            """INSERT INTO prediction_results (
                run_id, timestamp, status, sensor_readings_json, model_scores_json,
                candidates_json, errors_json, warnings_json, threshold, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    run_id,
                    result["timestamp"],
                    result["status"],
                    json.dumps(result["sensor_readings"]),
                    json.dumps(result["model_scores"]),
                    json.dumps(result["attention_candidates"]),
                    json.dumps(result.get("errors", [])),
                    json.dumps(result.get("input_warnings", [])),
                    ATTENTION_THRESHOLD,
                    app.state.model_version,
                )
                for result in results
            ],
        )
    return run_id


def add_warnings(result: dict[str, Any]) -> dict[str, Any]:
    readings = result["sensor_readings"]
    result["input_warnings"] = [
        f"{name}={float(readings[name]):g} is outside the dictionary's normal range "
        f"[{INPUT_RANGES[name]['normal_min']:g}, {INPUT_RANGES[name]['normal_max']:g}]."
        for name in backend().input_columns
        if name != "Mode_1Cool_2Heat"
        and not (INPUT_RANGES[name]["normal_min"] <= float(readings[name]) <= INPUT_RANGES[name]["normal_max"])
    ]
    result["threshold"] = ATTENTION_THRESHOLD
    result["model_version"] = app.state.model_version
    result["score_note"] = "Model score aligned with synthetic expert-system targets; not a calibrated failure probability."
    return result


def prepare_csv(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise HTTPException(422, "The CSV contains duplicate column names.")
    if "timestamp" not in frame.columns:
        raise HTTPException(422, "The CSV must include a column named timestamp.")
    missing = [name for name in backend().input_columns if name not in frame.columns]
    if missing:
        raise HTTPException(422, "The CSV is missing required columns: " + ", ".join(missing))
    if not 1 <= len(frame) <= MAX_CSV_ROWS:
        raise HTTPException(422, f"CSV must contain between 1 and {MAX_CSV_ROWS:,} data rows.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if frame["timestamp"].isna().any():
        bad_lines = (frame.index[frame["timestamp"].isna()][:5] + 2).tolist()
        raise HTTPException(422, f"The CSV has missing or invalid timestamps (example lines: {bad_lines}).")
    frame[backend().input_columns] = frame[backend().input_columns].apply(pd.to_numeric, errors="coerce")
    if frame[backend().input_columns].isna().any().any():
        locations = []
        for row_idx, col_idx in np.argwhere(frame[backend().input_columns].isna().to_numpy())[:8]:
            locations.append(f"row {row_idx + 2}, {backend().input_columns[col_idx]}")
        raise HTTPException(422, "CSV has missing or non-numeric readings: " + "; ".join(locations))
    sensor_values = frame[backend().input_columns].to_numpy(dtype=float)
    if not np.isfinite(sensor_values).all():
        raise HTTPException(422, "Sensor readings must all be finite numeric values.")

    for row_index, row in frame.iterrows():
        values, _ = validate_sensor_row(row.to_dict(), row_label=f"CSV row {row_index + 2}")
        for name, value in values.items():
            frame.at[row_index, name] = value
    return frame[["timestamp", *backend().input_columns]].sort_values("timestamp", kind="stable").reset_index(drop=True)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "loaded", "database": "sqlite"}


@app.get("/api/config")
def config() -> dict[str, Any]:
    model = backend()
    inputs = []
    for name in model.input_columns:
        limits = INPUT_RANGES[name]
        if name == "Mode_1Cool_2Heat":
            kind, default = "mode", 1.0
        elif name == "H62":
            kind, default = "binary", 0.0
        else:
            kind = "number"
            default = (limits["normal_min"] + limits["normal_max"]) / 2
        inputs.append({"name": name, "kind": kind, "default": default, **limits})
    return {
        "project_name": "MachinaSense",
        "input_columns": inputs,
        "output_columns": model.output_columns,
        "attention_threshold": ATTENTION_THRESHOLD,
        "model_version": app.state.model_version,
        "threshold_comparison": ">",
        "max_csv_rows": MAX_CSV_ROWS,
        "score_note": "Scores align with synthetic fuzzy expert-system targets; they are not calibrated real-world failure probabilities.",
        "time_series_note": "Timestamped readings are scored independently. The system does not forecast failures or estimate remaining useful life.",
    }


@app.post("/api/predict")
def predict_one(request: SinglePredictionRequest) -> dict[str, Any]:
    readings, warnings = validate_sensor_row(request.sensor_readings)
    timestamp = request.timestamp or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    row = {"timestamp": timestamp.isoformat(), **readings}
    try:
        results = backend().predict_timeseries(pd.DataFrame([row]))
    except (ValueError, RuntimeError) as error:
        raise HTTPException(422, str(error)) from error
    result = add_warnings(results[0])
    result["input_warnings"] = warnings
    run_id = persist_results("single", [result])
    result["run_id"] = run_id
    return result


@app.post("/api/predict/csv")
async def predict_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(415, "Upload a .csv file.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "CSV file exceeds the 20 MB upload limit.")
    try:
        frame = prepare_csv(pd.read_csv(BytesIO(content)))
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(422, f"Could not read the CSV: {error}") from error
    try:
        results = backend().predict_timeseries(frame)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(422, str(error)) from error
    results = [add_warnings(result) for result in results]
    run_id = persist_results("csv", results)
    for result in results:
        result["run_id"] = run_id
    return {"run_id": run_id, "count": len(results), "results": results}


@app.get("/api/shap/global/{output_name}")
def global_shap(output_name: str) -> dict[str, Any]:
    try:
        result = backend().get_global_shap(output_name)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error
    values = np.asarray(result["values"], dtype=float)
    data = np.asarray(result["data"], dtype=float)
    importance = np.mean(np.abs(values), axis=0)
    order = np.argsort(importance)[::-1]
    return {
        "output_name": output_name,
        "feature_names": result["feature_names"],
        "importance": [{"feature": result["feature_names"][i], "mean_abs_shap": float(importance[i])} for i in order],
        "shap_values": values.tolist(),
        "feature_values": data.tolist(),
        "explanation_rows": int(values.shape[0]),
    }


@app.post("/api/shap/local")
def local_shap(request: LocalExplanationRequest) -> dict[str, Any]:
    readings, _ = validate_sensor_row(request.sensor_readings)
    try:
        result = backend().explain_local(readings, request.output_name)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return result


@app.get("/api/history")
def history(limit: int = 50) -> dict[str, Any]:
    if not 1 <= limit <= 100:
        raise HTTPException(422, "limit must be from 1 to 100.")
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """SELECT r.id, r.run_id, r.timestamp, r.status, r.sensor_readings_json,
                      r.model_scores_json, r.candidates_json, r.errors_json,
                      r.warnings_json, r.threshold, r.model_version,
                      p.source, p.created_at
               FROM prediction_results r JOIN prediction_runs p ON p.id = r.run_id
               ORDER BY r.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return {
        "results": [
            {
                "id": row["id"], "run_id": row["run_id"], "timestamp": row["timestamp"],
                "status": row["status"], "sensor_readings": json.loads(row["sensor_readings_json"]),
                "model_scores": json.loads(row["model_scores_json"]),
                "attention_candidates": json.loads(row["candidates_json"]),
                "errors": json.loads(row["errors_json"]),
                "input_warnings": json.loads(row["warnings_json"]),
                "threshold": row["threshold"], "model_version": row["model_version"],
                "source": row["source"], "created_at": row["created_at"],
            }
            for row in rows
        ]
    }


@app.delete("/api/history/{run_id}")
def delete_history(run_id: int) -> dict[str, str]:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        cursor = connection.execute("DELETE FROM prediction_runs WHERE id = ?", (run_id,))
        if cursor.rowcount == 0:
            raise HTTPException(404, "Saved assessment was not found.")
    return {"status": "deleted"}
