"""Backend-only inference and explanation interface for the chiller project.

This module has no website framework dependency. A website can create one
ChillerBackend instance at startup and call its prediction/explanation methods.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

try:  # Support both package imports and running from the extracted folder.
    from .rule_engine import OUTPUT_COLUMNS, rank_fault_candidates
except ImportError:
    from rule_engine import OUTPUT_COLUMNS, rank_fault_candidates


class ChillerBackend:
    """Load saved artifacts once and expose prediction and SHAP methods."""

    def __init__(self, artifact_dir: str | Path | None = None):
        self.package_dir = Path(__file__).resolve().parent
        self.artifact_dir = Path(artifact_dir) if artifact_dir else self.package_dir / "model_exports"
        with (self.artifact_dir / "model_metadata.json").open(encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.input_columns = list(self.metadata["input_columns"])
        self.output_columns = list(self.metadata["output_columns"])
        if len(self.input_columns) != 17 or len(self.output_columns) != 26:
            raise ValueError("Model metadata must define exactly 17 inputs and 26 outputs.")
        if tuple(self.output_columns) != OUTPUT_COLUMNS:
            raise ValueError("Metadata output order does not match the rule engine contract.")

        self.model = tf.keras.models.load_model(
            self.artifact_dir / "mlp_model.keras", compile=False
        )
        self.scaler = joblib.load(self.artifact_dir / "input_scaler.pkl")
        self._shap_archive_path = self.package_dir / "shap_results_500_rows.npz"

    def predict_timeseries(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Predict rows from a DataFrame containing timestamp + 17 inputs.

        Missing, invalid, or non-finite inputs reject the upload; no silent
        imputation is applied. Extra columns are ignored. Rows are sorted by
        timestamp. One result is returned per input row.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if len(data) == 0:
            raise ValueError("The uploaded time series has no data rows.")
        if len(data) > 5000:
            raise ValueError("The current time-series upload limit is 5,000 rows.")

        frame = data.copy()
        frame.columns = [str(column).strip() for column in frame.columns]
        if frame.columns.duplicated().any():
            raise ValueError("The upload contains duplicate column names.")
        if "timestamp" not in frame.columns:
            raise ValueError("Required column missing: timestamp.")
        missing_columns = [name for name in self.input_columns if name not in frame.columns]
        if missing_columns:
            raise ValueError("Required sensor columns missing: " + ", ".join(missing_columns))

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        if frame["timestamp"].isna().any():
            raise ValueError("One or more timestamps are missing or invalid.")
        sensors = frame[self.input_columns].apply(pd.to_numeric, errors="coerce")
        if sensors.isna().any().any():
            raise ValueError("Sensor readings contain missing or non-numeric values; none were imputed.")
        if not np.isfinite(sensors.to_numpy(dtype=float)).all():
            raise ValueError("Sensor readings must all be finite numeric values.")

        frame = pd.concat([frame[["timestamp"]], sensors], axis=1)
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        scaled = self.scaler.transform(frame[self.input_columns])
        predictions = np.asarray(
            self.model.predict(np.asarray(scaled, dtype=np.float32), verbose=0)
        )
        expected_shape = (len(frame), len(self.output_columns))
        if predictions.shape != expected_shape:
            raise RuntimeError(
                f"Model returned shape {predictions.shape}; expected {expected_shape}."
            )

        results: list[dict[str, Any]] = []
        for row_index, prediction in enumerate(predictions):
            score_map = {
                name: (float(value) if np.isfinite(value) else None)
                for name, value in zip(self.output_columns, prediction)
            }
            ranking = rank_fault_candidates(score_map)
            results.append(
                {
                    "timestamp": frame.loc[row_index, "timestamp"].isoformat(),
                    "status": ranking["assessment_status"],
                    "message": ranking["message"],
                    "sensor_readings": {
                        name: float(frame.loc[row_index, name]) for name in self.input_columns
                    },
                    "model_scores": score_map,
                    "attention_candidates": ranking["candidates"],
                    "errors": ranking.get("errors", []),
                }
            )
        return results

    def get_global_shap(self, output_name: str) -> dict[str, Any]:
        """Return saved global SHAP values and aligned raw sensor rows."""
        if output_name not in self.output_columns:
            raise ValueError(f"Unknown model output: {output_name}")
        explanation_path = self.package_dir / "shap_explanation_inputs_500.csv"
        if not explanation_path.exists() or not self._shap_archive_path.exists():
            raise FileNotFoundError("The SHAP archive or aligned explanation inputs are missing.")

        explanation_inputs = pd.read_csv(explanation_path)
        if list(explanation_inputs.columns) != self.input_columns:
            raise ValueError("SHAP explanation input columns do not match model metadata.")
        with np.load(self._shap_archive_path, allow_pickle=False) as archive:
            archive_inputs = archive["input_columns"].astype(str).tolist()
            archive_outputs = archive["output_columns"].astype(str).tolist()
            if archive_inputs != self.input_columns or archive_outputs != self.output_columns:
                raise ValueError("SHAP archive column order does not match model metadata.")
            values = np.asarray(archive[f"shap_values__{output_name}"])
            bases = np.asarray(archive[f"base_values__{output_name}"])

        if values.shape != (len(explanation_inputs), len(self.input_columns)):
            raise ValueError("SHAP values and explanation input rows have incompatible shapes.")
        return {
            "output_name": output_name,
            "feature_names": self.input_columns,
            "values": values,
            "base_values": bases,
            "data": explanation_inputs.to_numpy(dtype=float),
        }

    def explain_local(self, sensor_readings: dict[str, float], output_name: str) -> dict[str, Any]:
        """Compute one on-demand local permutation-SHAP explanation.

        This is intentionally a separate call because it is slower than model
        inference. ``sensor_readings`` must contain all 17 model inputs.
        """
        if output_name not in self.output_columns:
            raise ValueError(f"Unknown model output: {output_name}")
        missing = [name for name in self.input_columns if name not in sensor_readings]
        if missing:
            raise ValueError("Sensor readings missing: " + ", ".join(missing))
        row = pd.DataFrame(
            [[float(sensor_readings[name]) for name in self.input_columns]],
            columns=self.input_columns,
        )
        if not np.isfinite(row.to_numpy(dtype=float)).all():
            raise ValueError("Sensor readings must all be finite numeric values.")

        background_path = self.package_dir / "shap_background_inputs_100.csv"
        if not background_path.exists():
            raise FileNotFoundError("The saved SHAP background inputs are missing.")
        background_raw = pd.read_csv(background_path)
        background_scaled = self.scaler.transform(background_raw[self.input_columns])
        row_scaled = self.scaler.transform(row)
        output_index = self.output_columns.index(output_name)

        def predict_one(scaled_rows):
            batch = self.model.predict(np.asarray(scaled_rows, dtype=np.float32), verbose=0)
            return batch[:, output_index]

        import shap

        explainer = shap.PermutationExplainer(predict_one, background_scaled)
        explanation = explainer(row_scaled, max_evals=2 * len(self.input_columns) + 1)
        values = np.asarray(explanation.values).reshape(-1, len(self.input_columns))[0]
        baseline = float(np.asarray(explanation.base_values).reshape(-1)[0])
        predicted_score = float(predict_one(row_scaled)[0])
        if not np.isfinite(predicted_score) or not 0.0 <= predicted_score <= 1.0:
            raise ValueError(
                "Local explanation is unavailable because this output score is outside [0, 1]."
            )
        return {
            "output_name": output_name,
            "feature_names": self.input_columns,
            "feature_values": row.iloc[0].to_dict(),
            "shap_values": {name: float(value) for name, value in zip(self.input_columns, values)},
            "baseline": baseline,
            "predicted_score": predicted_score,
        }
