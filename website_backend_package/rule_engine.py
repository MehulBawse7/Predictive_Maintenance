"""Conservative attention ranking for the chiller MLP's 26 output scores.

The prototype uses a strict > 0.5 attention cutoff. The source rule tables do
not prescribe repair actions, so this module reports inspection candidates and
does not diagnose a confirmed fault or recommend a specific repair.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any


ATTENTION_THRESHOLD = 0.5

OUTPUT_COLUMNS = (
    "Chiller_Condenser_Fault",
    "Chiller_ExpansionValve_Fault",
    "Chiller_Evaporator_Fault",
    "Chiller_Compressor_Fault",
    "Chiller_General_Fault",
    "Condenser_Valve_Fault",
    "Condenser_Filter_Fault",
    "Condenser_GasChargeNeeded",
    "Condenser_Fan_Fault",
    "Condenser_Coil_Fault",
    "Condenser_GasLeak",
    "Evaporator_LiquidLeak",
    "Evaporator_Valve_Fault",
    "Evaporator_StrainerFilter_Fault",
    "Evaporator_Piping_Fault",
    "Evaporator_HeatExchanger_Fault",
    "Evaporator_Electrical_Fault",
    "Evaporator_FlowSwitch_Fault",
    "Compressor_General_Fault",
    "Compressor_Contactor_Fault",
    "Compressor_Leak",
    "Compressor_OilLoss",
    "Compressor_SuperheatIncreaseNeeded",
    "General_Electrical_Fault",
    "General_LowPerformance",
    "General_UnitUndersizedOversized",
)

# Each group has identical source FIS rules and target scores. Preserve both
# names in the result, but do not claim the model can tell them apart.
DUPLICATE_GROUPS = (
    ("Condenser_Valve_Fault", "Condenser_Filter_Fault"),
    ("Condenser_Coil_Fault", "Condenser_GasLeak"),
    ("Evaporator_LiquidLeak", "Evaporator_Valve_Fault"),
    ("Compressor_Contactor_Fault", "Compressor_OilLoss"),
)

PARENT_OUTPUTS = (
    "Chiller_Condenser_Fault",
    "Chiller_Evaporator_Fault",
    "Chiller_Compressor_Fault",
    "Chiller_General_Fault",
)

SUBSYSTEM_PARENT = {
    "Condenser_Valve_Fault": "Chiller_Condenser_Fault",
    "Condenser_Filter_Fault": "Chiller_Condenser_Fault",
    "Condenser_GasChargeNeeded": "Chiller_Condenser_Fault",
    "Condenser_Fan_Fault": "Chiller_Condenser_Fault",
    "Condenser_Coil_Fault": "Chiller_Condenser_Fault",
    "Condenser_GasLeak": "Chiller_Condenser_Fault",
    "Evaporator_LiquidLeak": "Chiller_Evaporator_Fault",
    "Evaporator_Valve_Fault": "Chiller_Evaporator_Fault",
    "Evaporator_StrainerFilter_Fault": "Chiller_Evaporator_Fault",
    "Evaporator_Piping_Fault": "Chiller_Evaporator_Fault",
    "Evaporator_HeatExchanger_Fault": "Chiller_Evaporator_Fault",
    "Evaporator_Electrical_Fault": "Chiller_Evaporator_Fault",
    "Evaporator_FlowSwitch_Fault": "Chiller_Evaporator_Fault",
    "Compressor_General_Fault": "Chiller_Compressor_Fault",
    "Compressor_Contactor_Fault": "Chiller_Compressor_Fault",
    "Compressor_Leak": "Chiller_Compressor_Fault",
    "Compressor_OilLoss": "Chiller_Compressor_Fault",
    "Compressor_SuperheatIncreaseNeeded": "Chiller_Compressor_Fault",
    "General_Electrical_Fault": "Chiller_General_Fault",
    "General_LowPerformance": "Chiller_General_Fault",
    "General_UnitUndersizedOversized": "Chiller_General_Fault",
}

OUTPUT_MEANINGS = {
    "Chiller_Condenser_Fault": "Condenser-related chiller candidate",
    "Chiller_ExpansionValve_Fault": "Expansion-valve candidate",
    "Chiller_Evaporator_Fault": "Evaporator-related chiller candidate",
    "Chiller_Compressor_Fault": "Compressor-related chiller candidate",
    "Chiller_General_Fault": "General chiller candidate",
    "Condenser_Valve_Fault": "Condenser valve / filter ambiguous pair",
    "Condenser_Filter_Fault": "Condenser valve / filter ambiguous pair",
    "Condenser_GasChargeNeeded": "Refrigerant charge may need inspection",
    "Condenser_Fan_Fault": "Condenser fan candidate",
    "Condenser_Coil_Fault": "Condenser coil / refrigerant leak ambiguous pair",
    "Condenser_GasLeak": "Condenser coil / refrigerant leak ambiguous pair",
    "Evaporator_LiquidLeak": "Evaporator leak / valve ambiguous pair",
    "Evaporator_Valve_Fault": "Evaporator leak / valve ambiguous pair",
    "Evaporator_StrainerFilter_Fault": "Evaporator strainer/filter candidate",
    "Evaporator_Piping_Fault": "Evaporator piping candidate",
    "Evaporator_HeatExchanger_Fault": "Evaporator heat exchanger candidate",
    "Evaporator_Electrical_Fault": "Evaporator electrical candidate",
    "Evaporator_FlowSwitch_Fault": "Evaporator flow switch candidate",
    "Compressor_General_Fault": "General compressor candidate",
    "Compressor_Contactor_Fault": "Compressor contactor / oil-loss ambiguous pair",
    "Compressor_Leak": "Compressor leak candidate",
    "Compressor_OilLoss": "Compressor contactor / oil-loss ambiguous pair",
    "Compressor_SuperheatIncreaseNeeded": "Compressor superheat candidate",
    "General_Electrical_Fault": "General electrical candidate",
    "General_LowPerformance": "Low-performance candidate",
    "General_UnitUndersizedOversized": "Unit sizing candidate",
}


def _invalid_result(errors: list[str]) -> dict[str, Any]:
    return {
        "assessment_status": "incomplete",
        "threshold": ATTENTION_THRESHOLD,
        "message": (
            "Assessment incomplete: the model scores could not be safely ranked. "
            "Check the prediction output and input readings."
        ),
        "errors": errors,
        "candidates": [],
    }


def rank_fault_candidates(scores: Mapping[str, float]) -> dict[str, Any]:
    """Validate and rank one prediction's named output scores.

    Scores must contain all 26 outputs, be finite, and lie in the expert
    system's [0, 1] range. Every output is evaluated directly so a high
    component score is not hidden by a lower parent score.
    """

    if not isinstance(scores, Mapping):
        return _invalid_result(["Scores must be provided as a mapping of output name to value."])

    missing = [name for name in OUTPUT_COLUMNS if name not in scores]
    extra = [str(name) for name in scores if name not in OUTPUT_COLUMNS]
    errors: list[str] = []
    if missing:
        errors.append("Missing output scores: " + ", ".join(missing))
    if extra:
        errors.append("Unexpected output names: " + ", ".join(extra))

    numeric: dict[str, float] = {}
    for name in OUTPUT_COLUMNS:
        if name not in scores:
            continue
        try:
            value = float(scores[name])
        except (TypeError, ValueError, OverflowError):
            errors.append(f"{name} is not a numeric score.")
            continue
        if not isfinite(value):
            errors.append(f"{name} is not finite.")
        elif value < 0.0 or value > 1.0:
            errors.append(f"{name}={value:g} is outside the valid [0, 1] score range.")
        numeric[name] = value

    if errors:
        return _invalid_result(errors)

    eligible = set(OUTPUT_COLUMNS)

    # Replace the four aliases with a single candidate row when either name
    # crosses the cutoff; current source rules make their scores identical.
    grouped_names: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for group in DUPLICATE_GROUPS:
        grouped_names.update(group)
        active = [name for name in group if name in eligible and numeric[name] > ATTENTION_THRESHOLD]
        if active:
            group_score = max(numeric[name] for name in group)
            candidate = {
                    "outputs": list(group),
                    "meaning": OUTPUT_MEANINGS[group[0]],
                    "model_score": group_score,
                    "score_label": "expert-system score (not a calibrated probability)",
                    "interpretation": "Ambiguous pair: the supplied expert rules assign these outputs identical scores.",
                    "recommended_action": "Inspection required; no specific repair action is defined in the supplied source rules.",
                }
            parent = SUBSYSTEM_PARENT.get(group[0])
            if parent and numeric[parent] <= ATTENTION_THRESHOLD:
                candidate["parent_context"] = (
                    f"The component score is above the cutoff; its parent subsystem score "
                    f"({numeric[parent]:.3f}) is not. The component is still shown for review."
                )
            candidates.append(candidate)

    for name in eligible:
        if name in grouped_names or numeric[name] <= ATTENTION_THRESHOLD:
            continue
        candidate = {
                "outputs": [name],
                "meaning": OUTPUT_MEANINGS[name],
                "model_score": numeric[name],
                "score_label": "expert-system score (not a calibrated probability)",
                "interpretation": "Elevated expert-system-aligned attention flag; not a confirmed fault.",
                "recommended_action": "Inspection required; no specific repair action is defined in the supplied source rules.",
            }
        parent = SUBSYSTEM_PARENT.get(name)
        if parent and numeric[parent] <= ATTENTION_THRESHOLD:
            candidate["parent_context"] = (
                f"The component score is above the cutoff; its parent subsystem score "
                f"({numeric[parent]:.3f}) is not. The component is still shown for review."
            )
        candidates.append(candidate)

    candidates.sort(key=lambda item: (-item["model_score"], item["outputs"][0]))
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank

    return {
        "assessment_status": "complete",
        "threshold": ATTENTION_THRESHOLD,
        "threshold_comparison": ">",
        "message": (
            "Ranked candidates are model estimates aligned to the supplied fuzzy "
            "expert system. They are not confirmed faults or calibrated failure probabilities."
        ),
        "candidates": candidates,
    }
