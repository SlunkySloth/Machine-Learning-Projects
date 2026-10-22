"""Validated inference shared by the web application and tests."""
import math
import numpy as np
import pandas as pd
from .data import FEATURES, ZERO_MISSING
from .explain import exact_shapley


def validate_inputs(payload):
    if not isinstance(payload, dict) or set(payload) != set(FEATURES):
        raise ValueError("Supply exactly the eight documented feature names.")
    values, notes = {}, []
    for name in FEATURES:
        value = payload[name]
        if value is None and name in ZERO_MISSING:
            values[name] = np.nan
            notes.append(f"{name} is missing; the training-fitted median will be used.")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number" + (" or null." if name in ZERO_MISSING else "."))
        if value < 0 or value > 1_000_000:
            raise ValueError(f"{name} must be nonnegative and no greater than 1,000,000 (application limit).")
        if name in ["Pregnancies", "Age"] and value != int(value):
            raise ValueError(f"{name} must be an integer.")
        if name == "Age" and value < 21:
            raise ValueError("The dataset covers women aged 21 and above; Age must be at least 21.")
        if name == "DiabetesPedigreeFunction" and value <= 0:
            raise ValueError("DiabetesPedigreeFunction must be positive.")
        if value == 0 and name in ZERO_MISSING:
            notes.append(f"{name}=0 is treated as missing; the training-fitted median will be used.")
        values[name] = float(value)
    return pd.DataFrame([values], columns=FEATURES), notes


def predict(bundle, payload, explain=True):
    row, notes = validate_inputs(payload)
    probability = float(bundle["model"].predict_proba(row)[0, 1])
    components = {name: float(model.predict_proba(row)[0, 1]) for name, model in bundle["components"].items()}
    for name, bounds in bundle["reference"].items():
        value = row[name].iloc[0]
        if pd.isna(value) or (name in ZERO_MISSING and value == 0):
            continue
        if not bounds["min"] <= value <= bounds["max"]:
            notes.append(f"{name} lies outside the training range [{bounds['min']:g}, {bounds['max']:g}]; this is extrapolation.")
    result = {"probability": probability, "predicted_outcome": int(probability >= bundle["threshold"]),
              "threshold": bundle["threshold"], "components": components,
              "component_std": float(np.std(list(components.values()))), "notes": notes,
              "scope": "Educational classification of historical Pima Outcome labels; not a diagnosis or validated future risk forecast."}
    if explain:
        result["explanation"] = exact_shapley(bundle["model"], row.iloc[0], bundle["background"])
    return result


def what_if(bundle, payload, feature, value):
    if feature not in ["Glucose", "BMI", "BloodPressure", "Insulin", "SkinThickness"]:
        raise ValueError("What-if analysis supports measured physiological features only.")
    original, original_notes = validate_inputs(payload)
    scenario_payload = dict(payload)
    scenario_payload[feature] = value
    scenario, scenario_notes = validate_inputs(scenario_payload)
    base_p = float(bundle["model"].predict_proba(original)[0, 1])
    scenario_p = float(bundle["model"].predict_proba(scenario)[0, 1])
    grid = np.linspace(bundle["reference"][feature]["min"], bundle["reference"][feature]["max"], 35)
    rows = pd.concat([original] * len(grid), ignore_index=True)
    rows[feature] = grid
    sweep = bundle["model"].predict_proba(rows)[:, 1]
    notes = list(dict.fromkeys(original_notes + scenario_notes))
    if value is not None and value != 0 and not bundle["reference"][feature]["min"] <= value <= bundle["reference"][feature]["max"]:
        notes.append(f"Scenario {feature} is outside its training range; this is extrapolation.")
    return {"feature": feature, "value": value, "baseline_probability": base_p,
            "scenario_probability": scenario_p, "difference": scenario_p - base_p,
            "curve": [{"value": float(v), "probability": float(p)} for v, p in zip(grid, sweep)],
            "notes": notes, "interpretation": "An association within the model, not a treatment effect or medical recommendation."}
