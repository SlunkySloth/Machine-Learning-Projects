from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TARGET = "Outcome"
SEED = 42


def load_data(path=None):
    path = Path(path or ROOT / "data/pima_raw.csv")
    first = path.read_text().splitlines()[0]
    df = pd.read_csv(path) if "Outcome" in first else pd.read_csv(path, header=None, names=FEATURES + [TARGET])
    if set(df.columns) != set(FEATURES + [TARGET]):
        raise ValueError("Dataset must contain exactly the eight Pima features and Outcome.")
    df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="raise")
    if not df[TARGET].isin([0, 1]).all() or df[TARGET].nunique() != 2:
        raise ValueError("Outcome must contain both binary classes 0 and 1 without missing labels.")
    if np.isinf(df.to_numpy()).any() or (df[FEATURES] < 0).any().any():
        raise ValueError("Infinite and negative measurements are invalid.")
    if df.duplicated(subset=FEATURES).any():
        raise ValueError("Duplicate feature rows require an explicit group-aware split before training.")
    return df


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
