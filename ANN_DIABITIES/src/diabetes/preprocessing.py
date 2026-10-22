import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from .data import FEATURES, ZERO_MISSING


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Treat documented zero sentinels as missing; add experimental predictors.

    No population statistics are learned here. Imputation, scaling and selection
    are learned by the later pipeline steps separately within each training fold.
    """

    def __init__(self, engineered=False):
        self.engineered = engineered

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame) and list(X.columns) != FEATURES:
            raise ValueError("Features must match the documented input order.")
        if np.asarray(X).shape[1] != len(FEATURES):
            raise ValueError("Expected eight input features.")
        self.n_features_in_ = len(FEATURES)
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self)
        df = X[FEATURES].copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=FEATURES)
        df = df.astype(float)
        df[ZERO_MISSING] = df[ZERO_MISSING].replace(0, np.nan)
        if self.engineered:
            df["LogInsulin"] = np.log1p(df["Insulin"])
            df["LogPedigree"] = np.log1p(df["DiabetesPedigreeFunction"])
            df["GlucoseBMIInteraction"] = df["Glucose"] * df["BMI"] / 100
        return df

    def get_feature_names_out(self, input_features=None):
        names = FEATURES + (["LogInsulin", "LogPedigree", "GlucoseBMIInteraction"] if self.engineered else [])
        return np.asarray(names, dtype=object)


def pipeline(model, engineered=False, k="all"):
    return Pipeline([
        ("features", FeatureBuilder(engineered)),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("model", model),
    ])
