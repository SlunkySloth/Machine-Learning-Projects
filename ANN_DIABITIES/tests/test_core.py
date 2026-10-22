import io
import json
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from threadpoolctl import threadpool_limits
from diabetes.data import ROOT, FEATURES, ZERO_MISSING, load_data, sha256
from diabetes.preprocessing import FeatureBuilder, pipeline
from diabetes.evaluation import metrics, choose_threshold, bootstrap_intervals, paired_comparison
from diabetes.explain import exact_shapley
from diabetes.predict import validate_inputs, predict, what_if
from diabetes.app import make_handler

EXAMPLE = dict(zip(FEATURES, [2, 120, 70, 25, 85, 30, 0.45, 32]))


class LinearProbability:
    def predict_proba(self, X):
        p = 0.1 + np.asarray(X).sum(axis=1) * 0.01
        return np.column_stack([1 - p, p])


class FakeSocket:
    def __init__(self, request):
        self.request = io.BytesIO(request)
        self.response = io.BytesIO()

    def makefile(self, mode, *args):
        return self.request if "r" in mode else self.response

    def sendall(self, data):
        self.response.write(data)


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        limiter = threadpool_limits(limits=1)
        cls.addClassCleanup(limiter.restore_original_limits)
        cls.bundle = joblib.load(ROOT / "artifacts/model.joblib")
        cls.df = load_data()

    def test_real_dataset_and_fingerprint(self):
        self.assertEqual(self.df.shape, (768, 9))
        self.assertEqual(self.df.Outcome.value_counts().to_dict(), {0: 500, 1: 268})
        result = json.loads((ROOT / "reports/results.json").read_text())
        self.assertEqual(sha256(ROOT / "data/pima_raw.csv"), result["dataset_sha256"])

    def test_partitions_disjoint_and_exhaustive(self):
        split = json.loads((ROOT / "artifacts/splits.json").read_text())
        a, b, c = [set(split[key]) for key in ["train", "validation", "test"]]
        self.assertFalse(a & b or a & c or b & c)
        self.assertEqual(a | b | c, set(self.df.index))
        self.assertEqual([len(a), len(b), len(c)], [460, 154, 154])

    def test_zero_sentinels_and_valid_zero_pregnancy(self):
        row = pd.DataFrame([[0] * 7 + [21]], columns=FEATURES)
        transformed = FeatureBuilder().fit_transform(row)
        self.assertEqual(transformed.Pregnancies.iloc[0], 0)
        self.assertTrue(transformed[ZERO_MISSING].isna().all().all())

    def test_engineered_values_and_names(self):
        row = pd.DataFrame([EXAMPLE], columns=FEATURES)
        builder = FeatureBuilder(engineered=True)
        transformed = builder.fit_transform(row)
        self.assertAlmostEqual(transformed.LogInsulin.iloc[0], np.log1p(85))
        self.assertAlmostEqual(transformed.GlucoseBMIInteraction.iloc[0], 36)
        self.assertEqual(list(builder.get_feature_names_out()), list(transformed.columns))

    def test_no_target_or_reordered_features_allowed(self):
        with self.assertRaises(ValueError):
            FeatureBuilder().fit(self.df)
        with self.assertRaises(ValueError):
            FeatureBuilder().fit(self.df[FEATURES[::-1]])

    def test_imputer_not_updated_by_inference(self):
        sample = self.df.iloc[:100]
        p = pipeline(LogisticRegression(max_iter=1000))
        p.fit(sample[FEATURES], sample.Outcome)
        before = p.named_steps["imputer"].statistics_.copy()
        row = pd.DataFrame([EXAMPLE], columns=FEATURES)
        row.Glucose = 10000
        p.predict_proba(row)
        np.testing.assert_array_equal(before, p.named_steps["imputer"].statistics_)
        clean = sample[FEATURES].copy()
        clean[ZERO_MISSING] = clean[ZERO_MISSING].replace(0, np.nan)
        np.testing.assert_allclose(before, clean.median().to_numpy())

    def test_cross_validation_imputation_uses_fold_training_only(self):
        sample = self.df.iloc[:120]
        cv = StratifiedKFold(3, shuffle=True, random_state=42)
        result = cross_validate(pipeline(LogisticRegression(max_iter=1000)), sample[FEATURES],
                                sample.Outcome, cv=cv, return_estimator=True)
        for estimator, (train_idx, _) in zip(result["estimator"], cv.split(sample[FEATURES], sample.Outcome)):
            clean = sample.iloc[train_idx][FEATURES].copy()
            clean[ZERO_MISSING] = clean[ZERO_MISSING].replace(0, np.nan)
            np.testing.assert_allclose(estimator.named_steps["imputer"].statistics_, clean.median().to_numpy())

    def test_metrics_against_known_confusion_matrix(self):
        result = metrics([0, 0, 1, 1], [.1, .8, .7, .2])
        self.assertEqual([result[k] for k in ["tn", "fp", "fn", "tp"]], [1, 1, 1, 1])
        self.assertEqual(result["specificity"], .5)
        self.assertEqual(result["recall"], .5)
        self.assertAlmostEqual(result["brier"], (.01+.64+.09+.64)/4)

    def test_threshold_tie_prefers_nearest_half(self):
        self.assertAlmostEqual(choose_threshold([0, 1], [.1, .9]), .5)

    def test_bootstrap_repeatable(self):
        y, p = np.array([0, 0, 1, 1]), np.array([.1, .3, .6, .8])
        self.assertEqual(bootstrap_intervals(y, p, .5, 100), bootstrap_intervals(y, p, .5, 100))

    def test_identical_models_paired_comparison(self):
        comparison = paired_comparison([0, 0, 1, 1], [.1, .2, .8, .7], [.1, .2, .8, .7], iterations=100)
        self.assertEqual(comparison["mcnemar_exact_p"], 1)
        self.assertEqual(comparison["paired_auc_difference_95_ci"], [0, 0])

    def test_exact_shapley_known_linear_model(self):
        row = np.arange(1, 9, dtype=float)
        explanation = exact_shapley(LinearProbability(), row, np.zeros((3, 8)))
        np.testing.assert_allclose(list(explanation["contributions"].values()), row * .01, atol=1e-12)
        self.assertLess(explanation["additivity_error"], 1e-12)
        self.assertAlmostEqual(explanation["baseline"], .1)

    def test_saved_explanations_match_deployed_model(self):
        examples = json.loads((ROOT / "reports/local_explanations.json").read_text())
        row = pd.DataFrame([examples[0]["input"]], columns=FEATURES)
        self.assertAlmostEqual(self.bundle["model"].predict_proba(row)[0, 1], examples[0]["probability"])
        self.assertLess(max(e["additivity_error"] for e in examples), 1e-12)

    def test_serialized_inference_matches_test_predictions(self):
        table = pd.read_csv(ROOT / "reports/test_predictions.csv")
        rows = self.df.loc[table.row_id, FEATURES]
        p = self.bundle["model"].predict_proba(rows)[:, 1]
        np.testing.assert_allclose(p, table.CalibratedEnsemble, atol=1e-12)

    def test_prediction_and_explanation(self):
        result = predict(self.bundle, EXAMPLE)
        self.assertTrue(0 <= result["probability"] <= 1)
        self.assertEqual(len(result["components"]), 5)
        self.assertEqual(result["predicted_outcome"], int(result["probability"] >= result["threshold"]))
        self.assertAlmostEqual(result["probability"], result["explanation"]["probability"])

    def test_unknown_and_zero_measurements_supported(self):
        for value in [None, 0]:
            payload = {**EXAMPLE, "Insulin": value}
            result = predict(self.bundle, payload, explain=False)
            self.assertTrue(np.isfinite(result["probability"]))
            self.assertTrue(any("Insulin" in n and "missing" in n for n in result["notes"]))

    def test_invalid_inputs_rejected(self):
        for name, value in [("Age", 20), ("Age", 32.5), ("Pregnancies", -1), ("Glucose", float("nan")),
                            ("Glucose", float("inf")), ("BMI", True), ("DiabetesPedigreeFunction", 0),
                            ("Age", None), ("BMI", 1e8)]:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_inputs({**EXAMPLE, name: value})
        with self.assertRaises(ValueError):
            validate_inputs({**EXAMPLE, "Outcome": 1})

    def test_extrapolation_note(self):
        result = predict(self.bundle, {**EXAMPLE, "Glucose": 300}, explain=False)
        self.assertTrue(any("extrapolation" in n for n in result["notes"]))

    def test_what_if_equal_value_zero_difference(self):
        result = what_if(self.bundle, EXAMPLE, "Glucose", EXAMPLE["Glucose"])
        self.assertAlmostEqual(result["difference"], 0)
        self.assertEqual(len(result["curve"]), 35)
        with self.assertRaises(ValueError):
            what_if(self.bundle, EXAMPLE, "Age", 50)

    def request(self, path, payload=None, headers=None):
        body = b"" if payload is None else json.dumps(payload).encode()
        method = "GET" if payload is None else "POST"
        request_headers = {"Host": "127.0.0.1:8000", "Content-Type": "application/json", "Content-Length": str(len(body))}
        request_headers.update(headers or {})
        request = f"{method} {path} HTTP/1.0\r\n" + "\r\n".join(f"{k}: {v}" for k, v in request_headers.items()) + "\r\n\r\n"
        socket = FakeSocket(request.encode() + body)
        handler = make_handler(self.bundle)
        handler.log_message = lambda *args: None
        handler(socket, ("127.0.0.1", 12345), object())
        raw = socket.response.getvalue()
        head, body = raw.split(b"\r\n\r\n", 1)
        return int(head.split()[1]), body

    def test_http_predict_and_what_if(self):
        status, body = self.request("/api/predict", EXAMPLE)
        self.assertEqual(status, 200)
        self.assertIn("explanation", json.loads(body))
        status, body = self.request("/api/what-if", {"inputs": EXAMPLE, "feature": "BMI", "value": 28})
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["curve"]), 35)

    def test_http_validation_origin_and_content_type(self):
        self.assertEqual(self.request("/api/predict", {**EXAMPLE, "Age": 2})[0], 400)
        self.assertEqual(self.request("/api/predict", EXAMPLE, {"Origin": "http://another-site.example"})[0], 403)
        self.assertEqual(self.request("/api/predict", EXAMPLE, {"Content-Type": "text/plain"})[0], 415)

    def test_http_static_results_and_traversal(self):
        self.assertEqual(self.request("/")[0], 200)
        status, body = self.request("/api/results")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["dataset_rows"], 768)
        self.assertEqual(self.request("/figures/correlation.png")[0], 200)
        self.assertEqual(self.request("/figures/../../artifacts/model.joblib")[0], 404)
        self.assertEqual(self.request("/artifacts/model.joblib")[0], 404)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        unittest.main()
