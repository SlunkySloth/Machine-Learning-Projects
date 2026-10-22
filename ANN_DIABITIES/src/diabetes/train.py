"""Run the complete study: PYTHONPATH=src python -m diabetes.train."""
import argparse
import json
import platform
import time
import warnings
from pathlib import Path
from importlib.metadata import version
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.inspection import permutation_importance
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from threadpoolctl import threadpool_limits
from .data import ROOT, FEATURES, TARGET, ZERO_MISSING, SEED, load_data, sha256
from .preprocessing import pipeline
from .evaluation import metrics, choose_threshold, bootstrap_intervals, paired_comparison
from .explain import exact_shapley


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_figure(name):
    plt.tight_layout()
    plt.savefig(ROOT / f"reports/figures/{name}.png", dpi=160, bbox_inches="tight")
    plt.close()


def eda(train):
    """Exploration uses training rows only, reserving validation/test outcomes."""
    clean = train.copy()
    clean[ZERO_MISSING] = clean[ZERO_MISSING].replace(0, np.nan)
    quality = []
    for f in FEATURES:
        s = clean[f]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        quality.append({"feature": f, "missing_or_zero": int(s.isna().sum()),
                        "missing_pct": float(s.isna().mean() * 100),
                        "zero_sentinel_count": int((train[f] == 0).sum()) if f in ZERO_MISSING else 0,
                        "iqr_outliers": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
                        "minimum": float(s.min()), "maximum": float(s.max()), "median": float(s.median())})
    pd.DataFrame(quality).to_csv(ROOT / "reports/data_quality.csv", index=False)
    clean.describe().to_csv(ROOT / "reports/training_descriptive_statistics.csv")
    clean.corr().to_csv(ROOT / "reports/training_correlations.csv")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, f in zip(axes.flat, FEATURES):
        for label, color in [(0, "#167d8d"), (1, "#d47a3d")]:
            ax.hist(clean.loc[clean[TARGET] == label, f].dropna(), bins=18,
                    alpha=0.6, color=color, label=f"Outcome {label}")
        ax.set_title(f)
        ax.set_ylabel("Training observations")
    axes.flat[0].legend()
    save_figure("feature_distributions")
    fig, ax = plt.subplots(figsize=(9, 8))
    corr = clean.corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Pearson correlations — training data; zeros treated as missing")
    save_figure("correlation")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    train[TARGET].value_counts().sort_index().plot.bar(ax=axes[0], color=["#167d8d", "#d47a3d"], rot=0)
    axes[0].set_title("Training target balance")
    axes[0].set_xlabel("Outcome")
    axes[0].set_ylabel("Count")
    pd.DataFrame(quality).set_index("feature")["missing_pct"].plot.barh(ax=axes[1], color="#167d8d")
    axes[1].set_title("Training missing values / invalid zeros (%)")
    save_figure("quality_and_balance")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, f in zip(axes.flat, FEATURES):
        ax.boxplot(clean[f].dropna(), vert=False)
        ax.set_title(f)
    save_figure("outliers")
    return quality, corr


def model_specs():
    return {
        "LogisticRegression": (LogisticRegression(max_iter=2000, random_state=SEED),
                               {"model__C": [0.05, 0.2, 1, 5], "model__class_weight": [None, "balanced"]}),
        "RandomForest": (RandomForestClassifier(n_estimators=180, random_state=SEED, n_jobs=1),
                         {"model__max_depth": [3, 5, None], "model__min_samples_leaf": [2, 5, 10],
                          "model__max_features": ["sqrt", 0.8], "model__class_weight": [None, "balanced"]}),
        "GradientBoosting": (HistGradientBoostingClassifier(random_state=SEED),
                             {"model__max_iter": [80, 140], "model__max_leaf_nodes": [7, 15],
                              "model__learning_rate": [0.04, 0.08], "model__l2_regularization": [1, 5]}),
        "SVM": (SVC(probability=True, random_state=SEED),
                {"model__C": [0.2, 1, 5], "model__gamma": ["scale", 0.03, 0.1],
                 "model__class_weight": [None, "balanced"]}),
        "ANN": (MLPClassifier(max_iter=1500, early_stopping=True, n_iter_no_change=35,
                              validation_fraction=0.15, random_state=SEED),
                {"model__hidden_layer_sizes": [(16,), (32, 16), (32,)],
                 "model__alpha": [0.001, 0.01, 0.1], "model__learning_rate_init": [0.001, 0.003]}),
    }


def run(data_path=None, candidates=6, bootstrap=1000):
    started = time.time()
    for name in ["artifacts", "reports", "reports/figures"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    data_path = Path(data_path or ROOT / "data/pima_raw.csv")
    train_val, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.25, stratify=train_val[TARGET], random_state=SEED)
    X, y = train[FEATURES], train[TARGET]
    Xv, yv, Xt, yt = val[FEATURES], val[TARGET], test[FEATURES], test[TARGET]
    splits = {"train": train.index.tolist(), "validation": val.index.tolist(), "test": test.index.tolist()}
    write_json(ROOT / "artifacts/splits.json", splits)
    print(f"Split: train {len(train)}, validation {len(val)}, untouched test {len(test)}", flush=True)
    quality, corr = eda(train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Compare representations using identical folds; all decisions precede test evaluation.
    ablation = []
    for learner in ["LogisticRegression", "ANN"]:
        model = model_specs()[learner][0]
        for name, engineered, k in [("original", False, "all"), ("engineered", True, "all"), ("selected_5", False, 5)]:
            res = cross_validate(pipeline(model, engineered, k), X, y, cv=cv,
                                 scoring={"auc": "roc_auc", "f1": "f1", "brier": "neg_brier_score"})
            ablation.append({"learner": learner, "representation": name,
                             "mean_cv_auc": float(res["test_auc"].mean()), "std_cv_auc": float(res["test_auc"].std()),
                             "mean_cv_f1": float(res["test_f1"].mean()), "mean_cv_brier": float(-res["test_brier"].mean())})
    pd.DataFrame(ablation).to_csv(ROOT / "reports/feature_ablation.csv", index=False)
    # Prespecified selection: use mean AUC across the two ablation learners.
    selected = pd.DataFrame(ablation).groupby("representation").mean(numeric_only=True)["mean_cv_auc"].idxmax()
    engineered, k = {"original": (False, "all"), "engineered": (True, "all"), "selected_5": (False, 5)}[selected]
    print(f"Training-CV representation selected: {selected}", flush=True)
    models, tuning = {}, {}
    for name, (estimator, params) in model_specs().items():
        print(f"Tuning {name}: {candidates} candidates x 5 folds", flush=True)
        search = RandomizedSearchCV(pipeline(estimator, engineered, k), params, n_iter=candidates,
                                    scoring="roc_auc", cv=cv, refit=True, random_state=SEED,
                                    n_jobs=1, error_score="raise", return_train_score=True)
        search.fit(X, y)
        models[name] = search.best_estimator_
        tuning[name] = {"best_params": search.best_params_, "best_cv_auc": float(search.best_score_),
                        "cv_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                        "cv_train_auc": float(search.cv_results_["mean_train_score"][search.best_index_])}
        pd.DataFrame(search.cv_results_).to_csv(ROOT / f"reports/tuning_{name}.csv", index=False)
    best_baseline = max(tuning, key=lambda name: tuning[name]["best_cv_auc"])
    print(f"Prespecified comparator from training CV: {best_baseline}", flush=True)
    voter = VotingClassifier(estimators=list(models.items()), voting="soft", n_jobs=1)
    calibrated = CalibratedClassifierCV(voter, method="sigmoid", cv=cv, ensemble=False, n_jobs=1)
    print("Fitting ensemble with out-of-fold sigmoid calibration", flush=True)
    calibrated.fit(X, y)
    raw_ensemble = calibrated.calibrated_classifiers_[0].estimator
    dummy = pipeline(DummyClassifier(strategy="prior"))
    dummy.fit(X, y)
    models.update({"DummyPrior": dummy, "RawEnsemble": raw_ensemble, "CalibratedEnsemble": calibrated})
    val_prob = {name: model.predict_proba(Xv)[:, 1] for name, model in models.items()}
    thresholds = {name: choose_threshold(yv, p) for name, p in val_prob.items()}
    operating_threshold = thresholds["CalibratedEnsemble"]
    # Freeze the complete deployment bundle before accessing test predictions or labels.
    bg = X.sample(n=min(24, len(X)), random_state=SEED)
    reference = {f: {"min": float(X[f].replace(0, np.nan).min()) if f in ZERO_MISSING else float(X[f].min()),
                     "max": float(X[f].max())} for f in FEATURES}
    bundle = {"model": calibrated, "components": raw_ensemble.named_estimators_,
              "threshold": operating_threshold, "background": bg, "reference": reference,
              "representation": selected, "features": FEATURES, "seed": SEED,
              "sklearn_version": version("scikit-learn")}
    joblib.dump(bundle, ROOT / "artifacts/model.joblib", compress=3)
    write_json(ROOT / "reports/tuning_summary.json", tuning)
    write_json(ROOT / "reports/validation_thresholds.json", thresholds)
    print(f"Deployment frozen; threshold {operating_threshold:.2f}. Evaluating test once.", flush=True)
    prob = {name: model.predict_proba(Xt)[:, 1] for name, model in models.items()}
    default_metrics = {name: metrics(yt, p) for name, p in prob.items()}
    operating_metrics = {name: metrics(yt, p, thresholds[name]) for name, p in prob.items()}
    pd.DataFrame(default_metrics).T.rename_axis("model").to_csv(ROOT / "reports/test_metrics_default.csv")
    pd.DataFrame(operating_metrics).T.rename_axis("model").to_csv(ROOT / "reports/test_metrics_validation_threshold.csv")
    pred = pd.DataFrame({"row_id": test.index, "Outcome": yt.to_numpy(), **prob})
    pred.to_csv(ROOT / "reports/test_predictions.csv", index=False)
    intervals = bootstrap_intervals(yt, prob["CalibratedEnsemble"], operating_threshold, bootstrap)
    comparison = paired_comparison(yt, prob["CalibratedEnsemble"], prob[best_baseline], iterations=bootstrap)
    write_json(ROOT / "reports/bootstrap_intervals.json", intervals)
    write_json(ROOT / "reports/statistical_comparison.json", {"a": "CalibratedEnsemble", "b": best_baseline, **comparison})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, p in prob.items():
        if name == "DummyPrior":
            continue
        RocCurveDisplay.from_predictions(yt, p, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(yt, p, name=name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[1].axhline(yt.mean(), color="gray", linestyle="--", label="Positive prevalence")
    axes[1].legend(fontsize=8)
    axes[0].legend(fontsize=8)
    save_figure("roc_pr_curves")
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in [best_baseline, "RawEnsemble", "CalibratedEnsemble"]:
        observed, predicted = calibration_curve(yt, prob[name], n_bins=8, strategy="uniform")
        ax.plot(predicted, observed, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", label="Ideal")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="Test reliability diagram")
    ax.legend()
    save_figure("calibration")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, threshold in zip(axes, [0.5, operating_threshold]):
        ConfusionMatrixDisplay.from_predictions(yt, prob["CalibratedEnsemble"] >= threshold,
                                               display_labels=["Outcome 0", "Outcome 1"], ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Calibrated ensemble — threshold {threshold:.2f}")
    save_figure("confusion_matrices")
    importance = permutation_importance(calibrated, Xt, yt, n_repeats=12, random_state=SEED, scoring="roc_auc", n_jobs=1)
    imp = pd.DataFrame({"feature": FEATURES, "auc_drop_mean": importance.importances_mean,
                        "auc_drop_std": importance.importances_std}).sort_values("auc_drop_mean", ascending=False)
    imp.to_csv(ROOT / "reports/permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.iloc[::-1].plot.barh(x="feature", y="auc_drop_mean", xerr="auc_drop_std", ax=ax, color="#167d8d", legend=False)
    ax.set_title("Test permutation importance — explanatory only")
    ax.set_xlabel("Decrease in ROC-AUC; error bars: repeat SD")
    save_figure("permutation_importance")
    print("Computing exact Shapley explanations for 12 fixed test observations", flush=True)
    examples = []
    for idx in range(min(12, len(Xt))):
        result = exact_shapley(calibrated, Xt.iloc[idx], bg)
        result.update({"row_id": int(Xt.index[idx]), "input": Xt.iloc[idx].to_dict(), "observed_outcome": int(yt.iloc[idx])})
        examples.append(result)
    write_json(ROOT / "reports/local_explanations.json", examples)
    mean_phi = np.mean([[abs(e["contributions"][f]) for f in FEATURES] for e in examples], axis=0)
    order = np.argsort(mean_phi)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.asarray(FEATURES)[order], mean_phi[order], color="#167d8d")
    ax.set_title("Mean absolute exact Shapley values — first 12 test rows")
    ax.set_xlabel("Absolute probability contribution (24 training background rows)")
    save_figure("shapley_summary")
    e = examples[0]
    values = np.array(list(e["contributions"].values()))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(FEATURES, values, color=np.where(values >= 0, "#d47a3d", "#167d8d"))
    ax.axvline(0, color="gray")
    ax.set_title(f"Local explanation: baseline {e['baseline']:.3f} → prediction {e['probability']:.3f}")
    ax.set_xlabel("Contribution to positive-class probability")
    save_figure("local_shapley")
    components = np.vstack([m.predict_proba(Xt)[:, 1] for m in raw_ensemble.named_estimators_.values()])
    std = components.std(axis=0)
    pd.DataFrame({"row_id": test.index, "probability": prob["CalibratedEnsemble"],
                  "component_std": std, "absolute_error": abs(yt.to_numpy() - prob["CalibratedEnsemble"])}).to_csv(ROOT / "reports/uncertainty.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(prob["CalibratedEnsemble"], std, c=yt, cmap="coolwarm", alpha=0.7)
    ax.axvline(operating_threshold, linestyle="--", color="gray", label="Validation threshold")
    ax.set(xlabel="Calibrated probability", ylabel="Base-model probability SD", title="Model disagreement — a heuristic, not a confidence interval")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="Observed Outcome")
    save_figure("uncertainty")
    # Out-of-sample what-if sweeps explain the frozen model, never prescribe care.
    base = bg.median().to_frame().T
    scenarios = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, feature in zip(axes, ["Glucose", "BMI"]):
        grid = np.linspace(reference[feature]["min"], reference[feature]["max"], 35)
        rows = pd.concat([base] * len(grid), ignore_index=True)
        rows[feature] = grid
        sweep = calibrated.predict_proba(rows)[:, 1]
        ax.plot(grid, sweep, color="#167d8d")
        ax.set(xlabel=feature, ylabel="Predicted probability", title=f"One-feature what-if: {feature}")
        scenarios.extend({"feature": feature, "value": float(v), "probability": float(p)} for v, p in zip(grid, sweep))
    save_figure("what_if")
    write_json(ROOT / "reports/what_if.json", scenarios)
    summary = {
        "seed": SEED, "dataset_rows": len(df), "dataset_sha256": sha256(data_path),
        "split_sizes": {name: len(ids) for name, ids in splits.items()},
        "split_positive_counts": {"train": int(y.sum()), "validation": int(yv.sum()), "test": int(yt.sum())},
        "selected_representation": selected, "best_cv_baseline": best_baseline,
        "tuning": tuning, "operating_threshold": operating_threshold,
        "default_test_metrics": default_metrics, "operating_test_metrics": operating_metrics,
        "ensemble_operating_95_ci": intervals, "statistical_comparison": comparison,
        "training_quality": quality, "training_target_correlations": corr[TARGET].drop(TARGET).to_dict(),
        "largest_shapley_additivity_error": max(e["additivity_error"] for e in examples),
        "uncertainty_mean_component_std": float(std.mean()), "bootstrap_iterations": bootstrap,
        "candidates_per_model": candidates, "cv_folds": 5,
        "python": platform.python_version(),
        "packages": {pkg: version(pkg) for pkg in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib"]},
        "elapsed_seconds": round(time.time() - started, 2),
        "study_limitations": ["Small historical single-population dataset; no external validation.",
                              "Outcome classification is not a validated future-onset forecast or diagnosis.",
                              "Feature representation and hyperparameters share training CV; CV best scores are selection-biased.",
                              "Calibration folds use training-selected hyperparameters; final holdout remains independent.",
                              "Bootstrap intervals condition on fixed fitted models and observed class prevalence.",
                              "Shapley background substitution ignores feature dependence and cannot establish causation.",
                              "Component disagreement is an uncalibrated heuristic, not predictive coverage.",
                              "Validation F1 threshold is exploratory and not a clinically approved operating point."]}
    write_json(ROOT / "reports/results.json", summary)
    versions = [f"{name}=={ver}" for name, ver in summary["packages"].items()]
    (ROOT / "requirements-lock.txt").write_text("# Versions used for the included artifacts; Python " + summary["python"] + "\n" + "\n".join(versions) + "\n")
    with PdfPages(ROOT / "reports/results_appendix.pdf") as pdf:
        for path in sorted((ROOT / "reports/figures").glob("*.png")):
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.imshow(plt.imread(path))
            ax.axis("off")
            fig.suptitle(path.stem.replace("_", " ").title())
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(pd.DataFrame(default_metrics).T[["accuracy", "recall", "f1", "roc_auc", "pr_auc", "brier"]].round(4).to_string(), flush=True)
    print(f"Finished in {summary['elapsed_seconds']:.1f}s. See reports/results.json", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Headerless Pima CSV or CSV with standard Pima column names")
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.candidates < 1 or args.bootstrap < 100:
        parser.error("Use at least one search candidate and 100 bootstrap iterations.")
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        run(args.data, args.candidates, args.bootstrap)
    write_json(ROOT / "reports/training_warnings.json", sorted(set(str(w.message) for w in captured)))


if __name__ == "__main__":
    main()
