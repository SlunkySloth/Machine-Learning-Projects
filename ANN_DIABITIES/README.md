# Diabetes prediction — rubric project

A working Python study and local application covering the technical requirements in [EvaluationRubrics_DiabetesPrediction.pdf](EvaluationRubrics_DiabetesPrediction.pdf). Includes an ANN, four comparison learners, a dummy baseline, an ensemble, calibration, explainability, a notebook and presentation materials. The real experiment has already been trained and its outputs are included.

**New to machine learning?** Read [the complete beginner guide](docs/BEGINNER_MACHINE_LEARNING_GUIDE.md): project history, every file, actual code excerpts, worked examples, concepts and practice exercises in one document.

**Start here:** [Rubric checklist](docs/rubric_checklist.md) · [Verified experiment notes](docs/experiment_notes.md) · [Report scaffold](docs/report_template.md) · [Presentation PDF](docs/presentation.pdf) · [Viva practice](docs/viva_guide.md).

The rubric requires **0% AI content in the report and plagiarism below 10%**. The supplied prose, slide content and code are AI-assisted support, not a compliant student-authored submission. Write your report yourself, read/cite the sources, revise the slides, and use the institution's originality process. Your name, college and team details remain placeholders. Individual contribution and viva performance must be demonstrated by you.

## Run the application

For a spreadsheet-readable dataset with column names, open `data/pima_with_headers.csv`. It contains the same 768 records as `data/pima_raw.csv`, with the header `Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome`. The original downloaded CSV remains available for the recorded experiment's checksum. Both files are accepted by the training loader; to train with the named copy, use `python train.py --data data/pima_with_headers.csv`.

From this folder, with Python and the listed dependencies available:

```bash
python app.py
```

Open **http://127.0.0.1:8000**. Load the example or enter all eight features, then generate the prediction and explanation. The UI shows the calibrated positive-class probability, validation-selected threshold, five component predictions, exact Shapley contributions, missing/extrapolation notes and what-if comparisons. It also shows the held-out results. Use `python app.py --port 8001` if port 8000 is busy. Stop with Ctrl+C.

The app is a local academic demonstrator. The Pima dataset describes historical women aged 21+ and its existing Outcome label; this project does not validate future diabetes onset or provide diagnosis. Glucose is a two-hour OGTT measurement, not a fasting test. Blank/zero laboratory values are supported; pregnancy zero is valid. The pedigree feature is a dataset-specific score, not a yes/no family-history answer. Inputs are not persisted.

## Fresh environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. The current workspace already has the required scientific packages. The bundled model was produced with **Python 3.14.6** and the versions in `requirements-lock.txt`. To load its binary reliably, use matching package versions. If you use another compatible environment, retrain there rather than assuming scikit-learn serialization works across versions. The broad `requirements.txt` supports installing and retraining; the lock file records the exact artifact-producing versions.

## Reproduce the experiment

```bash
python train.py
PYTHONPATH=src MPLCONFIGDIR=/tmp/diabetes-matplotlib python scripts/build_deliverables.py
python scripts/verify.py
```

The second command is the Linux form; alternatively install this package with `python -m pip install -e .` and run `python scripts/build_deliverables.py` without `PYTHONPATH`. On Windows, set `$env:PYTHONPATH="src"` in PowerShell before running the deliverable builder. The included full training run took about 68 seconds in this environment; other machines may take longer. Training overwrites generated artifacts and reports. The builder regenerates its notebook/notes/slides; preserve your own edited submission separately.

Run `python train.py --data path/to/diabetes.csv` for a compatible eight-feature binary Pima-schema dataset. Headerless and standard-column-name CSVs are supported; negative/infinite values, invalid labels or duplicate feature records fail validation. Revisit population assumptions and feature meaning before substituting another dataset.

Seed 42; stratified 60%/20%/20% train/validation/test; five-fold CV; six randomized candidates per model; 1,000 stratified test bootstrap draws. Each pipeline fits imputation, scaling and selection only on training rows. Training CV chooses features/parameters; validation chooses threshold; the frozen model is evaluated on test rows. No test-based model selection or post-result retuning occurs.

## Actual findings

The deployed calibrated ensemble has test ROC-AUC **0.811** (conditional bootstrap 95% interval **0.736–0.876**). At validation-selected threshold **0.33**, accuracy is **71.4%**, recall **70.4%**, F1 **0.633** and specificity **72.0%**. At common threshold 0.50 its accuracy is **72.7%** and recall **55.6%**. These are different operating points; both tables are saved.

Gradient boosting achieved a higher test ROC-AUC (**0.822**). Calibration worsened ensemble Brier score from **0.1709** to **0.1746** despite reducing binned ECE. The engineered representation did not win the training-CV ablation. These observed trade-offs are retained, not replaced with fabricated improvements. Review the full metric tables and limitations before making any conclusion.

## Deliverables and code guide

| Location | Contents |
|---|---|
| `data/pima_raw.csv` | Unmodified public dataset mirror; real records |
| `docs/project_brief.md` | Problem, motivation, objectives and scope |
| `docs/literature_review.md` | Primary-source reading notes and research-gap guidance |
| `docs/dataset_card.md` | Source, feature units, population and limitations |
| `docs/methodology.md` | Architecture diagram and technical method |
| `docs/experiment_notes.md` | Interpretations tied to the executed results |
| `docs/report_template.md` | Student-authored report scaffold |
| `docs/presentation.pdf` | Fourteen generated supporting slides |
| `web/presentation.html` | Editable, standalone offline slides; arrow keys and print support |
| `docs/presentation_outline.md` | Editable slide content |
| `docs/viva_guide.md` | Concept questions, troubleshooting and genuine contribution worksheet |
| `docs/rubric_checklist.md` | All 13 rubric items, marks, dates and remaining student responsibilities |
| `notebooks/diabetes_analysis.ipynb` | Inspect data, results, explanations and what-if evidence |
| `artifacts/model.joblib` | Frozen fitted model, components, threshold and training background |
| `artifacts/splits.json` | Exact training/validation/test row indexes |
| `reports/results.json` | Complete measured results, versions and data checksum |
| `reports/test_metrics_*.csv` | Common-threshold and validation-threshold comparisons |
| `reports/tuning_*.csv` | Full search configurations and CV scores |
| `reports/feature_ablation.csv` | Original/engineered/selected comparisons |
| `reports/figures/` | EDA, discrimination, calibration, confusion, XAI, uncertainty and what-if plots |
| `reports/results_appendix.pdf` | Collected research figures |
| `reports/verification.txt` | Actual regression-test outcomes |
| `reports/manifest.json` | Generated file sizes and integrity checksums |
| `src/diabetes/preprocessing.py` | Zero handling, experimental features, fold-local pipeline |
| `src/diabetes/train.py` | Splits, CV, tuning, calibration, evaluation and artifacts |
| `src/diabetes/evaluation.py` | Metrics, threshold search, intervals and paired comparisons |
| `src/diabetes/explain.py` | Exact finite-background interventional Shapley algorithm |
| `src/diabetes/predict.py` | Validated inference and scenario comparison |
| `src/diabetes/app.py`, `web/` | Local HTTP API and responsive prediction UI |
| `tests/test_core.py` | Leakage, partition, metrics, inference, attribution and HTTP checks |

Notebook dependencies are optional: `python -m pip install jupyter nbclient nbconvert`. Open the notebook from the project folder or `notebooks/`. It reads the already completed experiment and does not silently retrain. The presentation HTML opens directly from disk and also appears at `/presentation` while the application runs.

Review I: **8 October 2026**, 40 marks. Review II: **15 October 2026**, 40 marks. Review III: **22 October 2026**, 20 marks. The remaining submission work is original report writing, identity details, your own application screenshots, institutional originality checks, presentation revision and individual viva preparation.
