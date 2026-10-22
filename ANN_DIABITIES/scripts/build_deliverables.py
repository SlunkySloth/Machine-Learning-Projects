"""Generate evidence notes, a notebook and editable HTML / PDF slides.

Run after training: PYTHONPATH=src python scripts/build_deliverables.py
These supporting artifacts remain AI-assisted, not compliant original prose.
"""
from pathlib import Path
import base64
import html
import json
import textwrap
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from diabetes.data import ROOT, FEATURES, load_data


def build_notes(r):
    m = r["operating_test_metrics"]["CalibratedEnsemble"]
    raw = r["default_test_metrics"]["RawEnsemble"]
    calibrated = r["default_test_metrics"]["CalibratedEnsemble"]
    text = f"""# Verified experiment notes — supporting evidence

These notes describe the actual executed experiment. They are AI-assisted technical material to study and rewrite, not a student-authored report.

## Data and exploration

The 768 records are partitioned into 460 training, 154 validation and 154 test rows with seed 42. Positive cases: {r['split_positive_counts']}. The training-only invalid-zero rates include insulin 46.74% and skin thickness 28.48%, which limit the information available. Glucose has the strongest training Pearson association with Outcome (r=0.501), followed by BMI (r=0.337). Correlation is not causation; pairwise missingness and the selected cohort affect these estimates. IQR flags are retained rather than automatically deleted.

## Feature comparison and models

The mean AUC across fixed logistic and ANN learners selected the five-feature ANOVA representation on training CV. Logistic regression alone preferred the original representation, while the ANN improved with selection. The engineered representation did not win. This is an observed experiment, not evidence that feature engineering is generally ineffective.

The highest tuned training-CV AUC was the ANN ({r['tuning']['ANN']['best_cv_auc']:.4f}); it was fixed as the comparator before test evaluation. The best training-CV score is selection-biased. On the test set gradient boosting achieved ROC-AUC {r['default_test_metrics']['GradientBoosting']['roc_auc']:.4f}, above the calibrated ensemble's {m['roc_auc']:.4f}. Do not describe the ensemble as the best test classifier.

## Calibrated ensemble operating point

Validation selected threshold {r['operating_threshold']:.2f} by F1. At this frozen operating point, test accuracy is {m['accuracy']:.4f}, precision {m['precision']:.4f}, recall {m['recall']:.4f}, F1 {m['f1']:.4f}, specificity {m['specificity']:.4f}, ROC-AUC {m['roc_auc']:.4f}, average precision {m['pr_auc']:.4f}, and Brier {m['brier']:.4f}. Confusion matrix: TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']}. The 16 false negatives and 28 false positives preclude interpreting this demonstrator as reliable clinical diagnosis.

The default 0.50 threshold yields accuracy {calibrated['accuracy']:.4f} and recall {calibrated['recall']:.4f}. Lowering the threshold improved recall to {m['recall']:.4f} while reducing specificity from {calibrated['specificity']:.4f} to {m['specificity']:.4f}. Both metric tables are retained so comparisons do not silently mix operating points.

## Calibration and statistics

Sigmoid calibration changed Brier from {raw['brier']:.4f} to {calibrated['brier']:.4f}, a worsening, while ten-bin ECE changed from {raw['ece_10_bins']:.4f} to {calibrated['ece_10_bins']:.4f}. Ranking AUC was unchanged because the learned mapping is monotonic. A lower ECE alone is not a reason to claim probability calibration improved overall.

The ensemble ROC-AUC interval is {r['ensemble_operating_95_ci']['roc_auc'][0]:.4f}–{r['ensemble_operating_95_ci']['roc_auc'][1]:.4f} from 1,000 stratified test bootstrap draws. These intervals condition on the fixed trained model and observed class prevalence. At common threshold 0.50, exact McNemar versus the training-CV-selected ANN yields p={r['statistical_comparison']['mcnemar_exact_p']:.3f}, with two discordant errors in each direction. This does not establish equivalence. The paired ensemble-minus-ANN AUC difference is {r['statistical_comparison']['auc_difference_a_minus_b']:.4f}, with interval {r['statistical_comparison']['paired_auc_difference_95_ci']}. This narrow, conditional ranking comparison is distinct from McNemar and does not establish clinical superiority or dominance over every model.

## Explanations and uncertainty

Exact Shapley enumeration explains the entire deployed model for 12 fixed test cases against 24 training background rows. Maximum additivity error is {r['largest_shapley_additivity_error']:.3g}. The displayed mean attribution plot summarizes this small illustrative subset only. Permutation importance offers a different aggregate view on the test set. No retraining follows either interpretation.

Model disagreement is a descriptive SD across five component probabilities, not a calibrated confidence interval. What-if plots reflect a model association at fixed other measurements; they do not establish treatment effects. The interface warns on missing inputs and extrapolation.

## Files to inspect

`feature_ablation.csv`, both `test_metrics_*.csv`, `tuning_*.csv`, `bootstrap_intervals.json`, `statistical_comparison.json`, `local_explanations.json`, `uncertainty.csv`, `what_if.json`, `training_warnings.json`, and the plots are reproducible evidence. `results_appendix.pdf` collects the study figures. The UI's prediction values come from the saved model rather than invented examples.
"""
    (ROOT / "docs/experiment_notes.md").write_text(text)


def build_notebook():
    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text}
    def code(text):
        return {"cell_type": "code", "metadata": {}, "source": text, "execution_count": None, "outputs": []}
    cells = [
        md("# Diabetes prediction — reproducible analysis\n\nAI-assisted technical notebook. Write the submitted report yourself under the rubric's authorship rule. This notebook inspects the completed experiment; training is a separate explicit command and is not silently rerun."),
        code("from pathlib import Path\nimport sys, json\nROOT = Path.cwd()\nif not (ROOT / 'src/diabetes').exists():\n    ROOT = ROOT.parent\nassert (ROOT / 'src/diabetes').exists(), 'Open from project root or notebooks/'\nsys.path.insert(0, str(ROOT / 'src'))\nimport numpy as np\nimport pandas as pd\nimport joblib\nfrom threadpoolctl import threadpool_limits\nthreadpool_limits(limits=1)\nfrom diabetes.data import load_data, FEATURES, ZERO_MISSING, sha256\nfrom diabetes.predict import predict, what_if\nfrom IPython.display import display, Image\ndf = load_data()\nresults = json.loads((ROOT / 'reports/results.json').read_text())\nassert sha256(ROOT / 'data/pima_raw.csv') == results['dataset_sha256']\nprint('Dataset shape:', df.shape)\nprint('Population: historical Pima women aged 21 and above; Outcome classification only.')"),
        md("## Data understanding and partitions\n\nThe raw file is unchanged. Exploratory plots and preprocessing fitting use only training rows."),
        code("display(df.head())\nprint('Class counts:', df.Outcome.value_counts().to_dict())\nsplits = json.loads((ROOT / 'artifacts/splits.json').read_text())\nassert not (set(splits['train']) & set(splits['test']))\nassert not (set(splits['validation']) & set(splits['test']))\nprint('Split sizes:', results['split_sizes'])\ndisplay(pd.read_csv(ROOT / 'reports/data_quality.csv'))"),
        code("display(Image(filename=str(ROOT / 'reports/figures/feature_distributions.png')))\ndisplay(Image(filename=str(ROOT / 'reports/figures/correlation.png')))"),
        md("## Feature engineering and selection\n\nCompare original inputs, exploratory log/interaction features and fold-local ANOVA selection using fixed CV folds. Selection did not improve every learner. A CV-selected representation is not a clinical biomarker."),
        code("display(pd.read_csv(ROOT / 'reports/feature_ablation.csv'))\nprint('Selected representation:', results['selected_representation'])\ndisplay(pd.DataFrame(results['tuning']).T)"),
        md("## Independent test evaluation\n\nThe first table uses 0.50 for all models; the second uses separately frozen validation thresholds. Average precision is the PR-AUC convention. The ensemble is a prespecified framework and is not guaranteed to win."),
        code("display(pd.read_csv(ROOT / 'reports/test_metrics_default.csv'))\ndisplay(pd.read_csv(ROOT / 'reports/test_metrics_validation_threshold.csv'))\ndisplay(Image(filename=str(ROOT / 'reports/figures/roc_pr_curves.png')))\ndisplay(Image(filename=str(ROOT / 'reports/figures/calibration.png')))"),
        code("print('Conditional bootstrap intervals:')\nprint(json.dumps(results['ensemble_operating_95_ci'], indent=2))\nprint('Prespecified comparison against', results['best_cv_baseline'])\nprint(json.dumps(results['statistical_comparison'], indent=2))\ndisplay(Image(filename=str(ROOT / 'reports/figures/confusion_matrices.png')))"),
        md("## Full-model local explanation and inference\n\nShapley values are exact for the finite 24-row background and all eight raw-feature coalitions. Background substitution is not a causal intervention. The summary plot covers only 12 fixed examples."),
        code("bundle = joblib.load(ROOT / 'artifacts/model.joblib')\ninputs = dict(zip(FEATURES, [2,120,70,25,85,30,.45,32]))\nprediction = predict(bundle, inputs)\nprint(json.dumps(prediction, indent=2))\nassert prediction['explanation']['additivity_error'] < 1e-12\ndisplay(Image(filename=str(ROOT / 'reports/figures/local_shapley.png')))\ndisplay(Image(filename=str(ROOT / 'reports/figures/permutation_importance.png')))"),
        md("## Uncertainty and what-if\n\nComponent SD is a disagreement heuristic, not a confidence interval. The scenario below explores the trained model and is not medical advice or an estimated treatment benefit."),
        code("scenario = what_if(bundle, inputs, 'Glucose', 110)\nprint({key:value for key,value in scenario.items() if key != 'curve'})\ndisplay(Image(filename=str(ROOT / 'reports/figures/uncertainty.png')))\ndisplay(Image(filename=str(ROOT / 'reports/figures/what_if.png')))"),
        md("## Reproduce and reflect\n\nFrom the project root: `PYTHONPATH=src python -m diabetes.train`, then `PYTHONPATH=src python scripts/build_deliverables.py`. See the README for setup.\n\nWrite your own answers: Why did calibration worsen Brier? Why did the engineered ANN perform worse? What does the independent test establish? What additional external validation is necessary?"),
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": results_version()}}, "nbformat": 4, "nbformat_minor": 5}
    for idx, cell in enumerate(cells):
        cell["id"] = f"diabetes-{idx:02d}"
    (ROOT / "notebooks/diabetes_analysis.ipynb").write_text(json.dumps(notebook, indent=2))


def results_version():
    return json.loads((ROOT / "reports/results.json").read_text())["python"]


def build_slides(r):
    m = r["operating_test_metrics"]["CalibratedEnsemble"]
    slides = [
        ("Diabetes outcome classification", "ANN + calibrated ensemble + explanations", ["[Student names / roll numbers] · [College / department]", "Academic research demonstrator · 2026", "AI-assisted supporting deck: revise and write your submission yourself."], None),
        ("Problem and objectives", "Study probabilities, errors and interpretation", ["Classify historical Pima Outcome using eight numeric inputs.", "Compare five model families, including a real MLP ANN.", "Control preprocessing leakage and report independent test evidence.", "Integrate calibration, explanations, disagreement and what-if exploration."], None),
        ("Prior work and contribution", "Integration is established; experiments must justify it", ["Tasin et al. (2023): classifiers, XAI and deployed prediction interfaces.", "Hasan et al. (2024): AutoML, XAI and interactive scenarios.", "Guo et al. (2017): probability calibration; Lundberg & Lee (2017): SHAP.", "Contribution: a reproducible course-level framework and honest controlled comparisons.", "No claim of globally new algorithms or clinical validation."], None),
        ("Dataset and scope", "768 historical records · eight predictors · binary Outcome", ["500 negative / 268 positive cases; population: Pima women aged ≥21.", "Glucose is a two-hour OGTT measurement; input units matter.", "Insulin and skinfold measurements have substantial zero sentinels.", "Historical selected cohort; no external validation or future-onset horizon."], "quality_and_balance"),
        ("Architecture and leakage prevention", "Train 460 · validation 154 · test 154", ["Training: EDA, representation ablation, five-fold search and calibration.", "Within each fold: zero policy → median + indicators → scale → select → fit.", "Validation: F1 threshold choice; freeze deployment before test evaluation.", "Test: final metrics and interpretation only; no tuning feedback."], None),
        ("Exploratory analysis", "Interpret the training data only", ["Training glucose–Outcome Pearson r = 0.501; BMI r = 0.337.", "Training invalid-zero rates: insulin 46.74%; skinfold 28.48%.", "IQR flags retained; extreme values do not automatically imply errors.", "Association and missingness may reflect cohort and collection effects."], "correlation"),
        ("Feature ablation and implementation", "Compare representations before test evaluation", ["Original, log/interaction-engineered and ANOVA-selected representations.", "Identical five-fold CV; fixed logistic and ANN ablation learners.", "Selected-five won average CV AUC across the two learners.", "Engineering did not improve this experiment; retain the negative result.", "Six search candidates × five folds for each of five model families."], None),
        ("ANN and proposed framework", "Five fitted pipelines → soft vote → sigmoid calibration", ["Tuned ANN: 32 ReLU hidden units; Adam learning rate 0.003; L2 α=0.001.", "ANN early stopping uses only an internal training-fold subset.", "Logistic regression, forest, boosting, SVM and ANN vote equally.", "Out-of-fold sigmoid calibration; deployed threshold 0.33 from validation F1.", "Component probability SD exposes disagreement without a coverage claim."], None),
        ("Independent test results", "A combined framework need not beat its components", [f"Ensemble ROC-AUC {m['roc_auc']:.3f}; average precision {m['pr_auc']:.3f}.", f"At threshold 0.33: accuracy {m['accuracy']:.1%}, recall {m['recall']:.1%}, F1 {m['f1']:.3f}.", f"Specificity {m['specificity']:.1%}; precision {m['precision']:.1%}.", "Gradient boosting test ROC-AUC 0.822 exceeded ensemble 0.811.", "Common 0.50 and validation-threshold comparisons are both saved."], "roc_pr_curves"),
        ("Calibration and statistical evidence", "Measure ranking and probability error separately", ["Raw Brier 0.1709 → calibrated Brier 0.1746: a worsening.", "Ten-bin ECE improved, illustrating metric disagreement on a small sample.", "Ensemble ROC-AUC conditional bootstrap 95% CI: 0.736–0.876.", "Comparator fixed on training CV: ANN; McNemar at 0.50 gives p=1.0.", "Paired AUC difference is small; no claim of clinical superiority."], "calibration"),
        ("Explainability and what-if", "Explain the actual deployed model", ["Exact interventional Shapley: 256 coalitions × 24 training background rows.", "Baseline + eight contributions reconstructs the prediction.", "Local examples plus a clearly limited 12-case summary and permutation importance.", "Attributions and hypothetical changes are model associations, not causes."], "local_shapley"),
        ("Live application demonstration", "Validated inputs · probabilities · explanations · scenarios", ["Open localhost, load example, and generate prediction + explanation.", "Compare the five learners and explain disagreement.", "Try a missing insulin measurement and inspect the imputation note.", "Change one Glucose or BMI measurement and compare the scenario.", "Show validation for age outside dataset scope and extrapolation notes."], None),
        ("Limitations and conclusion", "What the evidence supports", ["Small historical single-population dataset; substantial missingness.", "Shared training CV makes best CV estimates optimistic; holdout stays separate.", "No external validation, prospective outcome validation or clinical approval.", "Calibration and feature engineering did not uniformly improve results.", "Delivered: reproducible models, measured trade-offs and explained local predictions."], None),
        ("References and discussion", "Read primary sources and answer in your own words", ["Pima preserved metadata: github.com/jbrownlee/Datasets.", "Tasin et al. (2023), Healthcare Technology Letters, DOI 10.1049/htl2.12039.", "Hasan et al. (2024), Information 16(1), DOI 10.3390/info16010007.", "Guo et al. (2017), ICML / PMLR 70:1321–1330.", "Lundberg & Lee (2017), NeurIPS 30.", "Full links and reading questions: docs/literature_review.md."], None),
    ]
    sections = []
    with PdfPages(ROOT / "docs/presentation.pdf") as pdf:
        for idx, (title, subtitle, bullets, image) in enumerate(slides):
            fig = plt.figure(figsize=(13.33, 7.5), facecolor="#f5f5ed")
            fig.text(.055, .93, "D+   DIABETES RESEARCH STUDIO", fontsize=11, color="#156b68", fontfamily="monospace")
            fig.text(.055, .82, title, fontsize=29, color="#173536", fontfamily="serif")
            fig.text(.055, .755, subtitle, fontsize=13, color="#627574")
            y = .65
            width = 47 if image else 95
            fontsize = 14 if image else 17
            for bullet in bullets:
                wrapped = textwrap.wrap(bullet, width=width)
                fig.text(.055, y, "• " + "\n  ".join(wrapped), fontsize=fontsize, color="#173536", linespacing=1.5, va="top")
                y -= (.037 * len(wrapped) + .035) if image else (.042 * len(wrapped) + .035)
            if image:
                ax = fig.add_axes([.47, .15, .5, .56])
                ax.imshow(plt.imread(ROOT / f"reports/figures/{image}.png"))
                ax.axis("off")
            fig.text(.055, .045, "ACADEMIC PROTOTYPE · NOT A DIAGNOSTIC SERVICE", fontsize=9, color="#627574", fontfamily="monospace")
            fig.text(.91, .045, f"{idx + 1:02d} / {len(slides):02d}", fontsize=10, color="#156b68")
            pdf.savefig(fig)
            plt.close(fig)
            image_html = ""
            if image:
                encoded = base64.b64encode((ROOT / f"reports/figures/{image}.png").read_bytes()).decode()
                image_html = f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(image.replace("_", " "))}">'
            sections.append(f'<section class="slide{" active" if idx == 0 else ""}"><p class="eyebrow">D+ / DIABETES RESEARCH STUDIO</p><h1>{html.escape(title)}</h1><p class="subtitle">{html.escape(subtitle)}</p><div class="content {"with-image" if image else ""}"><ul>{"".join("<li>" + html.escape(b) + "</li>" for b in bullets)}</ul>{image_html}</div><footer><span>AI-ASSISTED SUPPORTING DECK · REVISE BEFORE SUBMISSION</span><span>{idx + 1:02d} / {len(slides):02d}</span></footer></section>')
    document = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Diabetes Prediction · Presentation</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f5ed;color:#173536;font-family:Arial,sans-serif}.slide{display:none;min-height:100vh;padding:5vh 6vw 8vh}.slide.active{display:flex;flex-direction:column}.eyebrow{font:11px monospace;letter-spacing:2px;color:#156b68}h1{font:clamp(30px,4vw,57px) Georgia,serif;margin:3vh 0 12px}.subtitle{color:#627574;font-size:19px;margin:0 0 4vh}.content{flex:1;display:flex;align-items:center;gap:4vw}.content.with-image ul{flex:1}.content img{width:52%;max-height:55vh;object-fit:contain}.content ul{padding-left:24px;margin:0}.content li{font-size:clamp(15px,1.65vw,24px);line-height:1.6;margin:15px 0}footer{font:10px monospace;letter-spacing:1px;color:#627574;display:flex;justify-content:space-between;margin-top:4vh}nav{position:fixed;bottom:15px;right:20px;display:flex;gap:8px}button{background:#156b68;color:white;border:0;padding:10px 17px;border-radius:4px;cursor:pointer}button:focus-visible{outline:3px solid #cc713d}button:disabled{opacity:.3}@media(max-width:700px){.content{display:block}.content img{width:100%;max-height:35vh}.content li{font-size:16px;margin:9px 0}.slide{padding-bottom:80px}}@media print{.slide,.slide.active{display:flex;min-height:auto;height:185mm;break-after:page;padding:10mm}.content li{font-size:16px}.content img{max-height:110mm}h1{font-size:34px}nav{display:none}@page{size:A4 landscape;margin:8mm}}
</style></head><body>''' + "".join(sections) + '''<nav aria-label="Slide navigation"><button id="prev" aria-label="Previous slide">←</button><button id="next" aria-label="Next slide">→</button><button onclick="window.print()">Print / PDF</button></nav><script>
const slides=[...document.querySelectorAll('.slide')];let current=0;function show(value){current=Math.min(slides.length-1,Math.max(0,value));slides.forEach((slide,i)=>slide.classList.toggle('active',i===current));document.getElementById('prev').disabled=current===0;document.getElementById('next').disabled=current===slides.length-1;window.scrollTo(0,0)}document.getElementById('prev').onclick=()=>show(current-1);document.getElementById('next').onclick=()=>show(current+1);document.addEventListener('keydown',event=>{if(['ArrowRight',' ','PageDown'].includes(event.key)){event.preventDefault();show(current+1)}if(['ArrowLeft','PageUp'].includes(event.key)){event.preventDefault();show(current-1)}});show(0);
</script></body></html>'''
    (ROOT / "web/presentation.html").write_text(document)
    (ROOT / "docs/presentation_outline.md").write_text("# Editable presentation content — AI-assisted\n\nReplace identity placeholders and revise in your own words. The standalone HTML works offline and supports arrow keys. PDF slides are also supplied.\n\n" + "\n\n".join(f"## {idx+1}. {title}\n\n{subtitle}\n\n" + "\n".join("- " + bullet for bullet in bullets) for idx, (title, subtitle, bullets, image) in enumerate(slides)) + "\n")


def build_manifest():
    files = []
    for folder in ["src", "scripts", "web", "data", "docs", "notebooks", "artifacts", "reports"]:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.name not in ["manifest.json", "training.log", "verification.txt"]:
                files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (ROOT / "reports/manifest.json").write_text(json.dumps(files, indent=2) + "\n")


if __name__ == "__main__":
    results = json.loads((ROOT / "reports/results.json").read_text())
    build_notes(results)
    build_notebook()
    build_slides(results)
    build_manifest()
    print("Generated experiment notes, analysis notebook, 14-slide HTML/PDF presentation and artifact manifest.")
