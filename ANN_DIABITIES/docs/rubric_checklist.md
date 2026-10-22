# Evaluation rubric coverage

This checklist maps all 13 rows of `EvaluationRubrics_DiabetesPrediction.pdf` to implemented evidence. It is not a guaranteed mark allocation. Technical artifacts are provided; original report authorship, manual presentation, actual contributions and institutional checks remain the student's responsibility.

| # | Component | Marks | Evidence supplied | Remaining student task |
|---|---|---:|---|---|
| 1 | Problem definition & objectives | 5 | `docs/project_brief.md`, scope in UI | Write and defend your own problem statement |
| 2 | Literature survey & research gap | 8 | Six linked primary-source reading entries, comparison questions and modest contribution statement in `docs/literature_review.md` | Read papers, verify details, write your own survey |
| 3 | Dataset & data understanding | 8 | Bundled real CSV; source, feature units and limitations in `docs/dataset_card.md`; quality CSV, checksum | Explain dataset selection and limitations |
| 4 | Data preprocessing | 5 | Fold-local zero handling, imputation/indicators, scaling and selection; no resampling leakage; retained IQR flags; seed/splits | Explain policies and leakage prevention |
| 5 | Exploratory data analysis | 7 | Training distributions, correlation, balance, missingness and outlier plots; statistics and evidence notes | Interpret each plot independently |
| 6 | Feature engineering & selection | 7 | Original/engineered/ANOVA-selected comparison across two learners in `reports/feature_ablation.csv` | Explain hypotheses and observed comparison |
| 7 | Implementation | 10 | Five candidate models including ANN; dummy; five-fold CV; six search candidates per model; tuning CSVs | Walk through model training code |
| 8 | Novelty / proposed framework | 10 | Raw and calibrated soft ensemble, disagreement, exact Shapley, what-if, connected UI | Defend integration without claiming global algorithmic novelty |
| 9 | Evaluation & experimental analysis | 8 | Both metric tables, ROC/PR, Brier/ECE, confusion matrices, bootstrap CIs, exact McNemar and paired AUC comparison | Discuss errors, negative results and assumptions |
| 10 | Explainability & interpretation | 8 | Full-model local exact Shapley, 12-case summary, permutation importance, uncertainty plot and limitations | Explain noncausal interpretation |
| 11 | Application / UI | 4 | Local Python HTTP app, validated inputs, predictions, plots, what-if and explanation | Perform live demo and capture your own screenshots |
| 12 | Documentation & presentation | 15 | README, architecture, methodology, notebook, evidence PDF, report scaffold and editable presentation | Fill identity details, write original report, revise slides and complete official plagiarism check |
| 13 | Viva & individual contribution | 5 | `docs/viva_guide.md`, troubleshooting prompts and contribution template | Study, rehearse, record real contributions and answer viva yourself |

Review I: rows 1–6, **40 marks**, **8 October 2026**. Review II: rows 7–11, **40 marks**, **15 October 2026**. Review III: rows 12–13, **20 marks**, **22 October 2026**. Total: **100 marks**.

The rubric's report instruction is **“0% AI content and plagiarism must be <10%.”** None of these AI-assisted prose files is represented as compliant student-authored text. No plagiarism percentage or detector outcome is fabricated. Follow your institution's rule before submitting any generated material.
