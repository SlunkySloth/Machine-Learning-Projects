# Student-authored report scaffold

**This is an AI-assisted scaffold, not a finished submission.** The supplied rubric explicitly requires 0% AI report content and plagiarism below 10%. Write each section yourself after reading the papers and studying the code. Do not submit this scaffold or the AI-assisted notes unchanged. The institution, not this project, determines what assistance is permitted.

## Front matter

Project title: [choose your title]. Student name(s), roll number(s), college, department, supervisor, academic year: [fill in]. Add the institution's required certificate and declaration only using its approved format; do not fabricate signatures, approvals or contributions.

## Abstract

Write 150–250 words covering the problem, dataset/population, leakage safeguards, compared methods, actual results, principal limitation and contribution. Use the measured values in `reports/results.json`. Describe Outcome classification accurately.

## 1. Introduction, problem definition and objectives

Explain your motivation, who the educational demonstrator serves, what the binary label represents, and what remains outside scope. Formulate measurable objectives. Use `project_brief.md` as a technical reference and express your own understanding.

## 2. Literature survey and research gap

Read the linked papers in `literature_review.md`. Create your own comparison table recording methods, datasets, evaluation design, XAI, calibration and limitations. Cite each factual claim. Explain a specific gap addressed by your experiment without claiming already-established methods are new.

## 3. Dataset and data understanding

Use `dataset_card.md` and the original metadata. Explain every feature and unit, sample size, target imbalance, selected population, missing/invalid values, duplicate checks and limitations. State the actual source mirror and checksum. Include source attribution.

## 4. Preprocessing and exploratory analysis

Explain the zero-sentinel policy and why pregnancy zero is valid. Explain train-fitted median imputation, indicators, scaling and selection. Discuss why IQR flags are retained. Include the training-only distribution, balance, missingness, correlation and outlier plots. For each figure write an interpretation and limitation in your own words; do not simply paste a plot.

## 5. Feature engineering and selection

Describe the three tested representations, mathematical transforms and hypothesis. Include `reports/feature_ablation.csv`, explain the fixed folds and training-only selection, and discuss whether engineering improved the observed CV scores. Explain the distinction between statistical association and a clinically validated derived biomarker.

## 6. System architecture, training and proposed framework

Draw your architecture from the implemented flow in `methodology.md`. Explain ANN layers, activations, training, early stopping and L2 regularization, then compare the other four model families. Record the selected hyperparameters and search budget. Explain soft voting, out-of-fold sigmoid calibration and the validation threshold.

## 7. Experiments, evaluation and interpretation

Include both test-metric tables, the ROC/PR plot, reliability plot and confusion matrices. Define accuracy, precision, recall, F1, specificity, ROC-AUC, average precision and Brier score. Include the bootstrap interval and prespecified statistical comparison with their assumptions. Discuss the actual negative calibration/ensemble findings and CV optimism. Avoid making clinical claims from this holdout.

## 8. Explainability, uncertainty and what-if

Explain exact finite-background Shapley values and their additivity check; include a local example and the limited 12-example summary. Contrast permutation importance with local attribution. Interpret component disagreement as a heuristic. Present a what-if scenario without describing the change as a treatment effect.

## 9. Application implementation and testing

Run the local application. Capture your own desktop/mobile screenshots with prediction, explanation, validation errors and what-if output. Describe usability and input units. Document actual test outcomes from `reports/verification.txt`; distinguish automated verification from your manual observations.

## 10. Discussion, conclusion and future work

Compare the objectives with achieved evidence. Discuss the small historical cohort, missing measurements, lack of external validation, threshold uncertainty, calibration behavior and attribution dependence. Propose realistic next steps such as external evaluation and nested CV. No promise of clinical deployment follows from this project.

## References and appendices

Use a consistent citation style approved by your department. Verify paper metadata. Attach the results appendix, source-file guide, reproducibility commands and genuine team contribution log if required.

## Before submission

Replace every placeholder; write your own prose; review every cited paper; check all claims against saved outputs; obtain the institution's required AI-use guidance; run its official plagiarism check; record the measured percentage instead of predicting one; rehearse the live demo and viva. A detector score is not proof of authorship.
