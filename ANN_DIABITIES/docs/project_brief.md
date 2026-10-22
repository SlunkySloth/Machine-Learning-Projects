# Project brief and objectives — supporting material

Working title: **Diabetes Outcome Classification with an ANN and an Explainable Calibrated Ensemble**.

This is an AI-assisted technical brief, not a student-authored submission. The rubric requires 0% AI content in the report and plagiarism below 10%; prepare your own narrative from the evidence, cite sources, and follow your institution's rules.

## Problem and scope

Classify the binary Outcome label in the historical Pima diabetes benchmark using eight numeric measurements. Compare an artificial neural network with simpler and tree-based models, and test a combined framework that exposes probabilities and feature contributions. The motivation is to study the reliability and interpretation of classification, including errors and calibration, rather than presenting accuracy alone.

The source population comprises Pima women aged at least 21. The dataset has no usable follow-up time, patient identifiers, or external cohort in this project; do not describe the output as a validated forecast of future onset. The original dataset metadata describes the population and target definition. [Dataset metadata mirror](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names).

## Measurable objectives

1. Audit the dataset's features, missing-value sentinels, class proportions, duplicates and outliers.
2. Produce training-only distribution, correlation and quality visualizations with written interpretations based on the saved statistics.
3. Keep median imputation, missingness indicators, scaling and feature selection inside the fitted pipeline and each CV training fold.
4. Compare original, engineered and selected feature representations on identical training CV folds.
5. Tune logistic regression, random forest, histogram gradient boosting, RBF SVM and an MLP ANN using stratified five-fold CV.
6. Compare an equal-weight soft ensemble before and after out-of-fold sigmoid calibration, without assuming either improves performance.
7. Freeze the validation-selected decision threshold, then evaluate an independent test set using accuracy, precision, recall, F1, specificity, ROC-AUC, average precision, Brier score, log loss and ECE.
8. Report stratified bootstrap intervals and a prespecified paired comparison against the training-CV-selected baseline.
9. Explain the full deployed pipeline through raw-feature exact interventional Shapley values, permutation importance, component disagreement and noncausal what-if sweeps.
10. Provide a validated local prediction interface and enough technical evidence to reproduce and discuss the experiment.

## Contribution and research gap

Ensembles, probability calibration, SHAP and interactive explanation are established methods, including in diabetes prediction. This project's contribution is a reproducible integration and controlled experimental comparison for the course rubric; it is not a claim of a new algorithm or worldwide research novelty. The practical question is whether a small-data ANN/ensemble workflow can expose useful probabilities and explanations while avoiding leakage and honestly measuring trade-offs. Consult `literature_review.md` before writing a more specific research-gap argument.

## Success criteria and limitations

Success is a correct, reproducible comparison with a working interface and responsible interpretation. No accuracy target is invented. A failed improvement is a legitimate experimental result. External validation, clinical approval, prospective screening effectiveness and independently assessed originality remain outside the completed technical study.
