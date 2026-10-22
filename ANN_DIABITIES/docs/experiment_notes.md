# Verified experiment notes — supporting evidence

These notes describe the actual executed experiment. They are AI-assisted technical material to study and rewrite, not a student-authored report.

## Data and exploration

The 768 records are partitioned into 460 training, 154 validation and 154 test rows with seed 42. Positive cases: {'train': 160, 'validation': 54, 'test': 54}. The training-only invalid-zero rates include insulin 46.74% and skin thickness 28.48%, which limit the information available. Glucose has the strongest training Pearson association with Outcome (r=0.501), followed by BMI (r=0.337). Correlation is not causation; pairwise missingness and the selected cohort affect these estimates. IQR flags are retained rather than automatically deleted.

## Feature comparison and models

The mean AUC across fixed logistic and ANN learners selected the five-feature ANOVA representation on training CV. Logistic regression alone preferred the original representation, while the ANN improved with selection. The engineered representation did not win. This is an observed experiment, not evidence that feature engineering is generally ineffective.

The highest tuned training-CV AUC was the ANN (0.8214); it was fixed as the comparator before test evaluation. The best training-CV score is selection-biased. On the test set gradient boosting achieved ROC-AUC 0.8222, above the calibrated ensemble's 0.8107. Do not describe the ensemble as the best test classifier.

## Calibrated ensemble operating point

Validation selected threshold 0.33 by F1. At this frozen operating point, test accuracy is 0.7143, precision 0.5758, recall 0.7037, F1 0.6333, specificity 0.7200, ROC-AUC 0.8107, average precision 0.6710, and Brier 0.1746. Confusion matrix: TN=72, FP=28, FN=16, TP=38. The 16 false negatives and 28 false positives preclude interpreting this demonstrator as reliable clinical diagnosis.

The default 0.50 threshold yields accuracy 0.7273 and recall 0.5556. Lowering the threshold improved recall to 0.7037 while reducing specificity from 0.8200 to 0.7200. Both metric tables are retained so comparisons do not silently mix operating points.

## Calibration and statistics

Sigmoid calibration changed Brier from 0.1709 to 0.1746, a worsening, while ten-bin ECE changed from 0.1245 to 0.0863. Ranking AUC was unchanged because the learned mapping is monotonic. A lower ECE alone is not a reason to claim probability calibration improved overall.

The ensemble ROC-AUC interval is 0.7361–0.8759 from 1,000 stratified test bootstrap draws. These intervals condition on the fixed trained model and observed class prevalence. At common threshold 0.50, exact McNemar versus the training-CV-selected ANN yields p=1.000, with two discordant errors in each direction. This does not establish equivalence. The paired ensemble-minus-ANN AUC difference is 0.0152, with interval [0.0005555555555555314, 0.03203703703703687]. This narrow, conditional ranking comparison is distinct from McNemar and does not establish clinical superiority or dominance over every model.

## Explanations and uncertainty

Exact Shapley enumeration explains the entire deployed model for 12 fixed test cases against 24 training background rows. Maximum additivity error is 2.22e-16. The displayed mean attribution plot summarizes this small illustrative subset only. Permutation importance offers a different aggregate view on the test set. No retraining follows either interpretation.

Model disagreement is a descriptive SD across five component probabilities, not a calibrated confidence interval. What-if plots reflect a model association at fixed other measurements; they do not establish treatment effects. The interface warns on missing inputs and extrapolation.

## Files to inspect

`feature_ablation.csv`, both `test_metrics_*.csv`, `tuning_*.csv`, `bootstrap_intervals.json`, `statistical_comparison.json`, `local_explanations.json`, `uncertainty.csv`, `what_if.json`, `training_warnings.json`, and the plots are reproducible evidence. `results_appendix.pdf` collects the study figures. The UI's prediction values come from the saved model rather than invented examples.
