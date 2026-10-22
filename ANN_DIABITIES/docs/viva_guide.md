# Viva practice and individual contribution worksheet

Use these prompts to check your understanding. Explanations here are study aids; answer in your own words and point to the implementation.

| Question | What a sound answer should cover |
|---|---|
| What exactly are you predicting? | Existing Pima binary Outcome, not a validated future-onset forecast. Population and dataset limits. |
| Why use an ANN on only 768 records? | A comparison model required by the project; small MLP, regularization, early stopping; complexity is not guaranteed to help. |
| Is scikit-learn's MLP a real ANN? | Input features, learned weight matrices and biases, ReLU hidden layer, sigmoid binary output, loss minimization with Adam/backpropagation. |
| Why are some zeros missing but others valid? | Physiological measurement sentinels versus a valid zero pregnancy count; compare `FeatureBuilder`. |
| Where would leakage occur? | Fitting imputation/scaling/selection or resampling on all rows before splitting or CV. Each fitted step stays inside the training fold. |
| Why have validation and test separately? | Training CV selects representation/hyperparameters, validation selects threshold, test estimates held-out performance. |
| Why not fit again on validation after selecting the threshold? | Retraining changes probabilities and can invalidate the frozen operating point; the saved model stays training-only. |
| Are best CV scores unbiased? | No: representation and parameters are selected with those folds. Holdout remains independent; nested CV would improve the CV estimate. |
| How are engineered features justified? | Exploratory skew transforms and interaction hypothesis; controlled original/engineered/selected comparison. Report observed failure to improve. |
| What is the difference between boosting and voting? | Boosting sequentially learns residual/error structure; this soft voter averages independently fitted component probabilities. |
| Why use equal weights? | A prespecified simple ensemble avoids introducing another tuned weight vector on the small dataset. |
| How does calibration work? | Fit sigmoid mapping to out-of-fold probabilities; final voter trained on training rows; does not guarantee Brier improvement. |
| Why can ROC-AUC stay unchanged after calibration? | A monotonic mapping preserves ranking, even when probability magnitudes and threshold decisions change. |
| Why did Brier worsen while ECE improved? | Brier evaluates per-row squared probability error; ECE aggregates by chosen bins and is noisy on small samples. |
| What does threshold 0.33 mean? | An F1-maximizing validation choice from the fixed grid, not a clinical cutoff or diagnostic glucose threshold. |
| Why not report accuracy alone? | Imbalanced classes; false positives and false negatives; sensitivity/specificity; ranking and probability quality. |
| What is your PR-AUC convention? | Average precision, which differs from trapezoidal PR area. |
| What does the bootstrap interval cover? | Conditional test-sampling uncertainty for a fixed model and fixed class prevalence, not all model-training uncertainty. |
| What does McNemar test? | Paired disagreement in correctness at a common threshold; not AUC, calibration or equivalence. |
| What is the contribution if the ensemble loses? | A reproducible integrated workflow with honest ablation, evaluation, explanations and a functional interface; no fabricated improvement. |
| Are these components a globally new method? | No: prior work already combines ensembles and XAI. Describe a course-level integration and specific experimental question. |
| What is a Shapley contribution? | Weighted marginal change across all coalitions; baseline plus contributions reconstructs the model output for the chosen background. |
| Do you explain a surrogate or the final model? | The actual calibrated model including preprocessing, using raw-feature substitutions. |
| Why 256 coalitions? | Two inclusion choices for each of eight features: 2^8. Explain compute cost and background sampling. |
| Is a Shapley value causal? | No; interventional background substitution does not model causal effects and can break feature dependence. |
| Is the global SHAP plot representative? | Only 12 fixed test cases are summarized. Permutation importance is also supplied but remains model-specific. |
| Is model disagreement a confidence interval? | No: a descriptive SD across correlated base learners with no coverage guarantee. |
| Is lowering a what-if prediction a recommendation? | No; holding other inputs fixed explores associations and may create unrealistic profiles. |
| Can this be applied to men, children, or another population? | Not validated; cohort restriction and no external validation. UI age restriction alone does not solve generalization. |
| How do you reproduce the result? | Seed, exact CSV checksum, row indexes, package versions, training command and saved outputs. |

## Troubleshooting rehearsal

Model artifact missing: run the documented training command. Package-version mismatch: reproduce in an environment with `requirements-lock.txt` and the recorded Python version. Port busy: start with `--port 8001`. Incorrect values: inspect units, missingness policy and validation notes. Slow explanations: 256 coalitions × 24 background rows are intentional; keep the five model computations single-threaded for this small workload. Blank results page: inspect server output and confirm `reports/results.json` exists. Explain a bug you actually encountered; do not invent debugging work.

## Honest individual contribution record

| Member and roll number | Actual work completed | Files / experiments demonstrated | Date | Reviewed by |
|---|---|---|---|---|
| [fill in] | [fill in genuine work, including independent understanding and any permitted assistance] | [fill in] | [fill in] | [fill in] |

Record AI/tool assistance according to institutional policy. Do not claim the generated code as wholly unaided work. The 5 viva/contribution marks depend on the student demonstrating understanding and genuine participation; a prepared file cannot earn them automatically.
