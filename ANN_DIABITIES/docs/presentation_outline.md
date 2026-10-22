# Editable presentation content — AI-assisted

Replace identity placeholders and revise in your own words. The standalone HTML works offline and supports arrow keys. PDF slides are also supplied.

## 1. Diabetes outcome classification

ANN + calibrated ensemble + explanations

- [Student names / roll numbers] · [College / department]
- Academic research demonstrator · 2026
- AI-assisted supporting deck: revise and write your submission yourself.

## 2. Problem and objectives

Study probabilities, errors and interpretation

- Classify historical Pima Outcome using eight numeric inputs.
- Compare five model families, including a real MLP ANN.
- Control preprocessing leakage and report independent test evidence.
- Integrate calibration, explanations, disagreement and what-if exploration.

## 3. Prior work and contribution

Integration is established; experiments must justify it

- Tasin et al. (2023): classifiers, XAI and deployed prediction interfaces.
- Hasan et al. (2024): AutoML, XAI and interactive scenarios.
- Guo et al. (2017): probability calibration; Lundberg & Lee (2017): SHAP.
- Contribution: a reproducible course-level framework and honest controlled comparisons.
- No claim of globally new algorithms or clinical validation.

## 4. Dataset and scope

768 historical records · eight predictors · binary Outcome

- 500 negative / 268 positive cases; population: Pima women aged ≥21.
- Glucose is a two-hour OGTT measurement; input units matter.
- Insulin and skinfold measurements have substantial zero sentinels.
- Historical selected cohort; no external validation or future-onset horizon.

## 5. Architecture and leakage prevention

Train 460 · validation 154 · test 154

- Training: EDA, representation ablation, five-fold search and calibration.
- Within each fold: zero policy → median + indicators → scale → select → fit.
- Validation: F1 threshold choice; freeze deployment before test evaluation.
- Test: final metrics and interpretation only; no tuning feedback.

## 6. Exploratory analysis

Interpret the training data only

- Training glucose–Outcome Pearson r = 0.501; BMI r = 0.337.
- Training invalid-zero rates: insulin 46.74%; skinfold 28.48%.
- IQR flags retained; extreme values do not automatically imply errors.
- Association and missingness may reflect cohort and collection effects.

## 7. Feature ablation and implementation

Compare representations before test evaluation

- Original, log/interaction-engineered and ANOVA-selected representations.
- Identical five-fold CV; fixed logistic and ANN ablation learners.
- Selected-five won average CV AUC across the two learners.
- Engineering did not improve this experiment; retain the negative result.
- Six search candidates × five folds for each of five model families.

## 8. ANN and proposed framework

Five fitted pipelines → soft vote → sigmoid calibration

- Tuned ANN: 32 ReLU hidden units; Adam learning rate 0.003; L2 α=0.001.
- ANN early stopping uses only an internal training-fold subset.
- Logistic regression, forest, boosting, SVM and ANN vote equally.
- Out-of-fold sigmoid calibration; deployed threshold 0.33 from validation F1.
- Component probability SD exposes disagreement without a coverage claim.

## 9. Independent test results

A combined framework need not beat its components

- Ensemble ROC-AUC 0.811; average precision 0.671.
- At threshold 0.33: accuracy 71.4%, recall 70.4%, F1 0.633.
- Specificity 72.0%; precision 57.6%.
- Gradient boosting test ROC-AUC 0.822 exceeded ensemble 0.811.
- Common 0.50 and validation-threshold comparisons are both saved.

## 10. Calibration and statistical evidence

Measure ranking and probability error separately

- Raw Brier 0.1709 → calibrated Brier 0.1746: a worsening.
- Ten-bin ECE improved, illustrating metric disagreement on a small sample.
- Ensemble ROC-AUC conditional bootstrap 95% CI: 0.736–0.876.
- Comparator fixed on training CV: ANN; McNemar at 0.50 gives p=1.0.
- Paired AUC difference is small; no claim of clinical superiority.

## 11. Explainability and what-if

Explain the actual deployed model

- Exact interventional Shapley: 256 coalitions × 24 training background rows.
- Baseline + eight contributions reconstructs the prediction.
- Local examples plus a clearly limited 12-case summary and permutation importance.
- Attributions and hypothetical changes are model associations, not causes.

## 12. Live application demonstration

Validated inputs · probabilities · explanations · scenarios

- Open localhost, load example, and generate prediction + explanation.
- Compare the five learners and explain disagreement.
- Try a missing insulin measurement and inspect the imputation note.
- Change one Glucose or BMI measurement and compare the scenario.
- Show validation for age outside dataset scope and extrapolation notes.

## 13. Limitations and conclusion

What the evidence supports

- Small historical single-population dataset; substantial missingness.
- Shared training CV makes best CV estimates optimistic; holdout stays separate.
- No external validation, prospective outcome validation or clinical approval.
- Calibration and feature engineering did not uniformly improve results.
- Delivered: reproducible models, measured trade-offs and explained local predictions.

## 14. References and discussion

Read primary sources and answer in your own words

- Pima preserved metadata: github.com/jbrownlee/Datasets.
- Tasin et al. (2023), Healthcare Technology Letters, DOI 10.1049/htl2.12039.
- Hasan et al. (2024), Information 16(1), DOI 10.3390/info16010007.
- Guo et al. (2017), ICML / PMLR 70:1321–1330.
- Lundberg & Lee (2017), NeurIPS 30.
- Full links and reading questions: docs/literature_review.md.
