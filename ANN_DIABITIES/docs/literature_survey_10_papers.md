# Literature survey and research gap

Ten papers verified against publisher, proceedings or author sources on 8 October 2026. The first five are diabetes studies; the remaining five support model, calibration and explanation methods.

| No. | Paper | Approach | Connection to this project |
|---|---|---|---|---|
| 1 | Sisodia & Sisodia (2018). [Prediction of Diabetes using Classification Algorithms](https://doi.org/10.1016/j.procs.2018.05.122) | DT, SVM and Naive Bayes | Compare multiple baseline learners |
| 2 | Zou et al. (2018). [Predicting Diabetes Mellitus With Machine Learning Techniques](https://doi.org/10.3389/fgene.2018.00515) | RF, DT, neural nets; PCA / mRMR | Test feature selection and ANN |
| 3 | Sneha & Gangil (2019). [Analysis of diabetes mellitus for early prediction using optimal features selection](https://doi.org/10.1186/s40537-019-0175-6) | Feature selection; classifier comparison | Measure feature-selection benefit |
| 4 | Tasin et al. (2023). [Diabetes prediction using machine learning and explainable AI techniques](https://doi.org/10.1049/htl2.12039) | ML ensembles, SHAP / LIME, app | Explain the deployed classifier |
| 5 | Hasan et al. (2025). [Towards Transparent Diabetes Prediction: Combining AutoML and Explainable AI for Improved Clinical Insights](https://doi.org/10.3390/info16010007) | AutoGluon, XAI, Streamlit interface | Compare models and what-if outputs |
| 6 | Guo et al. (2017). [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) | Confidence calibration; temperature scaling | Measure Brier and ECE separately |
| 7 | Lundberg & Lee (2017). [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) | SHAP additive feature attribution | Explain full pipeline; check additivity |
| 8 | Breiman (2001). [Random Forests](https://doi.org/10.1023/A:1010933404324) | Randomized ensembles of decision trees | Tree-ensemble comparison baseline |
| 9 | Friedman (2001). [Greedy function approximation: A gradient boosting machine](https://doi.org/10.1214/aos/1013203451) | Stagewise additive / gradient boosting | Boosting comparison baseline |
| 10 | Cortes & Vapnik (1995). [Support-vector networks](https://doi.org/10.1007/BF00994018) | Margin-based kernel classification | Nonlinear SVM comparison baseline |

## Research gap addressed

This project tests whether feature selection, equal-weight ensembling and sigmoid calibration improve discrimination and probability quality under a fixed train/validation/test protocol. It also explains the entire deployed pipeline. These are established methods; the contribution is reproducible integration, controlled comparisons and honest reporting of trade-offs.

The experiment retains negative findings: the engineered representation did not win, gradient boosting had higher test AUC than the ensemble, and calibration worsened Brier score despite reducing binned ECE. These observations concern this project and are not universal criticisms of the papers. External validation remains an unresolved limitation.

Paper titles on the slide link to sources. Full titles and links also appear in the slide speaker notes. Study metrics should not be ranked against this project without comparing populations, partitions and preprocessing.

Publication-year notes: Hasan et al. is cited as Information 2025, 16(1), 7 (published online 26 December 2024); Tasin et al. is Healthcare Technology Letters 2023, 10, 1–10 (online December 2022). Guo et al. examine temperature scaling, while this project uses sigmoid calibration.
