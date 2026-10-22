# Learn machine learning through your diabetes project

**A beginner's guide, project history, code walkthrough, and complete file inventory in one file.**

Read this as a learning document. It is AI-assisted educational material, not the student-authored report required by your rubric. The file explains what was built, the reasons for the main design choices, and what the saved evidence supports. It does not reproduce private internal deliberations.

You do not need to know machine learning before starting. You do need to work through examples and try the exercises. This guide teaches the foundations and every major concept implemented here. Machine learning is a much larger field; finishing one guide is a starting foundation, not mastery of all its topics.

All numerical project results below come from your saved experiment unless explicitly labeled as an illustration. Code blocks labeled **Project excerpt** are extracted from the actual named files, with indentation preserved. They may be fragments of a function and are not necessarily runnable alone. Blocks labeled **Practice** are separate teaching examples, not claims that those lines exist in the application.

## Contents

1. [What was built and how the work progressed](#1-what-was-built-and-how-the-work-progressed)
2. [The complete system in one diagram](#2-the-complete-system-in-one-diagram)
3. [Python and data basics you need](#3-python-and-data-basics-you-need)
4. [What machine learning means](#4-what-machine-learning-means)
5. [Understanding the dataset and its target](#5-understanding-the-dataset-and-its-target)
6. [Loading and validating data](#6-loading-and-validating-data)
7. [Training, validation, test data, and leakage](#7-training-validation-test-data-and-leakage)
8. [Exploratory data analysis and statistics](#8-exploratory-data-analysis-and-statistics)
9. [Missing values, imputation, indicators, and outliers](#9-missing-values-imputation-indicators-and-outliers)
10. [Scaling, pipelines, and the fit/transform contract](#10-scaling-pipelines-and-the-fittransform-contract)
11. [Feature engineering, selection, and ablation](#11-feature-engineering-selection-and-ablation)
12. [Models, parameters, loss, and optimization](#12-models-parameters-loss-and-optimization)
13. [Logistic regression](#13-logistic-regression)
14. [Decision trees and random forests](#14-decision-trees-and-random-forests)
15. [Gradient boosting](#15-gradient-boosting)
16. [Support vector machines](#16-support-vector-machines)
17. [Neural networks from the beginning](#17-neural-networks-from-the-beginning)
18. [Cross-validation and hyperparameter search](#18-cross-validation-and-hyperparameter-search)
19. [Ensemble learning and probability calibration](#19-ensemble-learning-and-probability-calibration)
20. [Thresholds, confusion matrices, and every metric](#20-thresholds-confusion-matrices-and-every-metric)
21. [Actual results and how to interpret them](#21-actual-results-and-how-to-interpret-them)
22. [Bootstrap intervals and statistical comparisons](#22-bootstrap-intervals-and-statistical-comparisons)
23. [Explainability: permutation importance and Shapley values](#23-explainability-permutation-importance-and-shapley-values)
24. [Model disagreement and what-if analysis](#24-model-disagreement-and-what-if-analysis)
25. [Saving the model and reproducing an experiment](#25-saving-the-model-and-reproducing-an-experiment)
26. [How the website talks to the model](#26-how-the-website-talks-to-the-model)
27. [Testing, notebooks, and generated documents](#27-testing-notebooks-and-generated-documents)
28. [Complete file inventory](#28-complete-file-inventory)
29. [Running and inspecting the project yourself](#29-running-and-inspecting-the-project-yourself)
30. [Practice exercises with explanations](#30-practice-exercises-with-explanations)
31. [Troubleshooting and common mistakes](#31-troubleshooting-and-common-mistakes)
32. [Glossary and a learning schedule](#32-glossary-and-a-learning-schedule)
33. [Further topics and primary references](#33-further-topics-and-primary-references)

## 1. What was built and how the work progressed

At the start, this folder contained `EvaluationRubrics_DiabetesPrediction.pdf`. The PDF was supplied by you; I did not author it. There was no existing model, dataset, application, or report to extend.

The following is a high-level development history reconstructed from the work performed and saved artifacts. It describes actions and design reasons, rather than hidden internal reasoning.

| Stage | Action taken | Reason / evidence |
|---|---|---|
| 1 | Read the PDF and its 13 evaluation components | Determine the required deliverables instead of building only an ANN |
| 2 | Inspect the workspace and available Python libraries | Use installed scientific tools; avoid requiring a heavy neural-network framework for a small dataset |
| 3 | Ask whether you had a required dataset and ask for student/team details | Those affect the scope and personalization; no replacement dataset or identity details were provided in the conversation |
| 4 | Proceed with the public Pima dataset and record its source | It matches an eight-feature diabetes classification study; its population limits must be understood |
| 5 | Research relevant papers and primary library documentation | Check the literature and explain established methods without inventing global novelty |
| 6 | Download the real CSV from a public mirror | Train on real historical records, rather than fabricated examples |
| 7 | Build the package structure and data checks | Separate loading, preprocessing, training, evaluation, explanation, prediction, and serving |
| 8 | Design a stratified train/validation/test protocol before evaluation | Keep model decisions separate from the final performance estimate |
| 9 | Implement missing-value policies and fold-local preprocessing | Prevent imputation, scaling, and supervised feature selection from seeing held-out rows |
| 10 | Generate training-only quality summaries and EDA plots | Understand missingness, distributions, imbalance, correlations, and extreme values |
| 11 | Compare original, engineered, and selected feature representations | Test whether the proposed transformations help rather than assuming that they do |
| 12 | Tune five model families, including a real MLP ANN | Meet the comparison requirement and test simpler alternatives |
| 13 | Fit an equal-weight soft ensemble and out-of-fold sigmoid calibration | Integrate probability estimation and evaluate the effect of calibration |
| 14 | Choose thresholds using validation F1 and freeze the deployment bundle | Separate probability fitting, decision selection, and final testing |
| 15 | Evaluate the untouched test partition | Save genuine discrimination, classification, and probability-quality results |
| 16 | Compute intervals, paired comparison, importance, explanations, disagreement, and scenarios | Extend the evaluation beyond a single accuracy number |
| 17 | Build the local Python HTTP server and HTML/CSS/JavaScript interface | Make the saved model usable without a separate large web framework |
| 18 | Create the notebook, supporting documentation, slides, evidence PDF, and viva prompts | Connect the implementation to the remaining rubric components |
| 19 | Run regression checks, execute the notebook, and check the live server | Verify meaningful behavior rather than merely creating files |
| 20 | Package the project into a ZIP and record file checksums | Make the deliverables portable and inspectable |
| 21 | Simplify the frontend after your feedback | Replace decorative branding with a plain form, results, and expandable charts/explanations |
| 22 | Remove the introductory paragraph you asked to remove | Apply your requested frontend text change; the technical model and recorded scope did not change |
| 23 | Create this guide | Explain the existing project as a beginner learning path |

A server restart interrupted the first incomplete training attempt. The saved code and initial outputs remained available; the full study was restarted and completed. The recorded completed run took **68.42 seconds in that environment**. That is the training/evaluation duration, not the time needed to develop the whole project and documents.

The environment initially blocked the dataset download and later blocked local sockets needed by the web server and Jupyter. Those actions were rerun with the required approvals. They were environment restrictions, not evidence of a faulty classifier.

The project uses **scikit-learn's `MLPClassifier`**, not TensorFlow or PyTorch. An ANN does not require a particular framework. NumPy, pandas, SciPy, scikit-learn, Matplotlib, and joblib support the implementation. The HTTP server and test framework come from Python's standard library.

## 2. The complete system in one diagram

```text
Original rubric
      |
      v
Real CSV -> validate schema -> split rows with seed 42
                                  |
                   +--------------+---------------+
                   |              |               |
                TRAIN          VALIDATION        TEST
               460 rows         154 rows        154 rows
                   |              |               |
          Training-only EDA       |               |
          Feature comparisons     |               |
          5-fold model search     |               |
                   |              |               |
       zero handling -> median -> scaling -> selection -> model
                   |
       Logistic / Forest / Boosting / SVM / ANN
                   |
       mean probabilities -> sigmoid calibration
                   |              |
                   +---- choose threshold by validation F1
                                  |
                       save/freeze fitted bundle
                                  |
                      evaluate final test once <---+
                                  |
                      saved metrics and figures
                                  |
                     application loads the bundle
                                  |
                user inputs -> validation -> prediction
                                  |
                   explanations + what-if + disagreement
```

**Training is learning from labeled records. Inference is using what was already learned to process a new input.** Opening the website performs inference; it does not retrain the classifier.

There are three different kinds of output: a model artifact used for inference; experimental evidence used for evaluation; and documentation used for learning/presentation. A PDF or an accuracy table is not itself the fitted model.

## 3. Python and data basics you need

Python is the programming language. A **library** is reusable code installed for Python. `import` gives your file access to that library. `numpy` handles numeric arrays; `pandas` handles tables; `sklearn` supplies ML estimators and tools; `matplotlib` creates plots; `joblib` saves Python objects.

| Syntax or object | Meaning here | Example |
|---|---|---|
| Variable | A name referring to a value | `threshold = 0.33` |
| List | Ordered sequence | `['Glucose', 'BMI']` |
| Dictionary | Key/value mapping | `{'Age': 32}` |
| Function | Reusable operation | `load_data()` |
| Class | A definition for objects with state and methods | `FeatureBuilder` |
| Object | An instance of a class | `StandardScaler()` |
| Method | Function attached to an object | `model.fit(X, y)` |
| Argument | Value passed to a function/method | `k=5` |
| Boolean | `True` or `False` | `early_stopping=True` |
| `None` | Python's absence-of-value object | An optional argument |
| `np.nan` | Numeric missing-value marker | An unknown insulin measurement |
| DataFrame | Labeled rectangular table | `df` |
| Series | One labeled column/sequence | `df['Outcome']` |
| ndarray | NumPy numeric array | Predicted probabilities |
| `Path` | Filesystem path object | `ROOT / 'reports/results.json'` |

A DataFrame has **rows** and **columns**. A row is an observation; a column is a variable. `df.shape` reports `(number_of_rows, number_of_columns)`. Here it is `(768, 9)` because there are eight input columns and one target column.

`df[FEATURES]` selects several columns and returns a table. `df[TARGET]` selects one column and returns a Series. `df.iloc[0]` selects the first row by position; `df.loc[row_id]` uses an index label. Those can differ after splitting, because the split retains original row indexes.

**Practice — inspect data without fitting anything. Run from the project root:**

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / 'src'))
from diabetes.data import load_data, FEATURES, TARGET

df = load_data()
print(df.shape)                   # (768, 9)
print(df.head())                  # first five records
print(df[TARGET].value_counts())  # 500 zeros, 268 ones
X = df[FEATURES]
y = df[TARGET]
print(X.shape, y.shape)           # (768, 8), (768,)
```

When you see `predict_proba(... )[:, 1]`, read it as: return all rows (`:`), but only column 1. The current class order is `[0, 1]`, so column 1 holds the positive-class probability. A different class ordering must be checked rather than guessed.

The trailing underscore in names such as `statistics_`, `coefs_`, and `best_estimator_` conventionally means a value obtained after fitting. It is different from the configuration you supplied before fitting.

## 4. What machine learning means

A manually written rule says exactly how to decide, such as “if a number is above a chosen cutoff, return 1.” A learned model estimates a relationship from examples. We give it input measurements and known labels, and it adjusts its parameters to make useful predictions on new examples.

For this project:

```text
X = measurement inputs
 y = known Outcome labels
 f = fitted model
 p = f(X), estimated probability of label 1
```

An individual input vector is often written `x`. The training table `X` contains many such vectors. Training aims to learn a function that generalizes beyond the rows it saw.

### Types of learning

| Type | What the data provides | Example | Used here? |
|---|---|---|---|
| Supervised classification | Inputs and category labels | Predict Outcome 0 or 1 | Yes |
| Supervised regression | Inputs and a numerical target | Predict a house price | No |
| Unsupervised learning | Inputs without target labels | Group similar records | No |
| Reinforcement learning | Actions and rewards from interactions | Learn a game-playing policy | No |
| Semi-supervised learning | A mixture of labeled and unlabeled examples | Learn from partially labeled records | No |

**Binary classification** means two possible labels, here `0` and `1`. “Positive” is a statistical label; it does not mean “good.” A positive prediction means the model predicts label 1.

**Artificial intelligence** is the broader category. **Machine learning** is an approach within it. **Neural networks** are one class of ML models. **Deep learning** generally refers to neural networks with substantial depth; the selected one-hidden-layer MLP here is a small neural network, not a modern large deep model.

This is a tabular ML project. It does not contain an LLM, a chatbot, a language model, image recognition, embeddings, or a GPU training system.

### Learning is not understanding causes

A model might find that certain measurements are associated with label 1. That does not prove that changing one measurement will cause an outcome to change. Dataset labels may also be noisy or depend on collection practices. A statistically accurate classifier can still be unsuitable for another population or task.

## 5. Understanding the dataset and its target

The raw file is `data/pima_raw.csv`. It is a real historical public benchmark downloaded from the [recorded CSV mirror](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv). The [preserved source metadata](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names) describes the features and source population.

| Column | Beginner explanation | Unit/type | Missing-value treatment |
|---|---|---|---|
| `Pregnancies` | Number of times pregnant | Integer count | Zero is valid |
| `Glucose` | Plasma glucose measured after a two-hour OGTT | mg/dL | Zero becomes missing |
| `BloodPressure` | Diastolic blood pressure | mmHg | Zero becomes missing |
| `SkinThickness` | Triceps skinfold thickness | mm | Zero becomes missing |
| `Insulin` | Two-hour serum insulin | µU/mL | Zero becomes missing |
| `BMI` | Weight divided by squared height | kg/m² | Zero becomes missing |
| `DiabetesPedigreeFunction` | Dataset-specific pedigree score | Numeric score | Not a yes/no family-history input |
| `Age` | Age | Years | Form requires integer age ≥21 |
| `Outcome` | Historical binary outcome label | 0 or 1 | Target, never an input |

The CSV has no header. The loader supplies column names in their known order. For a compatible file that includes a header, the same loader accepts the standard names.

Of 768 rows, 500 are label 0 and 268 are label 1. The positive proportion is `268 / 768 ≈ 34.90%`. This is a **class imbalance**: the classes occur at different frequencies. A model that always predicts 0 would get about 65.10% accuracy on the full dataset without understanding any measurements.

That is why accuracy alone is insufficient.

The existing label is not regenerated from a predictor. Defining `Outcome = Glucose > some_cutoff` and then using Glucose to predict it would create a circular task that differs from this study.

The project does not have an external hospital cohort, follow-up timeline, or prospective collection. Treat its output as a classification of this dataset's label rather than a proven future-onset forecast. The dataset's population constraints are facts about the experiment, even though the introductory paragraph was removed from the frontend at your request.

## 6. Loading and validating data

**Project excerpt — [src/diabetes/data.py](../src/diabetes/data.py), starting at line 7:**

```python
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TARGET = "Outcome"
SEED = 42
```

`FEATURES` fixes input names and order. `TARGET` fixes the label name. `ZERO_MISSING` identifies the special zero sentinels. `SEED` controls reproducible random choices.

**Project excerpt — [src/diabetes/data.py](../src/diabetes/data.py), starting at line 13:**

```python
def load_data(path=None):
    path = Path(path or ROOT / "data/pima_raw.csv")
    first = path.read_text().splitlines()[0]
    df = pd.read_csv(path) if "Outcome" in first else pd.read_csv(path, header=None, names=FEATURES + [TARGET])
    if set(df.columns) != set(FEATURES + [TARGET]):
        raise ValueError("Dataset must contain exactly the eight Pima features and Outcome.")
    df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="raise")
    if not df[TARGET].isin([0, 1]).all() or df[TARGET].nunique() != 2:
        raise ValueError("Outcome must contain both binary classes 0 and 1 without missing labels.")
    if np.isinf(df.to_numpy()).any() or (df[FEATURES] < 0).any().any():
        raise ValueError("Infinite and negative measurements are invalid.")
    if df.duplicated(subset=FEATURES).any():
        raise ValueError("Duplicate feature rows require an explicit group-aware split before training.")
    return df
```

Read the loader in this order:

1. Resolve the input file path.
2. Check the first line to decide whether it has a header.
3. Read the CSV into a DataFrame.
4. Reject the wrong column set.
5. Put the columns in the required order and convert them to numbers.
6. Require both binary classes and no missing target labels.
7. Reject infinity and negative measurements.
8. Reject duplicate input rows before splitting.
9. Return the validated table.

`errors='raise'` means malformed text should cause a visible error, not silently become a missing value. A **schema** is the expected structure: column names, types, and target meaning.

Two repeated patient records could land in both training and test data, producing an unrealistically easy test. Here identical feature rows cause an error. In a real dataset with patient IDs or repeated visits, a **group-aware split** would keep all records for a patient in the same partition. The project does not have patient IDs, so it cannot verify that broader kind of independence.

Validation of the dataset and validation of a web form are related but separate. Dataset checks protect training. Form checks protect inference.

## 7. Training, validation, test data, and leakage

Imagine studying from practice questions and then sitting an exam. Memorizing the exam answers before the exam would make the score misleading. **Data leakage** is information from held-out evaluation data entering a fitting or selection decision.

The project uses three partitions:

| Partition | Rows | Positive labels | Allowed use |
|---|---:|---:|---|
| train | 460 | 160 | EDA, representation/model fitting and calibration |
| validation | 154 | 54 | Select decision thresholds |
| test | 154 | 54 | Final assessment and interpretation after freezing |

**Project excerpt — [src/diabetes/train.py](../src/diabetes/train.py), starting at line 121:**

```python
    train_val, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.25, stratify=train_val[TARGET], random_state=SEED)
    X, y = train[FEATURES], train[TARGET]
    Xv, yv, Xt, yt = val[FEATURES], val[TARGET], test[FEATURES], test[TARGET]
```

The first split sets aside 20% for test. The second sets aside 25% of the remaining 80%: `0.25 × 0.80 = 0.20`. That leaves roughly 60% train, 20% validation, and 20% test. Integer rounding produces 460, 154, and 154 rows.

`stratify=df[TARGET]` approximately preserves class proportions. `random_state=42` makes that particular random shuffle repeatable for the same data and software behavior. The number 42 has no scientific advantage; consistency matters.

The original indexes are saved to `artifacts/splits.json`. The same row should never be in two partitions.

### What must be learned only from training data?

Medians for imputation, means/standard deviations for scaling, feature-selection scores, classifier parameters, and calibration mappings are fitted quantities. They must not be learned from the final test partition.

A leaky approach fits a scaler on all 768 rows, then splits the scaled output. A correct approach splits first and fits the scaler on training data. Inside cross-validation, the same rule applies again to each fold.

The main validation set chooses decision thresholds. Test results do not choose a new model, a new feature set, or a new seed. Choosing a better random split after reading the test accuracy is also a form of test-based selection.

The deployed model is intentionally not refitted on training plus validation after threshold selection. Refitting changes its probabilities, which can make the previously chosen threshold inappropriate. A different retraining protocol would need a fresh calibration/threshold strategy.

The implementation keeps the external test separate, but its best CV score is not an unbiased estimate: the same training CV supports several selections. That limitation is discussed in Section 18. Prevention of direct test leakage does not solve every form of selection optimism. [Official discussion of leakage and consistent preprocessing](https://scikit-learn.org/stable/common_pitfalls.html).

## 8. Exploratory data analysis and statistics

**EDA** means exploring the data before deciding what it tells you. We inspect training data to understand values, missingness, class balance, associations, and extreme measurements.

### The basic statistics

- **Mean**: sum of values divided by their count. Sensitive to extreme values.
- **Median**: middle value after sorting. Less affected by extreme values.
- **Variance**: average squared distance from the mean, using a specified denominator convention.
- **Standard deviation**: square root of variance; describes spread in the original units.
- **Quantile**: a point below which a specified proportion of observations lies.
- **Q1 / Q3**: the 25th and 75th percentiles.
- **IQR**: `Q3 − Q1`; spread of the middle half of the values.

**Illustration:** `[10, 11, 12, 13, 100]` has mean 29.2 and median 12. The final value substantially pulls the mean upward. This helps explain the choice of median imputation.

**Project excerpt — [src/diabetes/train.py](../src/diabetes/train.py), starting at line 47:**

```python
    for f in FEATURES:
        s = clean[f]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        quality.append({"feature": f, "missing_or_zero": int(s.isna().sum()),
                        "missing_pct": float(s.isna().mean() * 100),
                        "zero_sentinel_count": int((train[f] == 0).sum()) if f in ZERO_MISSING else 0,
                        "iqr_outliers": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
                        "minimum": float(s.min()), "maximum": float(s.max()), "median": float(s.median())})
```

This code counts missing values and IQR flags per training feature. It writes a table you can inspect instead of judging every data-quality issue by a chart.

### Reading the plots

| Plot | What it displays | What to ask |
|---|---|---|
| Histogram | Counts within value ranges | Is the distribution skewed? Do classes overlap? |
| Boxplot | Median, quartiles, and flagged extremes | Are high values plausible or measurement errors? |
| Class-count bar chart | Frequency of each label | How imbalanced is the dataset? |
| Missingness bar chart | Missing/invalid-zero percentage | Which measurements are often unavailable? |
| Correlation heatmap | Pairwise linear association | Are relationships linear, weak, or redundant? |
| Scatter plot | One variable versus another | Is there a pattern or a cluster of errors? |

**Pearson correlation** lies between −1 and +1. Positive means the variables tend to increase together; negative means one tends to increase when the other decreases. A value near zero does not prove no relationship; a nonlinear relationship can have little linear correlation.

In this training partition, Glucose's correlation with Outcome is approximately **0.501**, and BMI's is approximately **0.337**. These are measured associations, not causal effects. The code treats zero sentinels as missing before correlation. pandas uses available pairs, so different correlations can involve different subsets of records.

`reports/training_descriptive_statistics.csv` contains summaries. `reports/training_correlations.csv` contains the numeric heatmap values. The file `reports/figures/correlation.png` is the visualization of those relationships.

Practice question: If high BMI and label 1 appear together, what alternative explanations could exist? Possible answers include confounding, selection, measurement differences, or multiple related causes. An observational classifier cannot settle those causal questions by itself.

## 9. Missing values, imputation, indicators, and outliers

### Missing values

Some zero measurements in the CSV are used as placeholders for unavailable values. A zero pregnancy count is different: it is meaningful. Replacing every zero in the entire table would corrupt valid data and the label 0 itself.

**Project excerpt — [src/diabetes/preprocessing.py](../src/diabetes/preprocessing.py), starting at line 31:**

```python
    def transform(self, X):
        check_is_fitted(self)
        df = X[FEATURES].copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=FEATURES)
        df = df.astype(float)
        df[ZERO_MISSING] = df[ZERO_MISSING].replace(0, np.nan)
        if self.engineered:
            df["LogInsulin"] = np.log1p(df["Insulin"])
            df["LogPedigree"] = np.log1p(df["DiabetesPedigreeFunction"])
            df["GlucoseBMIInteraction"] = df["Glucose"] * df["BMI"] / 100
        return df
```

`np.nan` marks missing numeric information. `copy()` helps avoid changing the original input table. The conditional engineered columns will be explained in Section 11.

### Median imputation

**Imputation** fills unavailable values using a rule learned from observed training values. For insulin measurements `[80, 100, missing, 200]`, the observed median is 100, so the missing value would become 100.

In this project's training partition, zero/missing rates include:

| Training feature | Zero/missing count | Percentage |
| --- | --- | --- |
| Pregnancies | 0 | 0.00% |
| Glucose | 4 | 0.87% |
| BloodPressure | 18 | 3.91% |
| SkinThickness | 131 | 28.48% |
| Insulin | 215 | 46.74% |
| BMI | 6 | 1.30% |
| DiabetesPedigreeFunction | 0 | 0.00% |
| Age | 0 | 0.00% |

Nearly half the training insulin measurements are missing. Imputation is therefore a modeling assumption with a substantial limitation, not a recovery of the true laboratory results.

### Missingness indicators

`SimpleImputer(..., add_indicator=True)` can append a 0/1 column indicating that a feature was missing. That lets a model distinguish an observed median value from a median inserted because the measurement was unavailable. Indicators are created for features with missing values at fitting; they are not a promise of a separate indicator for every possible future missing feature. They can also be removed by subsequent feature selection.

### Why the median is fitted inside each fold

Suppose the training median is 100 but the test measurements would move the combined median to 120. Using 120 to fill training gaps incorporates information from the test distribution. The median must be fitted from training rows only and reused unchanged for inference.

### Outliers

A common exploratory rule flags values below `Q1 − 1.5×IQR` or above `Q3 + 1.5×IQR`. That is a statistical flag, not a diagnosis of an invalid measurement.

The implementation **retains** those values. Dropping all extreme records might remove exactly the high-risk cases that matter. The project counts flags, examines distributions, and documents the decision rather than introducing an unexplained exclusion rule.

The standard scaler used here is still sensitive to extreme values. A robust scaler would be a possible alternative experiment, but it is not part of the completed comparison and has not been silently substituted.

## 10. Scaling, pipelines, and the fit/transform contract

### Standard scaling

Glucose values can be in the hundreds while pedigree scores can be below 1. Models based on distances, gradients, or penalties can be affected by these different numeric scales.

A standard scaler computes:

```text
scaled_value = (value − training_mean) / training_standard_deviation
```

**Illustration:** if the training mean is 120 and training standard deviation is 30, then Glucose 150 becomes `(150−120)/30 = 1`. Glucose 90 becomes −1. These are distances in standard-deviation units, not new laboratory units.

The mean and standard deviation are learned from the training data reaching that scaler. Here the scaler comes after imputation, so it sees the filled training values. This is not normalization to a 0–1 range. StandardScaler uses a population-style variance convention (`ddof=0`) for its fitted statistics. [Official scaler documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

Tree-based splits typically do not require standard scaling; an ordered linear rescaling does not usually change which threshold separates records. The shared pipeline keeps preprocessing consistent across candidate families, while scaling is particularly useful for the ANN, logistic regression, and SVM.

### Pipeline

**Project excerpt — [src/diabetes/preprocessing.py](../src/diabetes/preprocessing.py), starting at line 47:**

```python
def pipeline(model, engineered=False, k="all"):
    return Pipeline([
        ("features", FeatureBuilder(engineered)),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("model", model),
    ])
```

Read from top to bottom:

1. `features`: convert documented zero sentinels and optionally create predictors.
2. `imputer`: learn medians and append missingness indicators.
3. `scaler`: learn means and scales.
4. `select`: retain all or the strongest `k` columns.
5. `model`: learn the classifier.

A pipeline is one object that ensures the same transformations happen during fitting and prediction. When the whole pipeline is passed to CV, each cloned pipeline learns its preprocessing from that fold's training subset.

### `fit`, `transform`, `fit_transform`, and `predict`

| Method | Purpose | Does it learn fitted state? |
|---|---|---|
| `fit(X, y)` | Learn from the supplied training data | Usually yes |
| `transform(X)` | Apply an already specified/fitted transformation | No refitting |
| `fit_transform(X, y)` | Fit, then transform that training data | Yes, once |
| `predict(X)` | Return labels using the estimator's rule | No |
| `predict_proba(X)` | Return probabilities for each class | No |

Our custom `FeatureBuilder.fit` validates input order and sets fitted attributes; it does not learn a population median. `SimpleImputer.fit` does learn medians. Knowing which steps learn statistics is central to leakage prevention.

`BaseEstimator` makes the custom class compatible with scikit-learn configuration/cloning. `TransformerMixin` supplies transformation-related behavior. `check_is_fitted` prevents transforming with an object that has not passed through fitting. Names such as `n_features_in_` record the fitted object's input shape.

`k='all'` retains every column. `k=5` selects five transformed columns; this includes the possibility of selecting an indicator rather than an original column.

## 11. Feature engineering, selection, and ablation

A **feature** is an input variable supplied to a model. **Feature engineering** creates alternative inputs from existing information. **Feature selection** removes inputs according to a criterion. Those are different operations.

The experimental engineered features are:

```text
LogInsulin = ln(1 + Insulin)
LogPedigree = ln(1 + DiabetesPedigreeFunction)
GlucoseBMIInteraction = Glucose × BMI / 100
```

The corresponding project lines are included in `FeatureBuilder.transform` in Section 9. `np.log1p(x)` calculates `ln(1+x)` accurately. A log transform compresses large positive numbers. An interaction lets a model use a joint relationship: the combined value changes when either input changes. The division by 100 only adjusts its numeric scale; it does not create a clinically validated score.

The transforms are hypotheses to test, not evidence that larger feature counts improve performance.

### ANOVA feature selection

`SelectKBest(f_classif, k=5)` scores each column independently using an ANOVA F statistic: roughly, variation between class means compared with variation within each class. A high score suggests that the feature's class means are well separated relative to within-class spread. It does not prove causation or guarantee useful interactions.

The score is **univariate**: each input is considered separately. Two individually weak variables could be important together, and a purely univariate selector can miss that. A top-five choice is relative to this data and procedure, not a universal ranking of diabetes risk factors. [Official `f_classif` definition](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html).

The selector uses `y`, so it must be fitted only on the training part of a CV fold.

### Ablation: test the contribution of a component

An **ablation study** compares versions of a system with specific components added, removed, or replaced. Here fixed logistic and ANN learners compare three representations using the same folds.

**Project excerpt — [src/diabetes/train.py](../src/diabetes/train.py), starting at line 132:**

```python
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
```

Actual mean training-CV ROC-AUC:

| Learner | Representation | Mean CV ROC-AUC | Fold SD |
| --- | --- | --- | --- |
| LogisticRegression | original | 0.8262 | 0.0209 |
| LogisticRegression | engineered | 0.8254 | 0.0194 |
| LogisticRegression | selected_5 | 0.8205 | 0.0178 |
| ANN | original | 0.7946 | 0.0272 |
| ANN | engineered | 0.7558 | 0.1154 |
| ANN | selected_5 | 0.8082 | 0.0371 |

The representation is chosen by averaging the two learners' mean AUCs for each representation. The approximate averages are 0.8104 for original, 0.7906 for engineered, and 0.8144 for selected-five. Selected-five wins that rule.

Notice that logistic regression alone does better with the original representation. The rule combines both learners, so a shared representation need not be best for each individual model. The engineered ANN also has substantial variation across folds. The experiment does not establish that engineering is always harmful; it says these exploratory features did not win this particular comparison.

When fitted on all 460 training rows, the deployed pipelines select **Glucose, SkinThickness, Insulin, BMI, and Age**. This was checked from the saved estimator, not inferred from the form labels. CV folds may select different columns because their training samples differ.

The eight raw form fields still preserve the documented schema. The selected classifier receives five transformed inputs after preprocessing. In this fitted version, changes to the three excluded raw inputs do not change predictions, and their exact Shapley contributions are zero. That does not mean they are universally irrelevant to health.

## 12. Models, parameters, loss, and optimization

A **model family** defines a kind of relationship that can be learned, such as a linear boundary, decision tree, or neural network. A **fitted model** is one particular trained object with particular learned parameters.

| Item | Learned or chosen? | Example |
|---|---|---|
| Parameter | Learned during fitting | Neural-network weight or logistic coefficient |
| Hyperparameter | Chosen before a fitting run | Number of hidden neurons or forest depth |
| Preprocessing statistic | Learned from training data | Median, mean, standard deviation |
| Decision threshold | Chosen for the decision rule | 0.33 selected using validation F1 |
| Evaluation metric | Calculated after predictions | Test recall or ROC-AUC |

A **loss function** assigns a penalty to predictions. Training adjusts parameters to reduce loss. An **evaluation metric** helps assess the result. A loss and a metric need not be the same: the ANN trains on cross-entropy while the parameter search ranks candidates by ROC-AUC.

### Binary cross-entropy / log loss

For label `y` and predicted positive probability `p`, the single-row loss is:

```text
loss = −[y × ln(p) + (1−y) × ln(1−p)]
```

If the true label is 1:

```text
p = 0.9 -> loss = −ln(0.9) ≈ 0.105
p = 0.5 -> loss = −ln(0.5) ≈ 0.693
p = 0.1 -> loss = −ln(0.1) ≈ 2.303
```

A confident wrong prediction is penalized much more than an uncertain prediction. The example uses natural logarithms. Actual library calculations guard against undefined logs at probability 0 or 1.

### Optimization

An **optimizer** is the procedure for changing parameters to reduce loss. A **gradient** tells us how the loss changes with each parameter. A basic gradient-descent update is:

```text
new_weight = old_weight − learning_rate × gradient
```

The learning rate controls step size. Very small steps can be slow. Very large steps can overshoot or destabilize learning. A more complex optimizer such as Adam adapts updates using information about past gradients.

### Overfitting and underfitting

**Underfitting** means the learned relationship is too restrictive, too weakly trained, or poorly represented to capture useful patterns. **Overfitting** means performance on the fitting data is much better than performance on new examples because the model follows noise or peculiarities of the sample.

More complexity can reduce training loss while harming generalization. The **bias–variance trade-off** is one way of describing this tension: restrictive models can have high systematic error; flexible models can change substantially across samples.

**Regularization** discourages excessive complexity. Examples here include L2 penalties, smaller network architectures, forest depth/minimum leaf size, and early stopping. These tools reduce risk; none guarantees absence of overfitting.

The study implements the following candidate model definitions:

**Project excerpt — [src/diabetes/train.py](../src/diabetes/train.py), starting at line 95:**

```python
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
```

The tuple for each dictionary entry contains the estimator and a dictionary of hyperparameter choices. Entries such as `model__C` address the `C` parameter of the pipeline step named `model`; the double underscore is scikit-learn's nested-parameter syntax.

## 13. Logistic regression

Despite the word “regression,” `LogisticRegression` here is a classification model.

For inputs `x1, x2, ...`, it computes a weighted sum:

```text
z = w1×x1 + w2×x2 + ... + b
p = sigmoid(z) = 1 / (1 + exp(−z))
```

Weights `w` and bias `b` are learned. The sigmoid maps any real-valued score to a probability between 0 and 1. The resulting log-odds are linear in the supplied features; the probability itself is a nonlinear sigmoid of that score.

**Illustration, not project coefficients:** two scaled inputs are `x1=1` and `x2=0.5`, weights are `0.8` and `0.4`, and bias is −0.3. Then `z=0.8+0.2−0.3=0.7`, and `p≈0.668`. At threshold 0.50, the predicted label is 1.

It is useful as a relatively simple comparison model. Its coefficients can be inspected, but correlated inputs, scaling, penalties, and feature selection affect their interpretation. A coefficient is not automatically a causal effect.

In this implementation, the search changes:

- `C`: inverse regularization strength. Smaller `C` usually means a stronger penalty.
- `class_weight`: either ordinary weighting or `'balanced'`, which gives different loss weights to classes based on their frequency.

The selected logistic settings were `C=0.05`, `class_weight='balanced'`. Class weighting does not create new samples and can change probability behavior as well as classification trade-offs.

## 14. Decision trees and random forests

A **decision tree** learns a sequence of threshold splits. For a simple illustration:

```text
Is feature A > a learned value?
  |-- no: follow left branch
  |-- yes: follow right branch
```

Later branches can test other features. A leaf combines the labels reaching it to form an estimate. The thresholds are learned from training data; the illustration is not a hand-coded diagnostic rule in this project.

The default forest trees use a criterion related to class impurity. For a binary node with positive fraction `q`, Gini impurity is:

```text
Gini = 1 − [q² + (1−q)²]
```

A node containing only one class has impurity 0. A half-and-half node has impurity 0.5. A split seeks a reduction in weighted impurity.

A large unrestricted tree can memorize a small dataset. A **random forest** fits many trees on bootstrap-resampled training data and considers random subsets of input features when splitting. Combining varied trees can reduce the variance of a single tree.

A forest probability averages tree probability estimates; it is not necessarily the fraction of trees whose hard labels equal 1.

This project uses 180 trees. The selected settings include maximum depth 5, minimum leaf size 10, `max_features='sqrt'`, and balanced class weights. Larger leaves and limited depth constrain tree complexity. Bootstrap training inside a forest is distinct from bootstrap **evaluation intervals** in Section 22.

Random forests are not the same as the five-model soft-voting ensemble. The forest is itself an ensemble of trees; the later voter combines that forest with four different model families.

## 15. Gradient boosting

**Boosting** builds models sequentially so later learners help correct what the current combined model gets wrong according to the loss. It differs from a forest's broadly parallel construction of independently randomized trees.

A conceptual update is:

```text
next_score(x) = current_score(x) + learning_rate × new_tree(x)
```

For binary classification, the score is related to log-odds and converted to a probability. A classification boosting method uses gradients of its loss; it is more precise to say it fits negative-gradient structure than simply saying “it fits label residuals.”

`HistGradientBoostingClassifier` groups continuous values into bins to speed split finding. “Hist” refers to histogram-style binning, not an EDA histogram drawn for the report.

The selected settings were 80 maximum boosting iterations, seven maximum leaf nodes, learning rate 0.04, and L2 regularization 1. These are configuration choices, not guaranteed optimal settings for every dataset.

The saved test ROC-AUC for this model was 0.8222, higher than the deployed ensemble's 0.8107. That difference is reported descriptively. The test set was not then used to switch the prespecified research interface to this model.

## 16. Support vector machines

A **support vector machine** seeks a decision boundary with a useful margin between classes. The training points most influential near that boundary are called support vectors.

A linear SVM uses a hyperplane. An RBF-kernel SVM can represent nonlinear relationships through similarity calculations:

```text
RBF_similarity(x, x') = exp(−gamma × squared_distance(x, x'))
```

`C` controls the trade-off between margin/regularization and fitting errors. `gamma` controls how quickly RBF similarity decays with distance. Larger gamma can create very local, complex behavior. Scaling matters because a large-unit feature can otherwise dominate a distance.

The project uses the default RBF kernel and requests `probability=True`. SVM decision scores are not probabilities by themselves; this option performs an additional internal probability-fitting procedure. It is separate from the sigmoid calibration of the final voter.

The selected settings were `C=5`, `gamma=0.03`, and no class weighting. The included run records a deprecation warning about `SVC(probability=True)` in the installed scikit-learn version. The experiment ran successfully; the warning concerns a future API change, not a failed fit. Reproducing the supplied artifact uses the recorded versions. A future migration should be tested and retrained rather than silently altering the saved experiment.

## 17. Neural networks from the beginning

### A neuron is a small calculation

An artificial neuron receives numbers, multiplies each by a learned weight, adds a bias, and applies a function called an **activation**:

```text
z = sum(weight × input) + bias
output = activation(z)
```

Weights say how strongly each incoming value contributes to the weighted sum. The bias shifts that sum. Neither is fixed by you in this project; training learns both.

### Layers and nonlinear activations

An **input layer** receives transformed feature values. A **hidden layer** computes intermediate values. An **output layer** produces the final score/probability.

The selected ANN has this shape:

```text
8 raw form fields
       |
preprocessing and selection
       |
5 selected numerical inputs
       |
32 hidden neurons, each with ReLU
       |
1 output neuron with sigmoid
       |
probability of label 1
```

**ReLU** means `max(0, z)`: −2 becomes 0, while 3 remains 3. It introduces nonlinearity. Stacking only linear functions would still produce a linear function, so nonlinear activations are essential to this model's ability to represent more complex relationships.

For binary classification, the output sigmoid maps the final score to `[0,1]`. Multi-class models often use softmax; that is background context, not the output activation of this binary model. The saved model's `out_activation_` is `'logistic'`, another name for the sigmoid.

### A worked forward pass

**Illustration with two inputs and one hidden neuron, not the saved network's actual weights:**

```text
Inputs: x1 = 0.5, x2 = −1.0
Hidden weights: 0.2, −0.4
Hidden bias: 0.1

Hidden z = 0.2×0.5 + (−0.4)×(−1.0) + 0.1 = 0.6
Hidden output = ReLU(0.6) = 0.6

Output weight = 1.5; output bias = −0.2
Output z = 1.5×0.6 − 0.2 = 0.7
Predicted probability = sigmoid(0.7) ≈ 0.6682
```

That is a **forward pass**. The actual model repeats this kind of computation across all 32 hidden neurons.

### Loss, gradients, backpropagation, and learning

After a forward pass on a labeled example, training computes loss. **Backpropagation** uses the chain rule to calculate gradients through the network. The **optimizer** then changes weights and biases using those gradients. Backpropagation computes the gradient; Adam decides how to use it for updates. They are related but different components.

A tiny worked update makes the idea tangible. Use a one-input sigmoid model with `x=1`, true label `y=1`, weight `w=0`, bias `b=0`, and learning rate 0.1. Initially `z=0`, `p=0.5`, and cross-entropy is about 0.6931.

For this simple model, the gradients are:

```text
dloss/dw = (p−y)×x = −0.5
dloss/db = p−y = −0.5

new_w = 0 − 0.1×(−0.5) = 0.05
new_b = 0 − 0.1×(−0.5) = 0.05
new_z = 0.05×1 + 0.05 = 0.1
new_p = sigmoid(0.1) ≈ 0.5250
new_loss = −ln(new_p) ≈ 0.6444
```

One update moved the probability toward the correct label and reduced loss. This is an illustration of basic gradient descent; the saved ANN uses Adam on mini-batches with regularization and many parameters.

### Training vocabulary

| Term | Meaning in this project |
|---|---|
| Epoch | A pass through the training examples for stochastic optimization |
| Mini-batch | A subset used to compute an update |
| Batch size | Number of examples per mini-batch; the estimator uses its `'auto'` default |
| Learning rate | Scale of optimizer updates; selected initial value 0.003 |
| Initialization | Starting random weights; controlled by `random_state` |
| Activation | Hidden ReLU; output sigmoid |
| Loss | Cross-entropy with regularization |
| L2 penalty | Penalizes large weight magnitudes; selected `alpha=0.001` |
| Early stopping | Stops when internal validation performance does not improve enough |
| Maximum iterations | A ceiling, not an instruction to always run that many epochs |

The model's construction and search choices appear in `model_specs` in Section 12. Its selected configuration has one hidden layer with 32 neurons, not two layers of 32 and 16; that two-layer option was a candidate, not the winner.

`validation_fraction=0.15` sets aside an internal part of the estimator's training data for early stopping. `n_iter_no_change=35` is its patience. In this classifier, early stopping monitors internal validation accuracy. It is distinct from the main 154-row validation set used later for threshold selection. [MLP early-stopping settings](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).

There is a subtle limitation: the preceding pipeline steps are fitted on the outer training fold before the MLP makes its internal split. Thus the ANN's internal stopping subset is not independent of earlier preprocessing/selection. The outer CV held-out fold and final external test are still kept separate and provide the main evaluation. Do not treat the internal stopping score as an independent final performance estimate.

### Inspecting the saved ANN

The fitted full-training network has weight matrices of shapes `(5,32)` and `(32,1)`, plus bias vectors of lengths 32 and 1.

```text
input-to-hidden weights: 5 × 32 = 160
hidden biases: 32
hidden-to-output weights: 32 × 1 = 32
output bias: 1
Total trainable parameters: 160 + 32 + 32 + 1 = 225
```

It stopped after **66 iterations**, well below the ceiling of 1,500. Its saved final training loss is about **0.4458** and best internal validation accuracy about **0.7971**. Those are training/early-stopping diagnostics; the final test ANN ROC-AUC is about 0.7956 and is a different quantity.

**Practice — inspect the actual learned object:**

```python
import joblib
bundle = joblib.load('artifacts/model.joblib')
ann_pipeline = bundle['components']['ANN']
ann = ann_pipeline.named_steps['model']
print([matrix.shape for matrix in ann.coefs_])
print([bias.shape for bias in ann.intercepts_])
print(ann.n_iter_, ann.loss_, ann.best_validation_score_)
print(ann.activation, ann.solver, ann.out_activation_)
```

Use the import/path setup from Section 3 before loading custom project objects. Learned weights are inspectable, but reading 225 numbers is not a simple explanation of a whole prediction. That motivates the explanation methods later.

An MLP is a real ANN whether implemented in scikit-learn, TensorFlow, or PyTorch. This project does not implement backpropagation from scratch; the library handles the numeric algorithm. The project code handles data, experimental design, configuration, integration, and evaluation. [Official MLP overview](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## 18. Cross-validation and hyperparameter search

### Five-fold cross-validation

A single split can produce a misleading impression of which settings work well. Cross-validation evaluates settings across several training/held-out arrangements using only the main training partition.

**Project excerpt — [src/diabetes/train.py](../src/diabetes/train.py), starting at line 129:**

```python
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
```

The 460 training rows are divided into five folds of 92 rows. Each evaluation trains on four folds (368 rows) and scores on the fifth. It repeats so every fold serves as the held-out fold once. Stratification keeps approximate class proportions in each.

```text
Run 1: [holdout] [train]   [train]   [train]   [train]
Run 2: [train]   [holdout] [train]   [train]   [train]
Run 3: [train]   [train]   [holdout] [train]   [train]
Run 4: [train]   [train]   [train]   [holdout] [train]
Run 5: [train]   [train]   [train]   [train]   [holdout]
```

The pipeline is independently fitted inside each run. A five-fold mean is the average score across the five held-out folds. The recorded fold standard deviation measures variation across those scores; it is not automatically a 95% confidence interval.

### Randomized search

A grid search tries every listed combination. A randomized search samples a limited number of combinations from the defined choices. Here each candidate family gets six configurations and each configuration is evaluated on five folds.

**Project excerpt — [src/diabetes/train.py](../src/diabetes/train.py), starting at line 149:**

```python
        search = RandomizedSearchCV(pipeline(estimator, engineered, k), params, n_iter=candidates,
                                    scoring="roc_auc", cv=cv, refit=True, random_state=SEED,
                                    n_jobs=1, error_score="raise", return_train_score=True)
        search.fit(X, y)
        models[name] = search.best_estimator_
        tuning[name] = {"best_params": search.best_params_, "best_cv_auc": float(search.best_score_),
                        "cv_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                        "cv_train_auc": float(search.cv_results_["mean_train_score"][search.best_index_])}
        pd.DataFrame(search.cv_results_).to_csv(ROOT / f"reports/tuning_{name}.csv", index=False)
```

- `n_iter=candidates`: six sampled configurations in the saved run.
- `scoring='roc_auc'`: rank configurations by held-out fold ROC-AUC.
- `refit=True`: after selecting a winner, fit that configuration on all 460 main training rows.
- `error_score='raise'`: expose fitting errors instead of silently ignoring them.
- `return_train_score=True`: retain fitting-data scores to inspect gaps.
- `n_jobs=1`: do this work in a single job for the small study.

The search performs `5 families × 6 candidates × 5 folds = 150` CV candidate fits, plus final refits. The representation ablation adds `2 learners × 3 representations × 5 folds = 30` fits. Ensemble calibration and library internals add more; those 150 are not the entire training cost.

The highest selected training-CV score belongs to the ANN, approximately 0.8214. It is therefore fixed as the comparator for the prespecified statistical comparison. That decision uses CV, not test ranking.

### Selection optimism and nested CV

If you try many settings and report only the highest mean CV score, the winner may benefit from chance. Here feature representations and tuning share the main training CV, so their best scores are **selection-biased**. That is different from directly fitting on the test set, but still important.

**Nested cross-validation** uses an inner CV for tuning and a separate outer CV for assessment, repeating tuning inside each outer training partition. It is more expensive and is not implemented in this completed study. The independent test partition is the separate assessment here.

Do not interpret six sampled configurations as a proof of global optimality. Hyperparameter search investigates a budgeted portion of a specified search space.

## 19. Ensemble learning and probability calibration

### Dummy baseline

A **baseline** provides a reference point. `DummyClassifier(strategy='prior')` predicts class probabilities based on the training class frequencies without using informative feature relationships.

**Project excerpt — [src/diabetes/train.py](../src/diabetes/train.py), starting at line 165:**

```python
    dummy = pipeline(DummyClassifier(strategy="prior"))
    dummy.fit(X, y)
```

Its test ROC-AUC is 0.5 because it gives the same probability to every row and cannot rank positives above negatives. Its test accuracy at 0.50 is about 64.94% because it predicts 0 for every test row. If a complicated model cannot beat such a baseline on the intended metrics, its complexity is not justified.

### Hard and soft voting

**Hard voting** combines predicted labels by majority. **Soft voting** averages class probabilities. With equal weights:

```text
ensemble_probability = (p1 + p2 + p3 + p4 + p5) / 5
```

**Illustration:** component probabilities `[0.20,0.30,0.40,0.50,0.60]` average to 0.40. Applying threshold 0.50 gives label 0. That is a probability-based combination, not a majority of independently thresholded labels.

**Project excerpt — [src/diabetes/train.py](../src/diabetes/train.py), starting at line 160:**

```python
    voter = VotingClassifier(estimators=list(models.items()), voting="soft", n_jobs=1)
    calibrated = CalibratedClassifierCV(voter, method="sigmoid", cv=cv, ensemble=False, n_jobs=1)
    print("Fitting ensemble with out-of-fold sigmoid calibration", flush=True)
    calibrated.fit(X, y)
    raw_ensemble = calibrated.calibrated_classifiers_[0].estimator
```

`VotingClassifier` clones/fits the candidate estimators; it does not simply hold references to their initial search objects. Each voter member contains its own preprocessing pipeline. Equal weights are fixed in advance to avoid adding an extra tuned weight vector on this small dataset.

### Why a probability can be misleading

Suppose a model assigns approximately 0.80 to many cases. If only about half actually have label 1, the probabilities overstate the observed positive frequency. **Calibration** assesses agreement between predicted probabilities and outcome frequencies over groups of similar scores.

Good ranking does not guarantee good probability magnitudes. A model can rank cases correctly while assigning overly extreme probabilities.

### Sigmoid calibration and out-of-fold predictions

The project fits a two-parameter sigmoid mapping to the raw ensemble scores. For conceptual purposes write `calibrated_p = sigmoid(a×raw_score+b)`. The library's sign convention can differ; the concept is a learned smooth mapping, not these illustrative parameter names.

**Out-of-fold predictions** are produced for training rows by estimators that did not fit those rows. With `ensemble=False`, the calibrator learns from cross-validated scores, then a full-training voter is fitted to supply scores at inference. It does not deploy an average of five fold voters in this configuration. [Official calibration description](https://scikit-learn.org/stable/modules/calibration.html).

The calibration folds still use hyperparameters chosen using the main training data. Parameter search is not repeated inside each calibration fold. The final test remains separate, but this is not a completely nested tuning/calibration protocol.

Calibration is measured, not assumed successful. On this holdout it **worsened Brier score** from 0.1709 to 0.1746 while improving the recorded binned ECE. The learned mapping preserves the observed ranking, so ROC-AUC is unchanged at 0.8107. Calibration does not necessarily improve accuracy at a fixed threshold either, because it changes probability values and can move records across that threshold.

The calibrated ensemble is the prespecified deployed research framework. It is not being described as the best individual model on every metric.

## 20. Thresholds, confusion matrices, and every metric

### A probability is not a label

A predicted probability of 0.40 is a score for label 1. A **threshold** converts that score into a decision:

```text
if p >= threshold: predict 1
else: predict 0
```

At threshold 0.50, 0.40 gives label 0. At threshold 0.33, the same probability gives label 1. The underlying probability did not change; the decision rule did.

**Project excerpt — [src/diabetes/evaluation.py](../src/diabetes/evaluation.py), starting at line 23:**

```python
def choose_threshold(y, p):
    """Maximize validation F1 on a prespecified grid; ties prefer 0.5."""
    grid = np.linspace(0.10, 0.90, 81)
    scores = np.array([f1_score(y, p >= t, zero_division=0) for t in grid])
    tied = np.flatnonzero(np.isclose(scores, scores.max()))
    return float(grid[tied[np.argmin(abs(grid[tied] - 0.5))]])
```

The function tries 81 thresholds from 0.10 through 0.90 in steps of 0.01 on the **validation set**. It selects the highest F1 and breaks approximate ties toward 0.50. The deployed ensemble threshold is 0.33. That is a model operating point, not a laboratory diagnostic cutoff.

Lowering the threshold generally predicts more positives: it can catch more true positives but also flag more negatives. Which error matters more is a decision about the application and costs. Optimizing F1 does not encode an approved clinical cost/benefit analysis.

### Confusion matrix

A confusion matrix counts the combinations of truth and prediction.

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actually 0 | True negative (TN) | False positive (FP) |
| Actually 1 | False negative (FN) | True positive (TP) |

For the deployed threshold of 0.33, the test counts are:

```text
             Predicted 0   Predicted 1
Actual 0         72            28
Actual 1         16            38
```

Total is `72+28+16+38 = 154`. There are 54 positive cases and 100 negative cases. The project code obtains these counts and calculates metrics:

**Project excerpt — [src/diabetes/evaluation.py](../src/diabetes/evaluation.py), starting at line 8:**

```python
def metrics(y, probabilities, threshold=0.5):
    y, p = np.asarray(y), np.asarray(probabilities)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(bins == b) * abs(y[bins == b].mean() - p[bins == b].mean())
              for b in range(10) if (bins == b).any())
    return dict(accuracy=float(accuracy_score(y, pred)), precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)),
                specificity=float(tn / (tn + fp)), roc_auc=float(roc_auc_score(y, p)),
                pr_auc=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, p)),
                log_loss=float(log_loss(y, p, labels=[0, 1])), ece_10_bins=float(ece),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=float(threshold))
```

`.ravel()` flattens the 2×2 matrix into `[TN, FP, FN, TP]` when the labels are `[0,1]`. `zero_division=0` defines a practical convention when a model predicts no positives and precision has a zero denominator.

### Accuracy

```text
Accuracy = (TP+TN) / total
         = (38+72) / 154 = 0.7143 = 71.43%
```

“How often was the hard label correct overall?” It can conceal poor minority-class performance.

### Precision

```text
Precision = TP / (TP+FP)
          = 38 / (38+28) = 0.5758 = 57.58%
```

“Among records predicted positive, how many really had positive labels?” Precision depends on how common positives are as well as model behavior.

### Recall / sensitivity

```text
Recall = TP / (TP+FN)
       = 38 / (38+16) = 0.7037 = 70.37%
```

“Among all truly positive records, how many did we identify?” Missing 16 of 54 positive test cases matters even when accuracy sounds acceptable.

### Specificity

```text
Specificity = TN / (TN+FP)
            = 72 / (72+28) = 0.7200 = 72.00%
```

“Among truly negative records, how many did we identify as negative?” False-positive rate is `1−specificity`, here 28%.

### F1 score

```text
F1 = 2 × precision × recall / (precision+recall)
   = 2×TP / (2×TP+FP+FN)
   = 76 / (76+28+16) = 0.6333
```

F1 is the harmonic mean of precision and recall. It omits true negatives, so it does not summarize every error type. It is not the ordinary arithmetic mean of precision and recall.

### ROC curve and ROC-AUC

A ROC curve shows true-positive rate against false-positive rate as the threshold changes. ROC-AUC summarizes ranking over thresholds. One interpretation is the probability that a randomly chosen positive receives a higher score than a randomly chosen negative, counting ties as half.

AUC 0.5 describes chance-level ranking for constant/random scores in the usual binary setting; 1.0 describes perfect ranking on the evaluated sample. The ensemble test AUC is about 0.8107. This is **not** “81.07% of cases were classified correctly,” nor is it the predicted probability for an individual.

A high ROC-AUC does not pick a good threshold or establish calibrated probabilities.

### Precision–recall curve and average precision

A precision–recall curve varies the threshold and displays precision versus recall. In imbalanced tasks it draws attention to the quality of positive predictions. A no-information reference depends on positive prevalence, about 0.3506 on this test set.

The code's `pr_auc` field uses `average_precision_score`. Average precision weights precision values by increases in recall:

```text
AP = sum over operating points of (increase in recall × precision)
```

It differs from trapezoidal area under a PR curve. The project's AP is about 0.6710. A report should state this convention instead of treating every “PR-AUC” implementation as identical. [Official average-precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

### Brier score

Brier measures the mean squared difference between probability and binary label:

```text
Brier = average((p−y)²)
```

**Illustration:** with `y=[0,1]`, probabilities `[0.1,0.9]` give `(0.01+0.01)/2 = 0.01`. Probabilities `[0.9,0.1]` give 0.81. Lower is better.

The ensemble Brier is about 0.1746. Brier is an overall probability-quality score affected by discrimination and calibration, not a pure measure of calibration alone.

### Log loss

This is the average cross-entropy calculation introduced in Section 12. It gives a strong penalty to confidently wrong probabilities. It evaluates probabilities rather than one particular threshold. A lower value is better; it need not rank models the same way as accuracy or AUC.

### Expected calibration error (ECE)

The implemented ECE groups probabilities into ten equal-width bins. In each occupied bin, it compares average probability with observed positive frequency, takes the absolute gap, and weights it by that bin's fraction of rows.

```text
ECE = sum(bin_weight × |observed_frequency − mean_probability|)
```

Empty bins are omitted. A probability of exactly 1 is put into the final bin. ECE depends on bin choices and can be unstable on small datasets. Opposing errors can be hidden by averaging within a bin.

The reliability chart uses eight bins while the ECE number uses ten; those choices are explicit in the code. The chart plots mean probability against observed fraction, with the diagonal representing ideal agreement. It is not an ROC curve.

In this experiment ECE improves after calibration while Brier worsens. That disagreement is possible because the measures aggregate errors differently. Do not select whichever metric tells the most flattering story. [Official probability-scoring and calibration guidance](https://scikit-learn.org/stable/modules/model_evaluation.html).

## 21. Actual results and how to interpret them

The first table uses the **same threshold, 0.50**, for all hard-label metrics. Ranking/probability metrics are computed from probabilities.

| Model | Accuracy | Precision | Recall | F1 | Specificity | ROC-AUC | AP | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | 0.6948 | 0.5574 | 0.6296 | 0.5913 | 0.7300 | 0.7826 | 0.6134 | 0.1905 |
| RandomForest | 0.7403 | 0.6000 | 0.7778 | 0.6774 | 0.7200 | 0.8215 | 0.6782 | 0.1720 |
| GradientBoosting | 0.7403 | 0.6346 | 0.6111 | 0.6226 | 0.8100 | 0.8222 | 0.6772 | 0.1640 |
| SVM | 0.7078 | 0.6047 | 0.4815 | 0.5361 | 0.8300 | 0.8004 | 0.6659 | 0.1773 |
| ANN | 0.7273 | 0.6250 | 0.5556 | 0.5882 | 0.8200 | 0.7956 | 0.6551 | 0.1799 |
| DummyPrior | 0.6494 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5000 | 0.3506 | 0.2277 |
| RawEnsemble | 0.7273 | 0.6071 | 0.6296 | 0.6182 | 0.7800 | 0.8107 | 0.6710 | 0.1709 |
| CalibratedEnsemble | 0.7273 | 0.6250 | 0.5556 | 0.5882 | 0.8200 | 0.8107 | 0.6710 | 0.1746 |

The deployed calibrated ensemble uses threshold **0.33** instead. Its metrics at that operating point are:

| Metric | Value at threshold 0.33 |
| --- | --- |
| accuracy | 0.7143 |
| precision | 0.5758 |
| recall | 0.7037 |
| f1 | 0.6333 |
| specificity | 0.7200 |
| roc_auc | 0.8107 |
| pr_auc | 0.6710 |
| brier | 0.1746 |
| tn | 72 |
| fp | 28 |
| fn | 16 |
| tp | 38 |

The same model has different accuracy, precision, recall, F1, and specificity at these two thresholds because its hard predictions differ. Its AUC, AP, and Brier do not change just because the decision threshold changes.

The selected ANN was the best of the tuned training-CV baselines. Gradient boosting achieves higher test AUC. This is not a contradiction: training samples, CV scores, selection effects, and test sampling variability differ. A training-CV ranking is not a promise about every later sample.

The raw voter and calibrated voter have identical ranking scores in this run, but different Brier and log loss. Calibration is a procedure applied to scores; success must be measured. It is legitimate to conclude that a component did not improve the selected experimental criterion.

### What to say about the result

A supported description is: “On the reserved test partition, the calibrated ensemble achieved ROC-AUC about 0.811. Its validation-selected threshold traded lower specificity for higher recall relative to threshold 0.50. The ensemble did not outperform every individual model, and sigmoid calibration increased its Brier score.”

An unsupported description is: “The system diagnoses diabetes with 81% accuracy,” because 0.811 is an AUC, the task is historical label classification, and the system is not validated as a diagnosis service.

The remaining limitation is not solved by improving page styling or removing a paragraph. Interface changes modify presentation; they do not change the dataset or validation protocol.

## 22. Bootstrap intervals and statistical comparisons

### Why point estimates are not enough

A test set of 154 records is one sample. Different records from the same relevant population could produce different scores. A point estimate such as 0.811 conceals that sampling uncertainty.

### Bootstrap resampling

A **bootstrap** draws a new sample of the same size **with replacement** from existing observations. “With replacement” means a row can appear multiple times while another row may not appear in that draw.

The implementation does this separately within each label class, preserving 100 negatives and 54 positives in every test resample. It computes metrics 1,000 times and takes the 2.5th and 97.5th percentiles as a percentile interval.

**Project excerpt — [src/diabetes/evaluation.py](../src/diabetes/evaluation.py), starting at line 31:**

```python
def bootstrap_intervals(y, p, threshold, iterations=1000, seed=42):
    """Stratified resampling: sampling uncertainty conditional on the fitted model."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    groups = [np.flatnonzero(y == v) for v in [0, 1]]
    draws = []
    for _ in range(iterations):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        draws.append(metrics(y[idx], p[idx], threshold))
    names = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "pr_auc", "brier"]
    return {k: [float(v) for v in np.quantile([d[k] for d in draws], [0.025, 0.975])] for k in names}
```

The ensemble's ROC-AUC interval is about **0.7361–0.8759**. This is a **conditional, stratified bootstrap interval** for the fitted model evaluated under the observed class mix. It does not include uncertainty from refitting, hyperparameter selection, representation selection, population shifts, or a different disease prevalence.

It is not a confidence interval for one person's prediction. It also does not mean there is a directly measurable 95% probability that a fixed unknown AUC lies inside this particular already-calculated interval. The procedure is intended to reflect repeated-sampling uncertainty under its assumptions, which matter especially for a small sample.

### Paired comparison

If two models are evaluated on the same rows, compare their predictions **pairwise**, not as though they came from unrelated test samples. The model errors can be strongly correlated.

**Project excerpt — [src/diabetes/evaluation.py](../src/diabetes/evaluation.py), starting at line 44:**

```python
def paired_comparison(y, a, b, threshold_a=0.5, threshold_b=0.5, iterations=1000, seed=42):
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    correct_a, correct_b = (a >= threshold_a) == y, (b >= threshold_b) == y
    n_ab = int((correct_a & ~correct_b).sum())
    n_ba = int((~correct_a & correct_b).sum())
    pvalue = float(binomtest(n_ab, n_ab + n_ba, 0.5).pvalue) if n_ab + n_ba else 1.0
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == v) for v in [0, 1]]
    deltas = []
    for _ in range(iterations):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return {"mcnemar_exact_p": pvalue, "a_only_correct": n_ab, "b_only_correct": n_ba,
            "auc_difference_a_minus_b": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "paired_auc_difference_95_ci": np.quantile(deltas, [0.025, 0.975]).tolist(),
            "note": "Single prespecified comparison; not proof of equivalence or clinical superiority."}
```

The code makes one prespecified comparison: calibrated ensemble versus the ANN selected by training CV. Both use threshold 0.50 for the McNemar calculation.

- Ensemble correct / ANN wrong: two rows.
- Ensemble wrong / ANN correct: two rows.
- Exact McNemar p-value: 1.0.

**McNemar's test** examines imbalance among these discordant correctness pairs. It does not test AUC or probability calibration. A large p-value does not prove equivalence; it can reflect limited evidence or low power.

The separate paired bootstrap uses identical resampled indexes for both AUC calculations and examines `ensemble_AUC − ANN_AUC`. The observed difference is about **0.0152**; its saved 95% percentile interval is approximately **0.0006–0.0320**. This interval is just above zero under the conditional procedure; it should be interpreted cautiously given the small sample, finite resampling, and missing training uncertainty. It is not a claim that the ensemble is superior to every other model or clinically useful.

These two tests need not agree because they ask different questions: thresholded correctness and ranking. Avoid searching every model pair and reporting only a favorable p-value; multiple testing requires additional consideration.

## 23. Explainability: permutation importance and Shapley values

**Explainability** means inspecting the behavior of a trained model. It can help locate mistakes or describe associations, but it is not proof of truth, causality, fairness, or safety.

### Permutation importance

For a fixed fitted model, shuffle one feature's values across test rows and recompute a performance metric. If this damages performance, the model was using useful information from that feature.

**Project excerpt — [src/diabetes/train.py](../src/diabetes/train.py), starting at line 220:**

```python
    importance = permutation_importance(calibrated, Xt, yt, n_repeats=12, random_state=SEED, scoring="roc_auc", n_jobs=1)
    imp = pd.DataFrame({"feature": FEATURES, "auc_drop_mean": importance.importances_mean,
                        "auc_drop_std": importance.importances_std}).sort_values("auc_drop_mean", ascending=False)
    imp.to_csv(ROOT / "reports/permutation_importance.csv", index=False)
```

The study repeats shuffling 12 times per feature and measures the reduction in ROC-AUC. A larger positive drop suggests more dependence on that feature under this calculation. A negative mean can occur because shuffling happens to improve the score on that sample.

The reported repeat SD is variation across shuffles, not a full confidence interval covering every source of uncertainty. Correlated predictors can substitute for one another, and shuffling can make unrealistic combinations. Importance is therefore conditional on the fitted model, data, and method.

This analysis happens after freezing/evaluating the model. It is not fed back into feature selection on the test set.

### Local versus global explanations

A **local explanation** concerns one input. A **global summary** aggregates behavior across inputs. A feature can be important overall but have little contribution for one row relative to the chosen reference.

Our mean-absolute Shapley summary covers only the first 12 fixed test cases. It is explicitly illustrative, not a population-wide summary of every case.

### Shapley values as shared credit

Imagine a team with several contributors. To allocate credit fairly, compare what each contributor adds across different possible subsets of the team. A Shapley value applies that idea to feature contributions.

For a set `S` of included features, define `v(S)` as the average model probability after keeping those features from the explained row and replacing omitted features using background records. Each feature's contribution averages the change caused by adding it across every subset that does not already include it.

```text
phi_j = sum over S not containing j of
        [|S|! × (M−|S|−1)! / M!] × [v(S plus j) − v(S)]
```

Here `M=8` raw inputs. A factorial, such as `3!`, means `3×2×1`; the weights account for the number of possible feature-ordering positions. You do not need to memorize the formula before understanding the comparison it describes.

Eight on/off choices produce `2^8=256` coalitions. Each coalition is averaged over 24 training background rows, giving `256×24=6,144` input rows for one explanation. This is manageable for eight inputs; with 30 inputs, exact enumeration would involve over a billion coalitions and require a different approach.

**Project excerpt — [src/diabetes/explain.py](../src/diabetes/explain.py), starting at line 7:**

```python
def exact_shapley(model, row, background):
    """Exact interventional Shapley values over eight raw input features.

    Enumerate all 256 coalitions. Missing features are replaced jointly with
    each background row, averaging predicted positive-class probabilities.
    Exactness is relative to this finite background, not the population.
    Feature dependence is not modeled and contributions are not causal.
    """
    x = np.asarray(row, dtype=float).reshape(-1)
    bg = np.asarray(background, dtype=float)
    m = len(FEATURES)
    if len(x) != m or bg.ndim != 2 or bg.shape[1] != m or len(bg) == 0:
        raise ValueError("Expected one eight-feature row and a nonempty eight-feature background.")
    masks = ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1).astype(bool)
    batch = np.where(masks[:, None, :], x[None, None, :], bg[None, :, :])
    p = model.predict_proba(pd.DataFrame(batch.reshape(-1, m), columns=FEATURES))[:, 1]
    values = p.reshape(2 ** m, len(bg)).mean(axis=1)
    phi = np.zeros(m)
    for j in range(m):
        for mask in range(2 ** m):
            if mask & (1 << j):
                continue
            size = mask.bit_count()
            weight = factorial(size) * factorial(m - size - 1) / factorial(m)
            phi[j] += weight * (values[mask | (1 << j)] - values[mask])
    return {"baseline": float(values[0]), "probability": float(values[-1]),
            "contributions": {name: float(value) for name, value in zip(FEATURES, phi)},
            "additivity_error": float(abs(values[0] + phi.sum() - values[-1])),
            "background_size": len(bg), "method": "exact interventional Shapley (256 coalitions)"}
```

Read this function in stages:

1. Turn the explained row and background into numeric arrays.
2. Create all 256 inclusion masks using bits.
3. For each mask/background record, combine included row values with omitted background values.
4. Call the actual calibrated pipeline on every combined row.
5. Average the 24 probabilities for each coalition.
6. For each feature, sum its weighted marginal changes.
7. Return the baseline, output, contributions, and additivity error.

A **bit mask** is a compact way of representing which features are included. The lowest bit represents one feature, the next bit another. `mask & (1 << j)` checks whether feature `j` is present. `mask.bit_count()` counts included features. These are programming techniques used to implement the mathematics, not additional ML models.

`np.where` builds the combined arrays efficiently rather than asking Python to construct thousands of rows in an inner loop. `reshape` changes array shape without changing which values it holds; the model receives the familiar eight-column DataFrame.

### Additivity and a real example

The explained output should satisfy:

```text
baseline_probability + sum(feature_contributions) = prediction_probability
```

For the form's current example inputs:

| Raw feature | Example value |
| --- | --- |
| Pregnancies | 2 |
| Glucose | 120 |
| BloodPressure | 70 |
| SkinThickness | 25 |
| Insulin | 85 |
| BMI | 30 |
| DiabetesPedigreeFunction | 0.45 |
| Age | 32 |

The actual saved model yields:

| Quantity | Measured value |
| --- | --- |
| Calibrated positive-class probability | 0.24874188 (24.8742%) |
| Threshold | 0.33 |
| Predicted Outcome | 0 |
| Background probability | 0.23144720 |
| Component probability SD | 0.07678972 |
| LogisticRegression | 0.37271502 |
| RandomForest | 0.44313103 |
| GradientBoosting | 0.34222283 |
| SVM | 0.28757061 |
| ANN | 0.21625145 |

Its feature contributions, in **percentage points**, are:

| Raw feature | Contribution (percentage points) |
| --- | --- |
| Pregnancies | +0.0000 |
| Glucose | -0.3847 |
| BloodPressure | +0.0000 |
| SkinThickness | +0.2443 |
| Insulin | -2.7323 |
| BMI | +1.4790 |
| DiabetesPedigreeFunction | +0.0000 |
| Age | +3.1232 |

Adding these contributions to the background probability reconstructs about **24.8742%**. A contribution of +3.1232 percentage points is an additive change relative to the background, not a 3.1232% relative multiplier.

Three contributions are zero because their raw columns are excluded by the saved selected-five pipelines. The baseline is about 23.1447%, which is the model's average output on the 24 background rows; it is not automatically equal to dataset prevalence.

The additivity error is around floating-point roundoff. For the 12 saved study examples, the maximum error was about `2.22×10^−16`.

The code explains the **full deployed calibrated pipeline**, including preprocessing and selection. It does not explain a separate surrogate logistic model and then pretend that surrogate is the ensemble.

The name **interventional Shapley** here describes a feature-substitution value function; it does not mean a verified medical intervention. Replacing omitted inputs with background values can break dependence with included inputs. “Exact” means exact enumeration for this particular finite background and value function, not a uniquely true explanation in the population. [Original SHAP framework paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).

The implementation computes these values directly and does not require installing the separate `shap` Python library.

## 24. Model disagreement and what-if analysis

### Disagreement is one uncertainty signal

The five learners can predict different probabilities for the same row. The application computes their standard deviation:

**Project excerpt — [src/diabetes/train.py](../src/diabetes/train.py), starting at line 251:**

```python
    components = np.vstack([m.predict_proba(Xt)[:, 1] for m in raw_ensemble.named_estimators_.values()])
    std = components.std(axis=0)
```

For the current example, the five component probabilities range from about 0.2163 to 0.4431, and their SD is about **0.0768**. The final calibrated probability is about 0.2487; it is not the raw mean itself. The raw mean is about 0.3324 before applying the calibrator.

A large disagreement can indicate sensitivity to model choice. A small disagreement is not proof of correctness: all models could share the same dataset bias. Because the learners are not independent samples of a formal uncertainty distribution, this SD is not a confidence interval or a coverage guarantee.

Two broad ideas in ML are **epistemic uncertainty**, related to limited knowledge/data/model specification, and **aleatoric uncertainty**, related to variability/noise in observations. The project does not formally separate or estimate those two quantities. Component SD is only a descriptive heuristic.

### What-if analysis

The interface takes the last submitted input profile, copies it, changes one measured feature, and compares model probabilities. It also computes a 35-point curve across that feature's observed training range.

**Project excerpt — [src/diabetes/predict.py](../src/diabetes/predict.py), starting at line 54:**

```python
def what_if(bundle, payload, feature, value):
    if feature not in ["Glucose", "BMI", "BloodPressure", "Insulin", "SkinThickness"]:
        raise ValueError("What-if analysis supports measured physiological features only.")
    original, original_notes = validate_inputs(payload)
    scenario_payload = dict(payload)
    scenario_payload[feature] = value
    scenario, scenario_notes = validate_inputs(scenario_payload)
    base_p = float(bundle["model"].predict_proba(original)[0, 1])
    scenario_p = float(bundle["model"].predict_proba(scenario)[0, 1])
    grid = np.linspace(bundle["reference"][feature]["min"], bundle["reference"][feature]["max"], 35)
    rows = pd.concat([original] * len(grid), ignore_index=True)
    rows[feature] = grid
    sweep = bundle["model"].predict_proba(rows)[:, 1]
    notes = list(dict.fromkeys(original_notes + scenario_notes))
    if value is not None and value != 0 and not bundle["reference"][feature]["min"] <= value <= bundle["reference"][feature]["max"]:
        notes.append(f"Scenario {feature} is outside its training range; this is extrapolation.")
    return {"feature": feature, "value": value, "baseline_probability": base_p,
            "scenario_probability": scenario_p, "difference": scenario_p - base_p,
            "curve": [{"value": float(v), "probability": float(p)} for v, p in zip(grid, sweep)],
            "notes": notes, "interpretation": "An association within the model, not a treatment effect or medical recommendation."}
```

`dict(payload)` makes a new dictionary so the scenario does not overwrite the baseline input. `np.linspace` generates equally spaced grid values. `pd.concat` repeats the baseline row, and assigning one column creates the sweep.

Changing Glucose from 120 to 110 while leaving other inputs fixed asks “how does this model's output respond?” It does not estimate what would happen to the person's actual outcome after a treatment. Related measurements might change together in reality, so keeping all other inputs fixed can make a scenario unrealistic.

An **extrapolation** occurs when a new value falls outside the range seen in the training data. The prediction code emits notes for such values. Being within every individual feature's range also does not prove the whole joint profile was well represented in training.

The standalone report what-if plot uses a median background profile. The interactive UI uses your last submitted profile. Those are different reference rows, so their curves need not match exactly.

## 25. Saving the model and reproducing an experiment

### What is a model artifact?

A fitted model contains learned state, not just its class name. If you save only `MLPClassifier(hidden_layer_sizes=(32,))`, that configuration does not preserve the learned 225 parameters.

The project stores a deployment bundle:

**Project excerpt — [src/diabetes/train.py](../src/diabetes/train.py), starting at line 172:**

```python
    bg = X.sample(n=min(24, len(X)), random_state=SEED)
    reference = {f: {"min": float(X[f].replace(0, np.nan).min()) if f in ZERO_MISSING else float(X[f].min()),
                     "max": float(X[f].max())} for f in FEATURES}
    bundle = {"model": calibrated, "components": raw_ensemble.named_estimators_,
              "threshold": operating_threshold, "background": bg, "reference": reference,
              "representation": selected, "features": FEATURES, "seed": SEED,
              "sklearn_version": version("scikit-learn")}
    joblib.dump(bundle, ROOT / "artifacts/model.joblib", compress=3)
```

The bundle includes the calibrated model, fitted voter components, threshold, background rows, input feature names, training reference ranges, representation choice, seed, and library version. It is saved **before** generating final test predictions.

`joblib.dump` serializes the Python objects to a binary file. `joblib.load` restores them. Saving the full pipeline preserves the medians/scales/selector along with the classifier; otherwise the application could preprocess new inputs differently from training.

Do not load untrusted joblib/pickle files: deserializing Python objects can execute code. This is a property of that format, not a problem unique to diabetes modeling. The supplied artifact should be used with the trusted local code and recorded software environment.

### Reproducibility

A reproducible experiment records enough detail to rerun the procedure and investigate differences:

| Recorded item | Why it matters |
|---|---|
| Original CSV and SHA-256 | Know exactly which data bytes were used |
| Original row indexes in each partition | Know which rows were fitted, thresholded, or evaluated |
| Random seed | Repeat shuffling, searches, sampling, and stochastic fitting |
| Search settings | Know which alternatives were actually considered |
| Selected thresholds | Reproduce hard-label decisions |
| Package/Python versions | Library behavior and binary compatibility can change |
| Predictions and metrics | Audit calculations independently |
| Warnings | Distinguish successful execution from future compatibility concerns |

A **checksum** such as SHA-256 is a deterministic fingerprint of file bytes. It can detect that a file changed. It is not proof of ethical data collection, clinical quality, or validity of a model.

The recorded completed environment used Python 3.14.6 and the exact versions listed in `requirements-lock.txt`. `requirements.txt` gives broader supported installation ranges for retraining. A requirements file is not an automatically isolated environment; you still need to activate the intended virtual environment.

Scikit-learn model persistence is not guaranteed across library versions. If you use a different compatible software stack, retrain instead of assuming the existing binary will load identically. A seed also does not guarantee identical floating-point results across every hardware/library combination.

`threadpool_limits(limits=1)` limits numeric library threading for this small workload. It is a resource/performance choice, not a learning algorithm or an accuracy improvement.

## 26. How the website talks to the model

### Three frontend technologies

| File / language | Role |
|---|---|
| `web/index.html` / HTML | Form fields, headings, result containers, table, details sections |
| `web/styles.css` / CSS | Font, spacing, colors, borders, responsive layout |
| `web/app.js` / JavaScript | Read form values, send requests, display returned results |

After your feedback, the page uses standard fonts, simple blue buttons, a plain form, and expandable explanation/chart sections. CSS media rules stack fields on narrow screens. The presentation HTML is a separate document and was not redesigned by the frontend simplification.

**Project excerpt — [web/index.html](../web/index.html), starting at line 24:**

```html
        <label>Glucose<small>2-hour OGTT plasma glucose (mg/dL)</small><input name="Glucose" type="number" min="0" max="1000000" step="any" value="120" placeholder="Unknown"></label>
```

HTML `name` attributes match the Python feature keys. `type='number'`, `min`, `step`, and `required` provide browser-side checks. A caller can bypass the browser, so Python validates again.

### The request path

```text
Browser form
  -> JavaScript builds a JSON dictionary
  -> POST /api/predict
  -> Python parses JSON
  -> validate_inputs
  -> fitted pipeline predict_proba
  -> threshold, components, Shapley explanation
  -> Python sends JSON response
  -> JavaScript updates the page
```

**JSON** is a text format for objects, lists, numbers, strings, booleans, and null. It is a way of transporting information, not a trained model.

**Project excerpt — [web/app.js](../web/app.js), starting at line 7:**

```javascript
async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed.');
  return data;
}
```

`fetch` sends an HTTP request. `async`/`await` allows the function to wait for a response without synchronously blocking the entire page. `JSON.stringify` turns the JavaScript object into request text; `response.json()` parses returned text. An HTTP response status can indicate success or an error.

### API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serve the form page |
| `/styles.css`, `/app.js` | GET | Serve frontend code |
| `/api/health` | GET | Confirm server/model availability |
| `/api/results` | GET | Return saved experimental results |
| `/api/predict` | POST | Validate inputs and produce a prediction/explanation |
| `/api/what-if` | POST | Compare a changed measurement with the baseline |
| `/figures/name.png` | GET | Serve a saved plot through restricted path handling |
| `/presentation` | GET | Serve the supporting slide document |

A **GET** generally asks for data; a **POST** submits a request body to process. The server returns 200 for success, 400 for invalid payloads, 403 for rejected cross-origin requests, 404 for unknown paths, and 415 for the wrong content type in the implemented checks.

### Server startup

**Project excerpt — [src/diabetes/app.py](../src/diabetes/app.py), starting at line 86:**

```python
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    path = ROOT / "artifacts/model.joblib"
    if not path.exists():
        parser.error("Train first: PYTHONPATH=src python -m diabetes.train")
    bundle = joblib.load(path)
    with threadpool_limits(limits=1):
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(bundle))
        print(f"Diabetes study interface: http://127.0.0.1:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            server.server_close()
```

The server loads the model once, binds to `127.0.0.1`, and serves requests. `127.0.0.1` is localhost: the loopback address for this machine. `8000` is a port, identifying the local service. The browser must reach the machine running the server; a remote workspace may need its usual port-forwarding arrangement.

`ThreadingHTTPServer` can handle requests in different threads. An inference lock serializes the application's prediction calls. This is a small local demonstrator, not a production deployment architecture.

The server reads static HTML/CSS/JS from disk for requests. A frontend edit normally requires a page refresh, not retraining. Changes to Python code in a running process generally require restarting that process. Changing a display paragraph does not change the saved model.

### Server-side validation and inference

**Project excerpt — [src/diabetes/predict.py](../src/diabetes/predict.py), starting at line 9:**

```python
def validate_inputs(payload):
    if not isinstance(payload, dict) or set(payload) != set(FEATURES):
        raise ValueError("Supply exactly the eight documented feature names.")
    values, notes = {}, []
    for name in FEATURES:
        value = payload[name]
        if value is None and name in ZERO_MISSING:
            values[name] = np.nan
            notes.append(f"{name} is missing; the training-fitted median will be used.")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number" + (" or null." if name in ZERO_MISSING else "."))
        if value < 0 or value > 1_000_000:
            raise ValueError(f"{name} must be nonnegative and no greater than 1,000,000 (application limit).")
        if name in ["Pregnancies", "Age"] and value != int(value):
            raise ValueError(f"{name} must be an integer.")
        if name == "Age" and value < 21:
            raise ValueError("The dataset covers women aged 21 and above; Age must be at least 21.")
        if name == "DiabetesPedigreeFunction" and value <= 0:
            raise ValueError("DiabetesPedigreeFunction must be positive.")
        if value == 0 and name in ZERO_MISSING:
            notes.append(f"{name}=0 is treated as missing; the training-fitted median will be used.")
        values[name] = float(value)
    return pd.DataFrame([values], columns=FEATURES), notes
```

The checks require exactly the expected keys, reject nonfinite numbers, negative measurements, unsupported missing required inputs, fractional count/age values, and ages below the dataset's form scope. The million-unit cap is a broad application limit, not a validated medical upper bound. Values beyond observed training ranges get extrapolation notes rather than being represented as well-supported predictions.

**Project excerpt — [src/diabetes/predict.py](../src/diabetes/predict.py), starting at line 35:**

```python
def predict(bundle, payload, explain=True):
    row, notes = validate_inputs(payload)
    probability = float(bundle["model"].predict_proba(row)[0, 1])
    components = {name: float(model.predict_proba(row)[0, 1]) for name, model in bundle["components"].items()}
    for name, bounds in bundle["reference"].items():
        value = row[name].iloc[0]
        if pd.isna(value) or (name in ZERO_MISSING and value == 0):
            continue
        if not bounds["min"] <= value <= bounds["max"]:
            notes.append(f"{name} lies outside the training range [{bounds['min']:g}, {bounds['max']:g}]; this is extrapolation.")
    result = {"probability": probability, "predicted_outcome": int(probability >= bundle["threshold"]),
              "threshold": bundle["threshold"], "components": components,
              "component_std": float(np.std(list(components.values()))), "notes": notes,
              "scope": "Educational classification of historical Pima Outcome labels; not a diagnosis or validated future risk forecast."}
    if explain:
        result["explanation"] = exact_shapley(bundle["model"], row.iloc[0], bundle["background"])
    return result
```

The first call runs the complete fitted pipeline and extracts probability for class 1. The component calls inspect each fitted voter member. The returned dictionary includes the final threshold decision and an optional explanation.

JSON `null` is Python `None` after parsing. The validator converts allowed missing laboratory fields to `np.nan`. That missing value reaches the same imputer fitted during training.

A complete trace for the form example is:

1. The form sends eight raw values.
2. Python confirms their schema, types, and basic limits.
3. Each fitted pipeline applies its own zero policy and learned transformations.
4. The selected columns become Glucose, SkinThickness, Insulin, BMI, and Age.
5. The five model probabilities are produced.
6. Their raw average is about 0.3324.
7. The learned sigmoid maps it to about 0.2487.
8. `0.2487 < 0.33`, so the predicted dataset label is 0.
9. Exact Shapley values reconstruct that probability relative to the background.
10. The browser presents the result and lets you explore a scenario.

The CSS does none of those calculations. The browser displays results; the Python model supplies them.

## 27. Testing, notebooks, and generated documents

### What was tested?

The existing test suite has **22 test methods**. Some methods cover several related cases using subtests. They include dataset shape/checksum, partition overlap, zero handling, engineered features, input order, imputation invariance, fold-local imputation, known metric values, threshold tie behavior, bootstrap repeatability, paired comparison, known-linear Shapley values, serialized inference, input validation, scenarios, HTTP behavior, and restricted static paths.

**Project excerpt — [tests/test_core.py](../tests/test_core.py), starting at line 93:**

```python
    def test_cross_validation_imputation_uses_fold_training_only(self):
        sample = self.df.iloc[:120]
        cv = StratifiedKFold(3, shuffle=True, random_state=42)
        result = cross_validate(pipeline(LogisticRegression(max_iter=1000)), sample[FEATURES],
                                sample.Outcome, cv=cv, return_estimator=True)
        for estimator, (train_idx, _) in zip(result["estimator"], cv.split(sample[FEATURES], sample.Outcome)):
            clean = sample.iloc[train_idx][FEATURES].copy()
            clean[ZERO_MISSING] = clean[ZERO_MISSING].replace(0, np.nan)
            np.testing.assert_allclose(estimator.named_steps["imputer"].statistics_, clean.median().to_numpy())
```

This test inspects each fitted CV imputer and compares its statistics to medians computed independently from that fold's training rows. That checks a meaningful leakage property; it does more than merely assert that a pipeline object exists.

**Project excerpt — [tests/test_core.py](../tests/test_core.py), starting at line 122:**

```python
    def test_exact_shapley_known_linear_model(self):
        row = np.arange(1, 9, dtype=float)
        explanation = exact_shapley(LinearProbability(), row, np.zeros((3, 8)))
        np.testing.assert_allclose(list(explanation["contributions"].values()), row * .01, atol=1e-12)
        self.assertLess(explanation["additivity_error"], 1e-12)
        self.assertAlmostEqual(explanation["baseline"], .1)
```

The known linear model has a mathematically predictable attribution. Comparing the implementation against that expectation checks the Shapley algorithm independently.

The last recorded full suite passed. The notebook executed all eight code cells with zero error outputs. The live health endpoint reported the model loaded. Frontend JavaScript syntax and DOM targets were checked, and a representative PDF slide was rendered and inspected. A browser automation/visual interaction suite was **not** run, so the project does not claim that kind of browser verification.

A passing test checks the behavior encoded in that test. It does not prove every possible input works, confirm clinical validity, or establish external generalization. Testing software and evaluating a classifier are related but distinct.

`unittest` is Python's standard testing framework. `scripts/verify.py` discovers the tests, runs them, and writes actual output to `reports/verification.txt`. Its `Tee` object copies text to both the console and log; it is not an ML component.

### Notebook

An `.ipynb` file contains ordered code/Markdown cells, metadata, execution counts, and saved outputs in a JSON-based format. `notebooks/diabetes_analysis.ipynb` inspects the completed data, quality, feature ablation, metrics, explanations, and scenarios. It does not silently train a new model on opening.

Notebook execution state matters: running cells out of order can leave variables inconsistent with the visible order. A fresh “restart and run all” is more reliable when checking that a notebook is self-contained.

### Report material and slides

`scripts/build_deliverables.py` reads measured results and generates supporting notes, the notebook, a fourteen-slide PDF, a standalone HTML presentation, a slide outline, and the artifact manifest. The training script separately creates the figures and figure appendix PDF.

A generated supporting report scaffold is not a completed original student report. Names, genuine contributions, original prose, screenshots, institutional checks, and viva performance remain student responsibilities. Removing the website's introductory paragraph did not change the academic rubric's authorship requirement.

### Literature and contribution

A literature survey compares methods, data, evaluation, and limitations in earlier work. A research gap must be supported by reading, not by claiming that an existing combination of ensemble learning, calibration, XAI, and a web interface has never been used.

This project offers a reproducible course-level integration and experimental comparison. Its value includes honest negative findings and interpretation of the actual pipeline. It does not invent a new learning algorithm or reproduce every cited paper's experiment.

## 28. Complete file inventory

This inventory includes the supplied rubric, the code/configuration files created, documentation, generated artifacts, experimental outputs, plots, and the project archive. “Generated” means an executed script produced it; it was not an independently authored raw dataset.

Python caches such as `__pycache__/*.pyc`, environment-owned hidden directories, and temporary files in `/tmp` are excluded because they are execution/environment byproducts, not project deliverables. The temporary PDF-rendered preview was also outside the project directory.

### Root files, dataset, and Python source

| File | Origin | Purpose |
| --- | --- | --- |
| `.gitignore` | Created | Ignore Python caches, virtual environment and notebook checkpoints |
| `DiabetesPrediction_Project.zip` | Generated/updated | Portable archive of project source, data, documentation and artifacts |
| `EvaluationRubrics_DiabetesPrediction.pdf` | Supplied by user | Original 13-component evaluation rubric; not authored in this project |
| `README.md` | Created/updated | Setup commands, delivered artifacts, actual results and rubric responsibilities |
| `app.py` | Created | Convenient root launcher for the local web server |
| `data/pima_raw.csv` | Downloaded | Unmodified real public dataset mirror; not synthetic or authored patient data |
| `pyproject.toml` | Created | Package metadata, supported Python/dependencies and src package discovery |
| `requirements-lock.txt` | Generated | Exact versions used for the included model artifact |
| `requirements.txt` | Created | Broad scientific dependency ranges for installing and retraining |
| `scripts/build_deliverables.py` | Created | Generate supporting evidence notes, notebook, slides and manifest |
| `scripts/verify.py` | Created | Run unittest suite and copy output to verification.txt |
| `src/diabetes/__init__.py` | Created | Identify diabetes as a Python package |
| `src/diabetes/app.py` | Created | Local HTTP routes, model loading, static serving and request processing |
| `src/diabetes/data.py` | Created | Paths, feature names, missingness policy, seed, CSV loading/checks and checksum |
| `src/diabetes/evaluation.py` | Created | Metrics, validation threshold selection, bootstrap and paired comparison |
| `src/diabetes/explain.py` | Created | Exact finite-background raw-feature Shapley computation |
| `src/diabetes/predict.py` | Created | Validated inference, component probabilities, extrapolation notes and scenarios |
| `src/diabetes/preprocessing.py` | Created | Custom transformer and imputation/scaling/selection/model pipeline |
| `src/diabetes/train.py` | Created | Run splits, EDA, ablation, search, ensemble, calibration, frozen evaluation and artifact export |
| `tests/test_core.py` | Created | 22 regression tests covering experimental and application behavior |
| `train.py` | Created | Convenient root launcher for the training/evaluation script |

### Documentation, notebook, and frontend

| File | Origin | Purpose |
| --- | --- | --- |
| `docs/BEGINNER_MACHINE_LEARNING_GUIDE.md` | Created now | This single-file beginner course, development history and file inventory |
| `docs/dataset_card.md` | Created | Source, feature units, population, data handling and limitations |
| `docs/experiment_notes.md` | Generated | Interpretations tied to the executed results, including negative findings |
| `docs/literature_review.md` | Created | Six primary-source reading entries, comparison questions and gap guidance |
| `docs/methodology.md` | Created | Architecture and detailed technical protocol |
| `docs/presentation.pdf` | Generated | Fourteen supporting slides in PDF format |
| `docs/presentation_outline.md` | Generated | Editable text content of supporting slides |
| `docs/project_brief.md` | Created | Problem, motivation, objectives, proposed contribution and scope |
| `docs/report_template.md` | Created | Scaffold for the student to write an original report |
| `docs/rubric_checklist.md` | Created | Map all 13 rubric items to evidence and student responsibilities |
| `docs/viva_guide.md` | Created | Concept questions, troubleshooting and genuine contribution worksheet |
| `notebooks/diabetes_analysis.ipynb` | Generated/executed | Analysis notebook with eight executed code cells and saved real outputs |
| `web/app.js` | Created/updated | Form handling, JSON requests, predictions, attribution rows, scenarios and results rendering |
| `web/index.html` | Created/updated | Simple form/results layout; later simplified and requested paragraph removed |
| `web/presentation.html` | Generated | Standalone offline slides with embedded images and keyboard navigation |
| `web/styles.css` | Created/updated | Simple responsive styling replacing the earlier decorative frontend |

### Model artifacts, result files, plots, and verification records

| File | Origin | Purpose |
| --- | --- | --- |
| `artifacts/model.joblib` | Generated | Frozen fitted pipeline, voter components, threshold, background and reference ranges |
| `artifacts/splits.json` | Generated | Exact original row indexes for train, validation and test |
| `reports/bootstrap_intervals.json` | Generated | Conditional test metric intervals for the calibrated ensemble |
| `reports/data_quality.csv` | Generated | Training-only missingness, invalid-zero counts, ranges and IQR flags |
| `reports/delivery_verification.json` | Generated | Notebook/server/PDF/JS verification record and browser-testing limitation |
| `reports/feature_ablation.csv` | Generated | Original/engineered/selected comparisons across fixed logistic and ANN learners |
| `reports/figures/calibration.png` | Generated | Test reliability diagram for comparator/raw/calibrated ensembles |
| `reports/figures/confusion_matrices.png` | Generated | Ensemble counts at default and validation-selected thresholds |
| `reports/figures/correlation.png` | Generated | Training Pearson correlation heatmap |
| `reports/figures/feature_distributions.png` | Generated | Training histograms by Outcome class |
| `reports/figures/local_shapley.png` | Generated | One fixed test example attribution plot |
| `reports/figures/outliers.png` | Generated | Training feature boxplots with statistical flags |
| `reports/figures/permutation_importance.png` | Generated | Test ROC-AUC importance drops with repeat SD |
| `reports/figures/quality_and_balance.png` | Generated | Training class counts and missing-value percentages |
| `reports/figures/roc_pr_curves.png` | Generated | Held-out ROC and precision-recall comparisons |
| `reports/figures/shapley_summary.png` | Generated | Mean absolute Shapley contributions for 12 fixed test examples only |
| `reports/figures/uncertainty.png` | Generated | Calibrated test probabilities versus component SD |
| `reports/figures/what_if.png` | Generated | Glucose and BMI model-response curves for a median reference profile |
| `reports/local_explanations.json` | Generated | Exact Shapley values, input values and observed labels for 12 fixed test examples |
| `reports/manifest.json` | Generated/updated | File sizes and SHA-256 fingerprints for managed generated/project folders |
| `reports/permutation_importance.csv` | Generated | Held-out AUC drops and shuffle-repeat SD for each raw feature |
| `reports/results.json` | Generated | Complete results, settings, versions, checksum and stated study limitations |
| `reports/results_appendix.pdf` | Generated | Collected study plots in one figure appendix |
| `reports/statistical_comparison.json` | Generated | Prespecified comparator and paired correctness/AUC calculations |
| `reports/test_metrics_default.csv` | Generated | All model metrics using common threshold 0.50 |
| `reports/test_metrics_validation_threshold.csv` | Generated | Model metrics using each frozen validation threshold |
| `reports/test_predictions.csv` | Generated | Test row IDs, actual labels and predicted probabilities from each model |
| `reports/training.log` | Generated | Console output of the completed training/evaluation run |
| `reports/training_correlations.csv` | Generated | Training-only correlation matrix with zero sentinels treated as missing |
| `reports/training_descriptive_statistics.csv` | Generated | Training-only feature/target summaries |
| `reports/training_warnings.json` | Generated | Recorded nonfatal deprecation warnings from the completed run |
| `reports/tuning_ANN.csv` | Generated | Candidate parameters and complete train/held-out CV scores for ANN |
| `reports/tuning_GradientBoosting.csv` | Generated | Candidate parameters and complete train/held-out CV scores for GradientBoosting |
| `reports/tuning_LogisticRegression.csv` | Generated | Candidate parameters and complete train/held-out CV scores for LogisticRegression |
| `reports/tuning_RandomForest.csv` | Generated | Candidate parameters and complete train/held-out CV scores for RandomForest |
| `reports/tuning_SVM.csv` | Generated | Candidate parameters and complete train/held-out CV scores for SVM |
| `reports/tuning_summary.json` | Generated | Selected settings, mean/std held-out CV scores and training CV scores |
| `reports/uncertainty.csv` | Generated | Test probability, component SD and absolute probability error |
| `reports/validation_thresholds.json` | Generated | Separate thresholds selected using validation F1 |
| `reports/verification.txt` | Generated | Actual recorded regression-suite output |
| `reports/what_if.json` | Generated | Frozen model Glucose/BMI sweeps around a median background profile |

The files depend on one another in this order:

```text
app.py -> src/diabetes/app.py -> predict.py -> explain.py
                    |                 |
              model.joblib       data.py constants

train.py -> src/diabetes/train.py
                    |
          data.py + preprocessing.py + evaluation.py + explain.py
                    |
       model artifact + experimental reports + plots
                    |
         scripts/build_deliverables.py
                    |
      notebook + supporting notes + slides + manifest

scripts/verify.py -> tests/test_core.py -> project functions/artifacts
```

`.gitignore` exists, but this environment's `.git` directory was not an initialized usable repository when inspected. No Git commit, branch, pull request, or remote deployment was created. The ZIP is a package of files, not proof of Git version history.

## 29. Running and inspecting the project yourself

### Step A: open a terminal in the project folder

```bash
cd /home/dog/projects/ANN_DIABITIES
```

On another machine, use the folder where you extracted the project ZIP. You should see `app.py`, `train.py`, `src`, `data`, and `artifacts`.

### Step B: activate/install an environment if needed

The current workspace already had the scientific packages. On a fresh Linux/macOS installation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In Windows PowerShell, the usual activation command is:

```powershell
.venv\Scripts\Activate.ps1
```

Use the recorded versions for loading the existing artifact reliably, or install compatible packages and retrain in the new environment. The saved run's unusually recent package versions may not be available for every older Python interpreter/platform. `requirements-lock.txt` records the environment; it is not a promise that any old interpreter can install it.

### Step C: open the prediction interface

```bash
python app.py
```

Visit `http://127.0.0.1:8000`. Click “Load example,” click “Predict,” expand feature contributions, and try a what-if comparison. Stop the terminal process with Ctrl+C. Use `python app.py --port 8001` if the port is already in use.

### Step D: inspect results without retraining

**Practice:**

```python
import json
from pathlib import Path
r = json.loads(Path('reports/results.json').read_text())
print(r['split_sizes'])
print(r['selected_representation'])
print(r['operating_threshold'])
print(r['operating_test_metrics']['CalibratedEnsemble'])
```

Reading JSON does not modify the model or reports. Use `python` to enter an interactive prompt, paste the practice code, and exit with `exit()` when finished.

### Step E: reproduce a full experiment when you intend to regenerate outputs

```bash
python train.py
python scripts/verify.py
```

Training regenerates/overwrites model artifacts and reports. Preserve a separate snapshot if you need to compare an original run. Do not repeatedly choose configurations based on the same test results.

To regenerate notes, slides, and the notebook on Linux:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/diabetes-matplotlib python scripts/build_deliverables.py
```

Alternatively, install the project package:

```bash
python -m pip install -e .
python scripts/build_deliverables.py
```

`-e` makes an editable package installation, so imports use source from the project. `PYTHONPATH=src` adds the source directory to the import search path for that command. `MPLCONFIGDIR` gives Matplotlib a writable cache directory; it is not a classifier setting.

The builder rewrites generated notes/slides/notebook but does not create this learning guide. Keep your own edited submission separate from generated files.

### Step F: inspect the five selected input names

**Practice — use the import setup from Section 3 first:**

```python
import joblib
bundle = joblib.load('artifacts/model.joblib')
pipe = bundle['components']['ANN']
names = pipe.named_steps['features'].get_feature_names_out()
names = pipe.named_steps['imputer'].get_feature_names_out(names)
names = pipe.named_steps['select'].get_feature_names_out(names)
print(list(names))
```

Each name transformation follows the same order as the data transformation. Looking only at the raw `FEATURES` list would miss the selector's effect.

### Step G: open the notebook or slides

The standalone `web/presentation.html` opens in a browser without a server. It includes embedded plot images and uses arrow keys to switch slides. `docs/presentation.pdf` is the printable version.

Jupyter is optional for the main application. If needed:

```bash
python -m pip install jupyter nbclient nbconvert
jupyter notebook notebooks/diabetes_analysis.ipynb
```

The notebook's path setup accepts being opened from the root or `notebooks/`. The setup assumes this project layout and does not discover arbitrary unrelated folders.

## 30. Practice exercises with explanations

Do these after reading the relevant sections. Keep experimental practice on the main training/validation partitions; do not repeatedly tune on the reserved test set. The exercises below either inspect saved outputs or use independent toy examples.

### Exercise 1: identify features and target

Why is the original table `(768,9)` while `X` is `(768,8)`?

**Answer:** the ninth column is `Outcome`, which belongs to `y`. Giving it to the classifier as an input would leak the answer.

### Exercise 2: understand the split

Why is the second `test_size` 0.25 rather than 0.20?

**Answer:** it operates on the remaining 80%, so a quarter of that is 20% of the original data. Using 0.20 again would reserve only 16% of the original for main validation.

### Exercise 3: mean versus median

Calculate mean and median for `[10,11,12,13,100]`. Which is less affected by the final value?

**Answer:** mean 29.2; median 12. The median is less affected. This does not prove it is optimal for every imputation problem.

### Exercise 4: standard scaling

Training mean is 100 and standard deviation is 20. What does 140 become? May you replace those statistics with statistics from the next user's row?

**Answer:** 2. No; the fitted scaler must keep its training statistics. A single new row must not redefine the preprocessing.

### Exercise 5: class imbalance

There are 90 negatives and 10 positives. A classifier always predicts 0. Calculate accuracy and recall.

**Answer:** 90% accuracy, 0% recall. High accuracy can coexist with total failure to identify positives.

### Exercise 6: a small confusion matrix

**Practice:**

```python
from diabetes.evaluation import metrics
result = metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.2], threshold=0.5)
for key in ['tn','fp','fn','tp','accuracy','precision','recall','f1','specificity','brier']:
    print(key, result[key])
```

**Answer:** all four counts are 1. Accuracy, precision, recall, F1, and specificity are 0.5. Brier is 0.345. This is a toy arithmetic check, not a new project experiment.

### Exercise 7: distinguish probability, threshold, and accuracy

A row has `p=0.40`. Its threshold is 0.33. What is its predicted label? Does that tell you accuracy?

**Answer:** label 1. Accuracy requires many predictions and their true labels; one probability is not the model's accuracy.

### Exercise 8: work through one learning update

For `x=1`, `y=1`, `w=0`, and `b=0`, repeat the gradient update from Section 17 with learning rate 0.1. Why does the probability increase?

**Answer:** both gradients are −0.5, so subtracting them increases weight and bias to 0.05. The score rises to 0.1 and sigmoid to about 0.525. Loss falls because the probability moves toward the true label.

### Exercise 9: count ANN parameters

If a network has 5 inputs, 32 hidden neurons, and 1 output, why is its parameter count 225 rather than `5×32×1`?

**Answer:** count weights for each connection layer and biases separately: `5×32 + 32 + 32×1 + 1 = 225`.

### Exercise 10: compare voting and calibration

Average `[0.2,0.3,0.4,0.5,0.6]`. If a calibrator maps that mean to 0.35, which value does the deployed threshold use?

**Answer:** raw average 0.4; the deployed calibrated model uses 0.35. Calibration is an additional learned mapping after combining probabilities.

### Exercise 11: recognize leakage

You compute the median insulin using all rows before fitting CV pipelines. Is fitting a different classifier in each fold enough to fix this?

**Answer:** no. The preprocessing already saw held-out rows. Fit the imputer inside each fold's training pipeline.

### Exercise 12: interpret a zero contribution

Pregnancies has a zero Shapley contribution in the saved example. Does that prove pregnancy information is useless in every diabetes model?

**Answer:** no. This fitted selected-five representation excludes that raw column. Its local contribution is a property of this model and value function.

### Exercise 13: explain statistical uncertainty

The ensemble AUC interval is roughly 0.736–0.876. Does that mean a user's probability is inside that interval?

**Answer:** no. It is a test-performance interval for a fixed model under a resampling procedure, not an individual prediction interval.

### Exercise 14: identify file responsibilities

Which file would you change for button color? For input validation? For candidate ANN widths?

**Answer:** `web/styles.css`; `src/diabetes/predict.py`; `model_specs` in `src/diabetes/train.py`. The last change requires a newly fitted artifact and fresh experiment, not just a page refresh.

### Exercise 15: run a small practice model on the saved training/validation split

**Practice — this creates an in-memory model and does not overwrite artifacts:**

```python
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from diabetes.data import load_data, FEATURES, TARGET
from diabetes.preprocessing import pipeline

frame = load_data()
splits = json.loads(Path('artifacts/splits.json').read_text())
training = frame.loc[splits['train']]
validation = frame.loc[splits['validation']]
practice_model = pipeline(LogisticRegression(max_iter=2000), k='all')
practice_model.fit(training[FEATURES], training[TARGET])
p = practice_model.predict_proba(validation[FEATURES])[:, 1]
print('Practice validation AUC:', roc_auc_score(validation[TARGET], p))
```

**Explanation:** the pipeline protects preprocessing, and this exercise measures on validation, not repeatedly on the reserved test. It does not reproduce the tuned logistic settings or the calibrated deployment bundle. Different configuration means a different model.

### Exercise 16: explain the negative findings out loud

Why is it acceptable that the engineered representation and calibrated ensemble did not win every metric?

**Answer:** their benefit was an experimental hypothesis. A correct comparison can refute that hypothesis on a sample. Reporting it is more informative than changing the procedure until a favorable number appears.

### Exercise 17: give a two-minute project explanation

Use this sequence without memorizing the exact wording:

1. What the target and source dataset are.
2. Why rows were separated into three partitions.
3. How preprocessing stays inside CV pipelines.
4. Which five families were tuned and what the ANN is.
5. How the voter, calibrator, and threshold connect.
6. What the actual holdout shows, including a negative finding.
7. What the explanations mean and cannot prove.
8. How the saved pipeline is loaded by the website.

If you can explain those points in your own words and locate their code, you understand the project's main structure.

## 31. Troubleshooting and common mistakes

| Situation | What to check / do |
|---|---|
| `ModuleNotFoundError: diabetes` in a practice snippet | Run from the project root and use the import setup in Section 3, or install the package with `pip install -e .` |
| Scientific library import fails | Activate the intended environment and install its requirements |
| Saved artifact fails to load in another environment | Compare Python/scikit-learn versions and retrain in the intended supported stack |
| `model.joblib` is missing | Run `python train.py` before starting the application |
| Port already in use | Use `python app.py --port 8001` and visit the matching URL |
| Website is unreachable | Confirm the server is running and the browser can reach that machine's localhost/forwarded port |
| Explanation is slower than a plain prediction | It intentionally evaluates 256 coalitions over 24 background rows |
| Age input is rejected | This form's scope requires an integer age of at least 21 |
| Blank insulin works but blank age fails | Only documented laboratory fields permit missing input |
| Changing Pregnancies does not change this saved prediction | It was excluded by the selected-five pipeline |
| Chart still looks like the original research palette | Saved experimental plots are separate from the simplified frontend CSS |
| Page looks old after an HTML/CSS edit | Refresh; if needed use a hard reload; static content is read from disk |
| Python logic change is not reflected in an already-running server | Restart the Python process |
| Matplotlib mentions a non-writable cache directory | Use a writable `MPLCONFIGDIR`; it concerns plotting infrastructure |
| Training warnings mention deprecated APIs | Read `training_warnings.json`; distinguish future migration work from actual execution failure |
| AUC is mistaken for accuracy | AUC evaluates ranking; accuracy evaluates hard labels at a threshold |
| CV mean is mistaken for independent test performance | CV was used for selection; the final test is separate |
| Precision is mistaken for recall | Precision conditions on predicted positives; recall conditions on actual positives |
| Brier is called a pure calibration measure | It measures overall squared probability error, with multiple contributing properties |
| Explanation is called proof of causation | It describes the model and chosen background, not causal effects |
| Low disagreement is called high certainty | Correlated models can agree on a wrong answer |
| Report copies generated prose as an original submission | Follow the supplied rubric and write the submission in your own words |

To troubleshoot well, find the layer where the problem occurs: data loading, pipeline fitting, inference, HTTP request, or display. A button-color issue does not require changing neural-network hyperparameters. A schema error does not require changing the optimizer.

## 32. Glossary and a learning schedule

### Compact glossary

| Term | Meaning |
|---|---|
| Observation / sample | One row of data |
| Feature / predictor | Input variable |
| Target / label | Outcome used to train/evaluate a supervised model |
| Classification | Predicting a category |
| Regression | Predicting a numerical target |
| Estimator | Object with a fitting interface |
| Fitted state | Values learned by `fit` |
| Training | Learning parameters/statistics from data |
| Inference | Applying a fitted model to inputs |
| Generalization | Performance beyond the fitting examples |
| Leakage | Held-out information influences fitting/selection |
| Imputation | Filling missing values using a specified rule |
| Sentinel | Special recorded value representing something else, such as missingness |
| Indicator | Often a binary feature recording a condition |
| Scaling | Transforming numeric magnitudes using a chosen rule |
| Encoding | Turning categorical values into machine-usable representations |
| One-hot encoding | Separate 0/1 columns for categories |
| Pipeline | Ordered combined preprocessing and estimator |
| Feature engineering | Creating alternative/derived predictors |
| Feature selection | Retaining a subset of predictors |
| Ablation | Controlled comparison of components/representations |
| Parameter | Quantity learned during fitting |
| Hyperparameter | Configuration selected outside a fitting run |
| Loss | Quantity minimized during training |
| Metric | Quantity used to evaluate predictions |
| Optimizer | Procedure for updating parameters |
| Gradient | Sensitivity of loss to a parameter |
| Backpropagation | Chain-rule calculation of network gradients |
| Activation | Nonlinear function applied at a neuron/layer |
| ReLU | `max(0,x)` |
| Sigmoid | Function mapping a score into `(0,1)` |
| Epoch | Pass through training data for stochastic optimization |
| Mini-batch | Subset used for an optimizer update |
| Regularization | Constraint/penalty against excessive complexity |
| Early stopping | Stop fitting based on lack of internal improvement |
| Overfitting | Fits sample peculiarities and generalizes poorly |
| Underfitting | Fails to learn sufficient useful structure |
| Cross-validation | Repeated fitting/held-out evaluation within available training data |
| Fold | One partition within CV |
| Out-of-fold score | Prediction for a row from a model not fitted on that row |
| Ensemble | Combination of multiple learners |
| Calibration | Agreement of probability magnitudes with outcome frequency |
| Threshold | Probability/score cutoff for a hard decision |
| Confusion matrix | Counts of truth/prediction combinations |
| ROC-AUC | Summary of binary ranking across thresholds |
| Average precision | Precision weighted by increases in recall |
| Brier score | Mean squared probability error |
| Bootstrap | Resampling with replacement |
| Confidence interval | Procedure-based expression of estimation uncertainty |
| Paired comparison | Comparison respecting predictions on the same observations |
| Shapley value | Weighted marginal contribution across feature coalitions |
| Background | Reference records used for the explanation value function |
| Extrapolation | Applying a model outside observed support |
| Distribution shift | Application data differs from the training distribution |
| Artifact | Persisted object/output, such as a fitted model |
| Serialization | Converting object state into a storable representation |
| API | Programmatic interface for requests/responses |
| HTTP | Request/response protocol used by the local web application |
| JSON | Structured text exchange format |
| Checksum | Fingerprint of file bytes |
| Reproducibility | Enough recorded detail to repeat/audit a procedure |

### Why no categorical encoder appears in this pipeline

All eight original predictors are numeric. Therefore a one-hot encoder is not needed. If a later dataset contains categories such as clinic names, integer labels can incorrectly imply order/distance; an appropriate encoder and column-specific preprocessing would be needed. That is a future design change, not a missing step in this numeric-only dataset.

The binary target values 0 and 1 are labels. They are not evidence that every categorical input should be treated as a numerical measurement.

### A suggested seven-session path

| Session | Read / do | Understanding to demonstrate |
|---|---|---|
| 1 | Sections 1–6; inspect shape, columns, class counts | Describe an observation, feature, target, and loader |
| 2 | Sections 7–10; work scaling/imputation examples | Explain three partitions and fold-local preprocessing |
| 3 | Sections 11–16; compare model families | Distinguish engineering, selection, linear models, forests, boosting, and SVM |
| 4 | Section 17; calculate forward pass/update/parameter count | Explain ANN weights, activations, loss, and optimization |
| 5 | Sections 18–22; work confusion metrics | Explain CV, search, calibration, thresholds, and interval limits |
| 6 | Sections 23–27; inspect the real example and website flow | Explain Shapley, disagreement, scenarios, artifacts, API, and tests |
| 7 | Sections 28–31; run the app, tests, and exercises | Trace files and give an accurate two-minute project explanation |

Go more slowly if needed. Solve one example by hand before relying on its library function. Being able to run a script and being able to explain the experiment are different achievements.

## 33. Further topics and primary references

This project covers a complete small supervised tabular-classification workflow, not every branch of ML. Natural next topics are linear regression, additional classification models, categorical preprocessing, learning curves, nested CV, group/time-based validation, class-imbalance strategies, external validation, causal inference, fairness evaluation, clustering, dimensionality reduction, CNNs, sequence models, transformers, and reinforcement learning. These topics are **not** all implemented here; do not claim them in the viva as completed components.

When extending this work, change one experimental question at a time, decide the evaluation protocol first, and preserve an untouched assessment dataset. Do not use this same test partition indefinitely to choose every future improvement.

Useful primary references linked near the concepts above:

- [Preserved Pima metadata](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names): feature/population provenance.
- [Scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): leakage and preprocessing consistency.
- [StandardScaler API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): fitted scaling behavior.
- [ANOVA feature scoring](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html): selection criterion.
- [Neural-network overview](https://scikit-learn.org/stable/modules/neural_networks_supervised.html) and [MLPClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html): model and settings.
- [Calibration guide](https://scikit-learn.org/stable/modules/calibration.html): probability mapping and CV behavior.
- [Evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html) and [average precision API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html): probability/ranking metrics.
- [Guo et al., 2017, calibration paper](https://proceedings.mlr.press/v70/guo17a.html): research context; this project does not reproduce temperature scaling.
- [Lundberg & Lee, 2017, SHAP framework](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html): additive feature attribution.
- [Tasin et al., 2023, diabetes classification and XAI](https://pmc.ncbi.nlm.nih.gov/articles/PMC10107388/): related applied study.
- [Hasan et al., institutional publication record](https://researchportal.northumbria.ac.uk/en/publications/towards-transparent-diabetes-prediction-combining-automl-and-expl/): established integration of AutoML and explanations.

The live library documentation can change. The local code and `requirements-lock.txt` identify the exact included experiment; compare versions when learning from current documentation.

Your main study goal is to explain, in your own words, the chain from **data → separate partitions → fold-local preprocessing → learned models → measured probabilities and decisions → explanations → application**. Every step has an implementation, a purpose, and a limitation that you can now inspect.
