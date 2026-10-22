# Dataset card

Name: Pima Indians Diabetes Database. Original owner listed in the historical metadata: National Institute of Diabetes and Digestive and Kidney Diseases. The dataset was historically donated to UCI; this project downloads the preserved headerless CSV from James Brownlee's public mirror, not from a current UCI API.

[CSV mirror](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) · [Preserved source metadata](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names). Consult the mirror/source terms before redistribution; this project does not invent a dataset license.

The bundled file has 768 rows: 500 Outcome=0 and 268 Outcome=1, eight numeric predictors and one binary target. Historical metadata restricts the population to women of Pima heritage aged at least 21. Selection, historical collection, small size, incomplete measurements and lack of an external cohort restrict generalization.

| Feature | Meaning / source measurement | Preprocessing |
|---|---|---|
| Pregnancies | Number of times pregnant | Zero is valid; integer input |
| Glucose | Plasma glucose, two-hour oral glucose tolerance test; mg/dL | Zero/blank becomes missing |
| BloodPressure | Diastolic blood pressure; mmHg | Zero/blank becomes missing |
| SkinThickness | Triceps skinfold thickness; mm | Zero/blank becomes missing |
| Insulin | Two-hour serum insulin; µU/mL | Zero/blank becomes missing |
| BMI | Weight divided by squared height; kg/m² | Zero/blank becomes missing |
| DiabetesPedigreeFunction | Dataset-specific diabetes pedigree score | Numeric; not a yes/no family-history answer |
| Age | Years | Integer; source population age ≥21 |
| Outcome | Historical binary diabetes class; 1 positive, 0 negative | Target only; never an input |

Feature units and population are based on the [preserved metadata](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names). Modern fasting-glucose diagnostic cutoffs must not be substituted for this dataset's measurement protocol.

Training-only summaries and missing-value counts are in `reports/data_quality.csv`, `reports/training_descriptive_statistics.csv` and `reports/training_correlations.csv`. The dataset checksum, versions, seed and actual split sizes are recorded in `reports/results.json`. The raw CSV stays unchanged. No generated or invented patient records are used in training.
