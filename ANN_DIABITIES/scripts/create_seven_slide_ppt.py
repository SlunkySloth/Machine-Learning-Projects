"""Build the seven-slide project deck from the recorded experiment."""
import json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / 'reports/results.json').read_text())
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
NAVY='11263B'; TEAL='007F82'; MINT='DFF3EF'; WHITE='FFFFFF'; INK='173247'; GRAY='5B6E7C'; BG='F5F8FA'; GOLD='EDB655'

def box(s,x,y,w,h,color,radius=False):
    sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb=RGBColor.from_string(color); sh.line.fill.background()
    return sh

def text(s,x,y,w,h,content,size=20,color=INK,bold=False):
    sh=s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf=sh.text_frame; tf.word_wrap=True
    tf.margin_left=tf.margin_right=Inches(.02)
    tf.margin_top=tf.margin_bottom=Inches(.02)
    for i,line in enumerate(content.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.text=line; p.font.name='Liberation Sans'; p.font.size=Pt(size); p.font.bold=bold; p.font.color.rgb=RGBColor.from_string(color)
        p.space_after=Pt(9)
    return sh

def slide(n,title,kicker,source,notes):
    s=prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb=RGBColor.from_string(BG)
    box(s,0,0,.18,7.5,TEAL)
    text(s,.6,.35,12,.35,kicker.upper(),12,TEAL,True)
    text(s,.6,.9,12, .8,title,32,NAVY,True)
    box(s,.6,6.94,12.1,.015,'D9E1E7')
    text(s,.6,7.05,11.2,.25,source,10,GRAY)
    text(s,12.1,7.03,.7,.3,f'{n:02d} / 07',10,GRAY)
    s.notes_slide.notes_text_frame.text=notes
    return s

def card(s,x,y,w,h,label,body,size=20):
    box(s,x,y,w,h,WHITE,True)
    text(s,x+.22,y+.2,w-.44,.45,label,22,TEAL,True)
    text(s,x+.22,y+.85,w-.44,h-.95,body,size)

# 1
s=slide(1,'Diabetes Prediction Using ANN','Machine learning project','ANN_DIABITIES • Project overview',
    'Introduce the project as classification of the historical Pima diabetes Outcome label. The ANN is the main studied neural model, and the application deploys an ensemble of five models. This is an academic demonstrator; it is not validated for diagnosis or future onset. Add your own name, register number and institution in the editable identity field before presenting.')
box(s,.6,2,7.6,3.95,NAVY,True)
text(s,.95,2.4,6.85,1.55,'From health measurements\nto an explainable prediction',34,WHITE,True)
text(s,.95,4.25,6.8,1.2,'Artificial neural network • Model comparison\nCalibrated ensemble • Local prediction app',22,'C8E0E5')
for yy,val,label in [(2.05,'768','Historical records'),(3.35,'8','Numeric input features'),(4.65,'5','Models in the ensemble')]:
    text(s,8.8,yy,3.6,.65,val,36,TEAL,True); text(s,8.8,yy+.67,3.6,.4,label,18)
text(s,.7,6.22,11.8,.4,'Presented by: [Your name]  |  [Register number]  |  [Department / Institution]',16,GRAY)

# 2
s=slide(2,'Problem statement and objectives','Why this project','Source: docs/project_brief.md; docs/methodology.md',
    'The task is to estimate the existing binary Outcome label from eight numeric measurements. The study evaluates whether a small ANN can learn useful patterns and how its performance compares with conventional models. Explainability and a working interface are project objectives, rather than proof of clinical usefulness.')
text(s,.65,1.92,11.9,.9,'Can an ANN classify diabetes outcomes from health measurements\nand explain its predictions in a usable application?',25,NAVY,True)
card(s,.65,3.1,3.88,2.65,'01  Build','Clean the data and train an ANN with a reproducible pipeline.')
card(s,4.73,3.1,3.88,2.65,'02  Compare','Evaluate four alternative learners and a calibrated ensemble.')
card(s,8.81,3.1,3.88,2.65,'03  Explain','Show feature contributions, model disagreement and what-if results.')
text(s,.7,6.12,11.8,.5,'Scope: academic outcome classification using a historical dataset.',20,TEAL,True)

# 3
s=slide(3,'Dataset and preprocessing','Data foundation','Source: docs/dataset_card.md; reports/results.json',
    'The Pima dataset contains 768 historical records of women of Pima heritage aged 21 or older. There are 500 negative and 268 positive outcomes. Glucose is a two-hour oral glucose tolerance test measurement. Zero is valid for pregnancies, but zero is treated as missing for glucose, blood pressure, skin thickness, insulin and BMI. Median imputation, missingness indicators, scaling and feature selection are fitted within training folds to avoid leakage. Five transformed predictors are selected using fold-local ANOVA scoring.')
card(s,.65,1.95,5.9,3.45,'Eight input features','Pregnancies • Glucose • Blood pressure\nSkin thickness • Insulin • BMI\nDiabetes pedigree score • Age')
text(s,.9,4.68,5.3,.55,'Target: Outcome = 0 or 1',20,TEAL,True)
text(s,6.9,1.98,5.6,.5,'Class distribution',22,NAVY,True)
box(s,6.95,2.75,5.55*500/768,.5,TEAL); box(s,6.95+5.55*500/768,2.75,5.55*268/768,.5,GOLD)
text(s,6.95,3.4,5.55,.5,'500 negative  |  268 positive',20)
text(s,6.95,4.2,5.5,1.8,'Zero handling → Median imputation\nMissingness indicators → Standard scaling\nTraining-CV selection → 5 predictors',20)
box(s,.65,5.75,12.05,.75,MINT,True)
text(s,.9,5.92,11.5,.5,'Stratified split: 460 train  /  154 validation  /  154 test  •  Seed 42',21,TEAL,True)

# 4
s=slide(4,'ANN architecture and training','Core model','Source: reports/tuning_summary.json; docs/methodology.md',
    'The ANN is a scikit-learn MLPClassifier, not a TensorFlow model. Its selected hidden architecture has one layer with 32 ReLU units and a sigmoid binary output. Adam uses a learning rate of 0.003 and L2 alpha of 0.001; early stopping uses an internal subset of training-fold data. Six randomized parameter configurations are evaluated in five stratified folds per model, using ROC-AUC. The ANN has the highest selected training-CV AUC of 0.8214, but tuned CV is selection-biased; its test AUC is 0.7956. The diagram is schematic and does not show all 32 hidden units.')
for x,label,sub in [(1.05,'5 selected inputs','After preprocessing'),(5.05,'32 hidden units','ReLU activation'),(9.05,'1 output','Sigmoid probability')]:
    box(s,x,2.15,3.15,1.45,NAVY if x==5.05 else TEAL,True)
    text(s,x+.2,2.45,2.75,.5,label,24,WHITE,True)
    text(s,x+.2,3.05,2.75,.4,sub,16,WHITE)
for x in [4.35,8.35]: text(s,x,2.6,.65,.5,'→',28,TEAL,True)
card(s,.65,4.05,5.9,2.2,'Selected configuration','Adam optimizer • Learning rate 0.003\nL2 regularization 0.001 • Early stopping')
card(s,6.75,4.05,5.95,2.2,'Training and tuning','5-fold stratified CV • 6 candidates/model\nBest ANN training-CV ROC-AUC: 0.821')

# 5
s=slide(5,'Prediction framework and application','End-to-end workflow','Source: docs/methodology.md; src/diabetes/predict.py; web/',
    'The application loads a frozen local model artifact. Its five components are ANN, logistic regression, random forest, gradient boosting and SVM. Positive-class probabilities are averaged equally, then sigmoid-calibrated using out-of-fold training predictions. Validation chooses the F1 threshold 0.33; the test set is evaluated only after freezing. Exact Shapley contributions explain the full pipeline against 24 training background examples. Model disagreement is the standard deviation of component probabilities, not a confidence interval. What-if changes describe model associations, not treatment effects.')
text(s,.7,1.95,11.8,.65,'ANN + Logistic Regression + Random Forest + Gradient Boosting + SVM',22,NAVY,True)
for x,w,label in [(.7,3.55,'Equal-weight voting'),(4.85,3.05,'Sigmoid calibration'),(8.55,3.95,'Threshold = 0.33')]:
    box(s,x,2.95,w,1,TEAL,True); text(s,x+.15,3.24,w-.3,.5,label,20,WHITE,True)
text(s,4.35,3.2,.5,.5,'→',25,TEAL,True); text(s,8,3.2,.5,.5,'→',25,TEAL,True)
card(s,.7,4.45,5.8,1.95,'Prediction interface','Enter 8 features → Probability and class\nMissing-value and extrapolation notes')
card(s,6.75,4.45,5.8,1.95,'Interpretation tools','Exact Shapley feature contributions\nComponent disagreement • What-if curves')

# 6
s=slide(6,'Measured results on the held-out test set','Evaluation • 154 records','Source: reports/test_metrics_*.csv; reports/bootstrap_intervals.json',
    'Left: test ROC-AUC measures ranking and is independent of decision threshold. Gradient boosting has the highest test AUC at 0.8222; the deployed calibrated ensemble is 0.8107, random forest 0.8215, SVM 0.8004, ANN 0.7956, logistic regression 0.7826, and dummy prior 0.5. Right: ensemble metrics use validation-selected threshold 0.33, not 0.50. At 0.33 there are 72 true negatives, 28 false positives, 16 false negatives and 38 true positives. Recall rises from 55.6% at threshold 0.50 to 70.4% at 0.33, while specificity falls from 82.0% to 72.0%. The AUC interval 0.736–0.876 uses 1,000 stratified test bootstrap samples and conditions on the fitted model and observed prevalence.')
text(s,.7,1.88,6,.45,'Test ROC-AUC by model',21,NAVY,True)
names=[('GradientBoosting','Gradient boosting'),('RandomForest','Random forest'),('CalibratedEnsemble','Calibrated ensemble'),('SVM','SVM'),('ANN','ANN'),('LogisticRegression','Logistic regression'),('DummyPrior','Dummy baseline')]
for i,(key,label) in enumerate(names):
    y=2.55+i*.49; auc=R['default_test_metrics'][key]['roc_auc']
    text(s,.7,y,2.7,.38,label,16,TEAL if key=='CalibratedEnsemble' else INK,key=='CalibratedEnsemble')
    box(s,3.5,y+.04,2.9,.23,'E1E8ED'); box(s,3.5,y+.04,2.9*auc,.23,TEAL if key=='CalibratedEnsemble' else '8199A8')
    text(s,6.55,y,.65,.38,f'{auc:.3f}',16,INK,True)
box(s,7.45,1.95,5.15,4.43,NAVY,True)
text(s,7.75,2.2,4.55,.65,'Deployed ensemble\nValidation threshold: 0.33',19,WHITE,True)
for x,y,v,l in [(7.8,3.1,'71.4%','Accuracy'),(10.15,3.1,'70.4%','Recall'),(7.8,4.25,'0.633','F1 score'),(10.15,4.25,'72.0%','Specificity')]:
    text(s,x,y,2.1,.6,v,30,'80DDD1',True); text(s,x,y+.65,2.1,.35,l,16,WHITE)
text(s,7.8,5.6,4.55,.55,'AUC 95% interval: 0.736–0.876',17,WHITE)
text(s,.7,6.4,11.9,.38,'Gradient boosting achieved the highest test AUC; the ensemble was the prespecified deployed framework.',16,TEAL,True)

# 7
s=slide(7,'Conclusion and future work','What the study demonstrates','Source: docs/experiment_notes.md; reports/results.json',
    'The study completes an ANN training pipeline, comparative evaluation, ensemble and explainable local application. It does not show that the ensemble is best: gradient boosting has higher test AUC. Calibration reduced ten-bin ECE from 0.1245 to 0.0863, but worsened Brier score from 0.1709 to 0.1746. The main limitations are small historical single-population data, missing measurements, no external validation and non-causal explanations. Future work should evaluate an independent and more diverse cohort, assess subgroup performance, compare calibration alternatives on validation data and choose operating thresholds with domain input. Supporting data source: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv. Revise presentation wording and add personal identity details to match institutional submission requirements.')
card(s,.65,1.95,5.9,2.0,'Completed','ANN training and model comparison\nPrediction app with explanations',18)
card(s,6.75,1.95,5.95,2.0,'Key finding','Useful predictive ranking, with trade-offs\nCalibration worsened the Brier score',18)
card(s,.65,4.2,5.9,2.1,'Limitations','Historical cohort; missing measurements\nNo external validation; non-causal explanations',18)
card(s,6.75,4.2,5.95,2.1,'Next steps','Test on diverse independent data\nAssess fairness, calibration and thresholds',18)
text(s,.75,6.47,11.9,.35,'Academic demonstrator • Outcome classification • Not validated for clinical diagnosis',16,TEAL,True)

prs.core_properties.title='Diabetes Prediction Using ANN — 7 Slides'
prs.core_properties.subject='Measured project methodology and results'
prs.core_properties.author='ANN_DIABITIES project'
out=ROOT/'docs/Diabetes_Prediction_7_Slides.pptx'
prs.save(out)
assert len(prs.slides)==7
print(out)
