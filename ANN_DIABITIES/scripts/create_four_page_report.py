"""Create an editable four-page report using standard-library OOXML."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from html import escape
import json

ROOT=Path(__file__).resolve().parents[1]
R=json.loads((ROOT/'reports/results.json').read_text())
NS='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
parts=[]

def para(value, size=21, bold=False, color='223647', before=0, after=100, keep=False):
    props=f'<w:spacing w:before="{before}" w:after="{after}" w:line="265" w:lineRule="auto"/>'
    if keep: props+='<w:keepNext/>'
    run=f'<w:rPr><w:rFonts w:ascii="Liberation Sans" w:hAnsi="Liberation Sans"/><w:sz w:val="{size}"/><w:color w:val="{color}"/>'+('<w:b/>' if bold else '')+'</w:rPr>'
    return f'<w:p><w:pPr>{props}</w:pPr><w:r>{run}<w:t xml:space="preserve">{escape(value)}</w:t></w:r></w:p>'

def p(value, **kwargs): parts.append(para(value,**kwargs))
def h(value): p(value,size=26,bold=True,color='007F82',before=160,after=100,keep=True)
def page(): parts.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
def methodology_image():
    cx,cy=6264910,2945440
    parts.append(f'''<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="80" w:after="60"/></w:pPr><w:r><w:drawing><wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="1" name="Methodology schematic" descr="Training, validation and test workflow for the ANN diabetes project"/><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="1" name="methodology_schematic.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="rIdDiagram"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>''')
def table(rows,widths):
    xml='<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:tblLayout w:type="fixed"/><w:tblBorders>'
    for edge in ['top','left','bottom','right','insideH','insideV']:
        xml+=f'<w:{edge} w:val="single" w:sz="4" w:color="D6E1E6"/>'
    xml+='</w:tblBorders><w:tblCellMar><w:top w:w="70" w:type="dxa"/><w:bottom w:w="70" w:type="dxa"/><w:left w:w="100" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>'
    xml+=''.join(f'<w:gridCol w:w="{w}"/>' for w in widths)+'</w:tblGrid>'
    for i,row in enumerate(rows):
        xml+='<w:tr><w:trPr><w:cantSplit/>'+('<w:tblHeader/>' if i==0 else '')+'</w:trPr>'
        for cell,w in zip(row,widths):
            bg='11263B' if i==0 else ('F0F6F7' if i%2 else 'FFFFFF')
            xml+=f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:shd w:fill="{bg}"/></w:tcPr>'+para(str(cell),size=19,bold=i==0,color='FFFFFF' if i==0 else '223647',after=0)+'</w:tc>'
        xml+='</w:tr>'
    parts.append(xml+'</w:tbl>')

p('DIABETES PREDICTION USING ANN',size=36,bold=True,color='11263B',after=110)
p('An explainable machine learning classification project',size=24,color='007F82',after=170)
p('Name: [Your name]   |   Register number: [Your register number]',size=19)
p('Department / Institution: [Your details]   |   Academic year: 2026–2027',size=19)
p('AI-assisted technical draft — revise in your own words for institutional submission.',size=18,color='5B6E7C',after=160)
h('Abstract')
p('This project classifies the historical diabetes Outcome label in the Pima Indians Diabetes Database using an artificial neural network (ANN) and four comparison models. The 768 records contain eight numeric inputs. Stratified training, validation and test partitions separate model fitting, threshold selection and evaluation. Fold-local imputation, scaling and feature selection reduce information leakage. A tuned multilayer perceptron is compared with logistic regression, random forest, gradient boosting and a support vector machine. An equal-weight ensemble is sigmoid-calibrated and deployed in a local application with feature explanations. On 154 held-out records, the calibrated ensemble achieved ROC-AUC 0.811, accuracy 71.4%, recall 70.4% and F1 0.633 at validation-selected threshold 0.33. Gradient boosting achieved a higher test AUC of 0.822. The study demonstrates a reproducible educational workflow, with limitations from population scope, missing measurements and lack of external validation.')
h('1. Introduction and objectives')
p('The problem is binary classification of an existing dataset label from health measurements. The objectives are to audit data quality, train an ANN, compare alternative learners, measure discrimination and probability quality, and provide an understandable prediction interface. The contribution is the integration and evaluation of established methods. The output is neither a validated forecast of future diabetes onset nor a clinical diagnosis.')
h('2. Dataset and source')
p('The data was obtained from James Brownlee’s public mirror of the Pima benchmark; preserved metadata attributes the original data to NIDDK [1]. It describes women of Pima heritage aged 21 or older, with 500 negative and 268 positive outcomes. The original file is data/pima_raw.csv; data/pima_with_headers.csv adds column names without changing any records.')
table([['Inputs','Measurement / meaning'],['Pregnancies; Glucose','Pregnancy count; two-hour OGTT glucose (mg/dL)'],['BloodPressure; SkinThickness','Diastolic pressure (mmHg); triceps skinfold (mm)'],['Insulin; BMI','Two-hour insulin (µU/mL); body mass index (kg/m²)'],['DiabetesPedigreeFunction; Age','Dataset-specific pedigree score; age in years']], [3700,6166])

page()
h('3. Preprocessing and experimental design')
p('Schema, label, numeric and duplicate-record checks precede the seeded stratified split: 460 training, 154 validation and 154 test records. Zero pregnancies is valid; zero glucose, blood pressure, skin thickness, insulin and BMI are treated as missing. Median imputation, missingness indicators, scaling and selection are fitted within training folds. IQR outliers are retained. Training invalid-zero rates were 46.74% for insulin and 28.48% for skin thickness.')
p('Original, engineered and ANOVA-selected representations were compared using identical five-fold training partitions. Mean AUC across fixed logistic and ANN learners selected five predictors. Log transforms and a glucose–BMI interaction did not win the ablation.')
h('4. ANN architecture and comparison models')
p('The scikit-learn MLPClassifier [2] has five preprocessed inputs, 32 ReLU hidden units and a sigmoid output. Adam uses learning rate 0.003, L2 alpha 0.001 and early stopping. Logistic regression, random forest, histogram gradient boosting and RBF SVM provide comparisons. Six randomized configurations per model use five-fold stratified CV and ROC-AUC. The ANN’s selected training-CV AUC of 0.8214 fixed it as the comparator; tuned CV scores remain selection-biased.')
h('5. Ensemble, calibration and decision threshold')
p('Five probabilities are averaged equally, then sigmoid-calibrated using five-fold out-of-fold training predictions [3]. Hyperparameters are selected before calibration; the voter is refitted on training rows. Validation selects the F1 threshold 0.33 from 0.10–0.90 in 0.01 steps. The calibrated model and threshold are frozen before held-out evaluation.')
methodology_image()
p('Figure 1. Methodology schematic: separate training, validation and test paths; the frozen model also powers the explainable local application.',size=18,color='5B6E7C',after=0)

page()
h('6. Results and interpretation')
p('Table 1 compares held-out performance at the common threshold 0.50. ROC-AUC measures ranking; accuracy and recall depend on the operating threshold. All values come from the recorded experiment, not from illustrative or synthetic results.')
rows=[['Model','ROC-AUC','Accuracy','Recall']]
for key,name in [('ANN','ANN'),('LogisticRegression','Logistic regression'),('RandomForest','Random forest'),('GradientBoosting','Gradient boosting'),('SVM','SVM'),('RawEnsemble','Raw ensemble'),('CalibratedEnsemble','Calibrated ensemble'),('DummyPrior','Dummy prior')]:
    m=R['default_test_metrics'][key]
    rows.append([name,f"{m['roc_auc']:.3f}",f"{m['accuracy']:.1%}",f"{m['recall']:.1%}"])
table(rows,[4066,1800,2000,2000])
p('Table 1. Test comparison at threshold 0.50; n = 154.',size=18,color='5B6E7C',before=70)
p('Gradient boosting obtained the highest test ROC-AUC (0.822), followed by random forest (0.821). The ANN achieved 0.796, while the calibrated ensemble achieved 0.811. The ensemble was the prespecified deployed framework; these findings do not support describing it as the best test classifier.')
h('6.1 Deployed operating point: threshold 0.33')
table([['Accuracy','Precision','Recall','F1','Specificity'],['71.4%','57.6%','70.4%','0.633','72.0%']], [1973,1973,1973,1973,1974])
p('The corresponding confusion counts are 72 true negatives, 28 false positives, 16 false negatives and 38 true positives. Lowering the ensemble threshold from 0.50 to 0.33 increased recall from 55.6% to 70.4%, while reducing specificity from 82.0% to 72.0%. This exposes the trade-off between detecting more positive cases and generating additional false positives.',before=110)
h('6.2 Calibration and uncertainty')
p('Sigmoid calibration reduced ten-bin expected calibration error from 0.1245 to 0.0863, but worsened the Brier score from 0.1709 to 0.1746 (lower is better). Therefore, the study does not claim that calibration improved probability quality overall. ROC-AUC remained unchanged under the monotonic calibration mapping.')
p('Using 1,000 stratified bootstrap resamples, the ensemble ROC-AUC 95% interval was 0.736–0.876. This interval conditions on the fixed model and observed prevalence; it excludes training and tuning uncertainty. At threshold 0.50, exact McNemar comparison with the ANN gave p = 1.000, with two discordant errors in each direction. This does not prove equivalence. The paired ensemble-minus-ANN AUC difference was 0.0152, with conditional interval 0.0006–0.0320; it does not establish clinical superiority.')

page()
h('7. Explainability and application')
p('Exact finite-background interventional Shapley values explain the full deployed pipeline using all 256 coalitions of the eight raw inputs and 24 sampled training background records. Explanations were recorded for 12 fixed test cases. The maximum additivity error was 2.22 × 10⁻¹⁶, confirming that the baseline plus contributions reproduced the model probability. The mean attribution summary describes only these examples. Background substitution can create unrealistic feature combinations, so attribution is descriptive rather than causal.')
p('Permutation importance measures changes in test AUC after shuffling a feature. The standard deviation of five component probabilities reports model disagreement, not a calibrated confidence interval. What-if comparisons change a measurement while keeping other inputs fixed; they show model associations rather than intervention benefits.')
p('The Python application serves a local HTML/CSS/JavaScript interface and loads artifacts/model.joblib. Users enter eight features or load an example to obtain a class, calibrated probability, threshold, component predictions, explanations and input notes. Validation rejects invalid values, supports missing laboratory measurements, and flags inputs outside training ranges. Inputs are not persisted.')
h('8. Execution and verification')
p('From the project folder, run python app.py and open http://127.0.0.1:8000. Stop the server with Ctrl+C. To retrain using the named dataset, run python train.py --data data/pima_with_headers.csv. Retraining overwrites generated model artifacts and reports. On a new machine, install requirements.txt in a virtual environment; the bundled binary is best loaded with the package versions recorded in requirements-lock.txt, or retrained in that environment.')
p('The saved verification log records 22 passing regression tests covering dataset integrity, disjoint partitions, fold-local imputation, metrics, prediction, Shapley additivity, input validation and HTTP behavior. Existing delivery checks also record a healthy live server and a notebook run with no error outputs. Browser interaction testing was not recorded in those checks.')
h('9. Conclusion and future work')
p('The project completes a reproducible ANN study, comparative evaluation and an explainable local prediction application. The results demonstrate useful ranking on this holdout while preserving negative findings about calibration and ensemble dominance. Main limitations are the small historical single-population cohort, missing measurements, absence of external validation, selection-biased CV estimates and non-causal explanations. Future work should use a diverse independent cohort, nested CV, subgroup evaluation and validation-based calibration comparisons. Any clinical use would require separate evidence and an appropriate evaluation design.')
h('References and experiment evidence')
p('[1] Pima Indians Diabetes Database: preserved source metadata and CSV, James Brownlee dataset mirror. Original owner: NIDDK.',size=18,after=45)
p('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names',size=16,after=65,color='007F82')
p('[2] scikit-learn documentation: MLPClassifier.',size=18,after=45)
p('https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html',size=16,after=65,color='007F82')
p('[3] scikit-learn documentation: Probability calibration.',size=18,after=45)
p('https://scikit-learn.org/stable/modules/calibration.html',size=16,after=65,color='007F82')
p('Measured evidence: reports/results.json, test_metrics_*.csv, feature_ablation.csv, verification.txt; implementation: src/diabetes/. References accessed 8 October 2026.',size=17,color='5B6E7C',after=0)

sect='<w:sectPr><w:headerReference w:type="default" r:id="rIdHeader"/><w:footerReference w:type="default" r:id="rIdFooter"/><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="800" w:right="1020" w:bottom="850" w:left="1020" w:header="350" w:footer="350"/></w:sectPr>'
document=f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{NS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:body>'+''.join(parts)+sect+'</w:body></w:document>'
header=f'<w:hdr xmlns:w="{NS}">'+para('ANN_DIABITIES  |  PROJECT REPORT',size=16,color='5B6E7C',after=0)+'</w:hdr>'
footer=f'<w:ftr xmlns:w="{NS}"><w:p><w:pPr><w:jc w:val="right"/></w:pPr><w:r><w:rPr><w:sz w:val="16"/><w:color w:val="5B6E7C"/></w:rPr><w:t xml:space="preserve">Page </w:t></w:r><w:fldSimple w:instr="PAGE"/><w:r><w:t xml:space="preserve"> of 4</w:t></w:r></w:p></w:ftr>'
types='''<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/><Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>'''
rels='''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
docrels='''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdHeader" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/><Relationship Id="rIdFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/></Relationships>'''
out=ROOT/'docs/Diabetes_Project_Report_4_Pages.docx'
types=types.replace('</Types>','<Default Extension="png" ContentType="image/png"/></Types>')
docrels=docrels.replace('</Relationships>','<Relationship Id="rIdDiagram" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/methodology_schematic.png"/></Relationships>')
with ZipFile(out,'w',ZIP_DEFLATED) as z:
    for name,value in {'[Content_Types].xml':types,'_rels/.rels':rels,'word/document.xml':document,'word/_rels/document.xml.rels':docrels,'word/header1.xml':header,'word/footer1.xml':footer}.items(): z.writestr(name,value)
    z.write(ROOT/'docs/figures/methodology_schematic.png','word/media/methodology_schematic.png')
print(out)
