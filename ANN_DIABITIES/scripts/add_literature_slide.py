"""Add a sourced ten-paper literature table to the original project deck."""
from pathlib import Path
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT=Path(__file__).resolve().parents[1]
PAPERS=[
dict(author='Sisodia & Sisodia (2018)',title='Prediction of Diabetes using Classification Algorithms',short='Diabetes classification algorithms',method='DT, SVM and Naive Bayes',relevance='Compare multiple baseline learners',url='https://doi.org/10.1016/j.procs.2018.05.122'),
dict(author='Zou et al. (2018)',title='Predicting Diabetes Mellitus With Machine Learning Techniques',short='Diabetes with machine learning',method='RF, DT, neural nets; PCA / mRMR',relevance='Test feature selection and ANN',url='https://doi.org/10.3389/fgene.2018.00515'),
dict(author='Sneha & Gangil (2019)',title='Analysis of diabetes mellitus for early prediction using optimal features selection',short='Diabetes and optimal feature selection',method='Feature selection; classifier comparison',relevance='Measure feature-selection benefit',url='https://doi.org/10.1186/s40537-019-0175-6'),
dict(author='Tasin et al. (2023)',title='Diabetes prediction using machine learning and explainable AI techniques',short='Diabetes prediction with explainable AI',method='ML ensembles, SHAP / LIME, app',relevance='Explain the deployed classifier',url='https://doi.org/10.1049/htl2.12039'),
dict(author='Hasan et al. (2025)',title='Towards Transparent Diabetes Prediction: Combining AutoML and Explainable AI for Improved Clinical Insights',short='Transparent diabetes: AutoML + XAI',method='AutoGluon, XAI, Streamlit interface',relevance='Compare models and what-if outputs',url='https://doi.org/10.3390/info16010007'),
dict(author='Guo et al. (2017)',title='On Calibration of Modern Neural Networks',short='Calibration of modern neural networks',method='Confidence calibration; temperature scaling',relevance='Measure Brier and ECE separately',url='https://proceedings.mlr.press/v70/guo17a.html'),
dict(author='Lundberg & Lee (2017)',title='A Unified Approach to Interpreting Model Predictions',short='Unified interpretation of predictions',method='SHAP additive feature attribution',relevance='Explain full pipeline; check additivity',url='https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html'),
dict(author='Breiman (2001)',title='Random Forests',short='Random Forests',method='Randomized ensembles of decision trees',relevance='Tree-ensemble comparison baseline',url='https://doi.org/10.1023/A:1010933404324'),
dict(author='Friedman (2001)',title='Greedy function approximation: A gradient boosting machine',short='A gradient boosting machine',method='Stagewise additive / gradient boosting',relevance='Boosting comparison baseline',url='https://doi.org/10.1214/aos/1013203451'),
dict(author='Cortes & Vapnik (1995)',title='Support-vector networks',short='Support-vector networks',method='Margin-based kernel classification',relevance='Nonlinear SVM comparison baseline',url='https://doi.org/10.1007/BF00994018')
]
prs=Presentation(ROOT/'docs/Diabetes_Prediction_7_Slides.pptx')
s=prs.slides.add_slide(prs.slide_layouts[6])
s.background.fill.solid(); s.background.fill.fore_color.rgb=RGBColor.from_string('F5F8FA')

def rect(x,y,w,h,color):
 sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
 sh.fill.solid(); sh.fill.fore_color.rgb=RGBColor.from_string(color); sh.line.fill.background()

def text(x,y,w,h,value,size,color='173247',bold=False):
 sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=sh.text_frame
 tf.word_wrap=True; tf.margin_left=tf.margin_right=Inches(.02); tf.margin_top=tf.margin_bottom=Inches(.02)
 tf.text=value
 for p in tf.paragraphs:
  p.font.name='Liberation Sans'; p.font.size=Pt(size); p.font.bold=bold; p.font.color.rgb=RGBColor.from_string(color); p.space_after=Pt(0)
 return sh

rect(0,0,.18,7.5,'007F82')
text(.6,.35,12,.35,'RESEARCH CONTEXT • 10 VERIFIED PAPERS',12,'007F82',True)
text(.6,.9,12,.7,'Literature Survey and Research Gap',30,'11263B',True)
text(.6,1.56,12,.3,'Five diabetes studies and five papers on the methods used in this project',12,'5B6E7C')
table=s.shapes.add_table(11,3,Inches(.6),Inches(1.98),Inches(12.1),Inches(4.3)).table
for col,w in zip(table.columns,[4.75,3.75,3.6]): col.width=Inches(w)
headers=['Paper / author and year','Approach','Connection to this project']
for i in range(11):
 table.rows[i].height=Inches(.39)
 values=headers if i==0 else [PAPERS[i-1]['author']+' — '+PAPERS[i-1]['short'],PAPERS[i-1]['method'],PAPERS[i-1]['relevance']]
 for j,value in enumerate(values):
  cell=table.cell(i,j); cell.fill.solid(); cell.fill.fore_color.rgb=RGBColor.from_string('11263B' if i==0 else ('EAF3F4' if i%2 else 'FFFFFF'))
  cell.margin_left=cell.margin_right=Inches(.08); cell.margin_top=cell.margin_bottom=Inches(.025)
  cell.text=value
  for p in cell.text_frame.paragraphs:
   p.font.name='Liberation Sans'; p.font.size=Pt(12 if i==0 else 11.3); p.font.bold=i==0; p.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else '173247'); p.space_after=Pt(0)
  if i and j==0:
   cell.text_frame.paragraphs[0].runs[0].hyperlink.address=PAPERS[i-1]['url']
rect(.6,6.39,12.1,.46,'DFF3EF')
text(.75,6.46,11.8,.35,'Gap tested: do selection, voting and calibration improve a fixed holdout? Evaluate trade-offs and explain the full pipeline.',12.2,'007F82',True)
rect(.6,6.94,12.1,.015,'D9E1E7')
text(.6,7.05,11.2,.25,'Sources: linked paper titles; full references in speaker notes and docs/literature_survey_10_papers.md',9.5,'5B6E7C')
text(12.1,7.03,.7,.3,'03 / 08',10,'5B6E7C')
notes=['The table summarizes contributions, not unsupported claims that every cited study has the same limitation. Research gap is the controlled experimental question tested by this project, not a claim of a new algorithm. Published accuracies use different data and protocols and are not directly comparable. Guo et al. study temperature scaling; this project uses sigmoid calibration. Hasan et al. appeared online 26 December 2024 and is cited by its 2025 journal issue. Tasin et al. appeared online in 2022 and is cited by its 2023 issue. The last five papers establish general methods rather than diabetes-specific experiments.\n\nReferences:']
for i,p in enumerate(PAPERS,1): notes.append(f"[{i}] {p['author']}. {p['title']}. {p['url']}")
s.notes_slide.notes_text_frame.text='\n'.join(notes)
# Insert the new slide after objectives, then update every slide counter.
ids=prs.slides._sldIdLst
new_id=ids[-1]; ids.remove(new_id); ids.insert(2,new_id)
for i,sl in enumerate(prs.slides,1):
 for sh in sl.shapes:
  if sh.has_text_frame and (sh.text.endswith('/ 07') or sh.text.endswith('/ 08')):
   sh.text=f'{i:02d} / 08'
   for p in sh.text_frame.paragraphs:
    p.font.name='Liberation Sans'; p.font.size=Pt(10); p.font.color.rgb=RGBColor.from_string('5B6E7C')
prs.core_properties.title='Diabetes Prediction Using ANN — Literature Survey Included'
out=ROOT/'docs/Diabetes_Prediction_8_Slides.pptx'
prs.save(out)
assert len(prs.slides)==8
(ROOT/'docs/literature_survey_10_papers.json').write_text(json.dumps(PAPERS,indent=2))
md=['# Literature survey and research gap','', 'Ten papers verified against publisher, proceedings or author sources on 8 October 2026. The first five are diabetes studies; the remaining five support model, calibration and explanation methods.','', '| No. | Paper | Approach | Connection to this project |','|---|---|---|---|---|']
for i,p in enumerate(PAPERS,1): md.append(f"| {i} | {p['author']}. [{p['title']}]({p['url']}) | {p['method']} | {p['relevance']} |")
md+=['','## Research gap addressed','', 'This project tests whether feature selection, equal-weight ensembling and sigmoid calibration improve discrimination and probability quality under a fixed train/validation/test protocol. It also explains the entire deployed pipeline. These are established methods; the contribution is reproducible integration, controlled comparisons and honest reporting of trade-offs.','', 'The experiment retains negative findings: the engineered representation did not win, gradient boosting had higher test AUC than the ensemble, and calibration worsened Brier score despite reducing binned ECE. These observations concern this project and are not universal criticisms of the papers. External validation remains an unresolved limitation.','', 'Paper titles on the slide link to sources. Full titles and links also appear in the slide speaker notes. Study metrics should not be ranked against this project without comparing populations, partitions and preprocessing.','', 'Publication-year notes: Hasan et al. is cited as Information 2025, 16(1), 7 (published online 26 December 2024); Tasin et al. is Healthcare Technology Letters 2023, 10, 1–10 (online December 2022). Guo et al. examine temperature scaling, while this project uses sigmoid calibration.']
(ROOT/'docs/literature_survey_10_papers.md').write_text('\n'.join(md)+'\n')
print(out)
