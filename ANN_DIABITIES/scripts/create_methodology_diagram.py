"""Draw the implemented workflow as reusable PNG, SVG and PDF assets."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/figures'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Liberation Sans','svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(13.4,6.3))
fig.patch.set_facecolor('white')
ax.set(xlim=(0,13.4),ylim=(-.15,6.15)); ax.axis('off')
navy='#11263B'; teal='#007F82'; muted='#5B6E7C'

def node(x,y,title,body,w=3.6,h=.77,fill='#E5F3F1'):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.035,rounding_size=0.08',facecolor=fill,edgecolor=teal if fill!='#11263B' else navy,linewidth=1.1))
    c='white' if fill==navy else navy
    ax.text(x,y+.14,title,ha='center',va='center',fontsize=12.5,fontweight='bold',color=c)
    ax.text(x,y-.15,body,ha='center',va='center',fontsize=10.6,color=c)

def arrow(a,b,dashed=False,points=None):
    pts=[a]+(points or [])+[b]
    for i,(start,end) in enumerate(zip(pts,pts[1:])):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>' if i==len(pts)-2 else '-',mutation_scale=13,linewidth=1.3,color=muted,linestyle='--' if dashed else '-'))

node(2.2,5.62,'Pima dataset','768 records · 8 inputs + Outcome')
node(6.7,5.62,'Validate and stratify','Schema checks · Seed 42')
arrow((4.04,5.62),(4.84,5.62))
for x,label,body in [(2.2,'Training partition','460 records'),(6.7,'Validation partition','154 records'),(11.2,'Test partition','154 records · untouched holdout')]:
    node(x,4.45,label,body)
    arrow((6.7,5.2),(x,4.86),points=[(6.7,5.05),(x,5.05)] if x!=6.7 else None)
node(2.2,3.22,'Fold-local preprocessing','Zero handling · Impute · Scale')
node(2.2,1.99,'Feature selection and tuning','5 predictors · 5 learners · 5-fold CV')
node(2.2,.76,'Ensemble and calibration','Equal voting · Out-of-fold sigmoid')
for ya,yb in [(4.04,3.63),(2.81,2.4),(1.58,1.17)]: arrow((2.2,ya),(2.2,yb))
node(6.7,3.22,'Select F1 threshold','Validation predictions → threshold 0.33')
arrow((6.7,4.04),(6.7,3.63))
arrow((4.05,.76),(4.48,3.22),points=[(4.4,.76),(4.4,3.22)])
arrow((4.48,3.22),(4.85,3.22))
node(6.7,.76,'Freeze deployed bundle','Calibrated model + fixed threshold',fill=navy)
arrow((6.7,2.81),(6.7,1.17))
node(11.2,.76,'Held-out evaluation','Metrics · Confusion matrix · Bootstrap')
arrow((8.55,.76),(9.35,.76))
node(11.2,2.6,'Local application','Probability · Shapley · What-if',fill=navy)
# Test records travel along the outside of the application box.
arrow((11.2,4.04),(11.2,1.17),points=[(13.12,4.04),(13.12,1.35),(11.2,1.35)])
arrow((8.55,.76),(9.35,2.6),points=[(8.8,.76),(8.8,2.6)])
ax.text(6.7,-.02,'Training fits transformations and models; validation chooses the threshold; test data only evaluates the frozen model.',ha='center',va='center',fontsize=10.2,color=muted)
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.04)
for ext in ['png','svg','pdf']:
    fig.savefig(OUT/f'methodology_schematic.{ext}',dpi=240,facecolor='white')
print(OUT/'methodology_schematic.png')
