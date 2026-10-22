'use strict';
const $ = id => document.getElementById(id);
let lastInputs = null;
let busy = false;
const friendly = {Pregnancies:'Pregnancies',Glucose:'Glucose',BloodPressure:'Blood pressure',SkinThickness:'Skin thickness',Insulin:'Insulin',BMI:'BMI',DiabetesPedigreeFunction:'Pedigree score',Age:'Age'};
const percent = value => (value * 100).toFixed(1) + '%';
async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed.');
  return data;
}
function readInputs() {
  const inputs = {};
  for (const [key, value] of new FormData($('prediction-form'))) inputs[key] = value === '' ? null : Number(value);
  return inputs;
}
function textNode(tag, content, className) { const node = document.createElement(tag);node.textContent = content;if(className)node.className=className;return node; }
function showPrediction(data) {
  $('empty-state').hidden = true;
  $('prediction-output').hidden = false;
  $('what-if').hidden = false;
  $('probability').textContent = percent(data.probability);
  $('meter-fill').style.width = percent(data.probability);
  $('threshold-mark').style.left = percent(data.threshold);
  $('threshold').textContent = data.threshold.toFixed(2);
  $('disagreement').textContent = data.component_std.toFixed(3);
  $('decision').textContent = `Predicted dataset label: Outcome ${data.predicted_outcome} · ${data.predicted_outcome ? 'at or above' : 'below'} the validation-selected threshold.`;
  const e = data.explanation;
  $('explanation-baseline').textContent = `Background probability ${percent(e.baseline)} + feature contributions = ${percent(e.probability)}. Contribution units: percentage points.`;
  const contributions = Object.entries(e.contributions).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  const scale = Math.max(...contributions.map(([,v])=>Math.abs(v)), .001);
  $('contributions').replaceChildren(...contributions.map(([name,value])=>{
    const row = textNode('div','','contribution');
    const track = textNode('div','','contribution-track');
    const bar = textNode('div','','contribution-bar');
    bar.style.width = (Math.abs(value)/scale*100)+'%';
    bar.style.background = value>=0 ? 'var(--orange)' : 'var(--teal)';track.append(bar);
    row.append(textNode('span',friendly[name]),track,textNode('span',(value>=0?'+':'')+(value*100).toFixed(1),'contribution-value'));
    return row;
  }));
  $('components').replaceChildren(...Object.entries(data.components).map(([name,p])=>{const row=textNode('div','','component');row.append(textNode('span',name),textNode('strong',percent(p)));return row;}));
  $('prediction-notes').replaceChildren(...data.notes.map(note=>textNode('li',note)));
  $('what-if-result').textContent = 'Change a value to compare with the last submitted profile.';
  $('what-if-chart').replaceChildren();
}
$('prediction-form').addEventListener('submit', async event=>{
  event.preventDefault();if(busy)return;busy=true;
  const inputs=readInputs();const button=$('predict-button');
  button.disabled=true;button.textContent='Calculating…';$('form-error').textContent='';
  try {const data=await api('/api/predict',inputs);lastInputs=inputs;showPrediction(data);}
  catch(error){$('form-error').textContent=error.message;}
  finally{busy=false;button.disabled=false;button.textContent='Predict';}
});
$('load-example').addEventListener('click',()=>{
  const example={Pregnancies:2,Glucose:120,BloodPressure:70,SkinThickness:25,Insulin:85,BMI:30,DiabetesPedigreeFunction:.45,Age:32};
  for(const [name,value] of Object.entries(example)) $('prediction-form').elements[name].value=value;
  $('form-error').textContent='';
});
function svgNode(tag,attrs){const node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,val]of Object.entries(attrs))node.setAttribute(key,val);return node;}
function plotCurve(curve,feature){
  const svg=svgNode('svg',{viewBox:'0 0 610 230',role:'img','aria-label':`What-if predicted probability as ${feature} changes`});
  const lo=curve[0].value,hi=curve[curve.length-1].value;
  const x=v=>55+(v-lo)/(hi-lo||1)*520;const y=p=>185-p*150;
  for(const p of [0,.5,1]){svg.append(svgNode('line',{x1:55,y1:y(p),x2:575,y2:y(p),stroke:'#d8dfd5'}));const t=svgNode('text',{x:18,y:y(p)+4,fill:'#627574','font-size':10});t.textContent=percent(p);svg.append(t);}
  svg.append(svgNode('polyline',{points:curve.map(d=>`${x(d.value)},${y(d.probability)}`).join(' '),fill:'none',stroke:'#156b68','stroke-width':3}));
  for(const v of [lo,(lo+hi)/2,hi]){const t=svgNode('text',{x:x(v),y:208,'text-anchor':'middle',fill:'#627574','font-size':11});t.textContent=v.toFixed(1);svg.append(t);}
  const label=svgNode('text',{x:315,y:229,'text-anchor':'middle',fill:'#627574','font-size':11});label.textContent=friendly[feature];svg.append(label);$('what-if-chart').replaceChildren(svg);
}
$('what-if-form').addEventListener('submit',async event=>{
  event.preventDefault();if(!lastInputs||busy)return;busy=true;const button=event.currentTarget.querySelector('button');button.disabled=true;$('what-if-error').textContent='';
  try{const data=await api('/api/what-if',{inputs:lastInputs,feature:$('what-if-feature').value,value:Number($('what-if-value').value)});$('what-if-result').textContent=`Original ${percent(data.baseline_probability)} → scenario ${percent(data.scenario_probability)} (${data.difference>=0?'+':''}${(data.difference*100).toFixed(1)} percentage points). ${data.notes.join(' ')}`;plotCurve(data.curve,data.feature);}
  catch(error){$('what-if-error').textContent=error.message;}
  finally{busy=false;button.disabled=false;}
});
$('what-if-feature').addEventListener('change',()=>{const f=$('what-if-feature').value;$('what-if-value').value=lastInputs?.[f] || (f==='BMI'?30:100);});
async function loadEvidence(){
  try{const data=await api('/api/results');const m=data.operating_test_metrics.CalibratedEnsemble;
    const stats=[['Dataset records',data.dataset_rows],['Train / validation / test',`${data.split_sizes.train} / ${data.split_sizes.validation} / ${data.split_sizes.test}`],['Ensemble test ROC-AUC',m.roc_auc.toFixed(3)],['Test recall at selected threshold',percent(m.recall)]];
    $('study-stats').replaceChildren(...stats.map(([label,value])=>{const box=textNode('div','','stat');box.append(textNode('small',label),textNode('strong',value));return box;}));
    const keys=['accuracy','precision','recall','f1','specificity','roc_auc','pr_auc','brier'];
    $('results-table').replaceChildren(...Object.entries(data.default_test_metrics).map(([name,metrics])=>{const row=textNode('tr','',name==='CalibratedEnsemble'?'highlight':'');row.append(textNode('td',name),...keys.map(key=>textNode('td',metrics[key].toFixed(3))));return row;}));
    const ci=data.ensemble_operating_95_ci.roc_auc;
    $('study-result-note').textContent=`Calibrated ensemble ROC-AUC 95% stratified bootstrap interval: ${ci[0].toFixed(3)}–${ci[1].toFixed(3)} (${data.bootstrap_iterations} resamples; fixed fitted model). Representation selected on training CV: ${data.selected_representation}.`;
    $('limitations-list').replaceChildren(...data.study_limitations.map(note=>textNode('li',note)));
  }catch(error){$('study-stats').append(textNode('p','Results could not be loaded: '+error.message,'error'));}
}
loadEvidence();
