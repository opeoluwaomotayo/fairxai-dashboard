import json
from pathlib import Path
import joblib, numpy as np, pandas as pd, shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
BASE=["age","income","education_years","employment_years","debt_ratio"]
def _rate(mask,v):
    x=np.asarray(v)[np.asarray(mask)]; return float(np.mean(x)) if len(x) else 0.0
def fairness_metrics(y,p,g):
    y=np.asarray(y);p=np.asarray(p);g=np.asarray(g); a=g==0;b=g==1
    sr0=_rate(a,p);sr1=_rate(b,p);tpr0=_rate(a&(y==1),p);tpr1=_rate(b&(y==1),p);fpr0=_rate(a&(y==0),p);fpr1=_rate(b&(y==0),p)
    return {"selection_rate_group_0":sr0,"selection_rate_group_1":sr1,"demographic_parity_difference":sr1-sr0,"true_positive_rate_group_0":tpr0,"true_positive_rate_group_1":tpr1,"equal_opportunity_difference":tpr1-tpr0,"false_positive_rate_group_0":fpr0,"false_positive_rate_group_1":fpr1,"equalized_odds_gap":max(abs(tpr1-tpr0),abs(fpr1-fpr0))}
def train(dataset="data/socioeconomic.csv",model_path="models/dashboard.joblib",report_path="models/audit.json"):
    df=pd.read_csv(dataset); tr,te=train_test_split(df,test_size=.25,random_state=42,stratify=df.label)
    variants={"Baseline (group included)":BASE+["group"],"Audited (group excluded)":BASE}; report={}; bundle={}
    for name,features in variants.items():
        m=RandomForestClassifier(n_estimators=160,max_depth=7,random_state=42);m.fit(tr[features],tr.label);p=m.predict(te[features])
        report[name]={"performance":{"accuracy":float(accuracy_score(te.label,p)),"precision":float(precision_score(te.label,p,zero_division=0)),"recall":float(recall_score(te.label,p,zero_division=0)),"f1":float(f1_score(te.label,p,zero_division=0))},"fairness":fairness_metrics(te.label,p,te.group)}
        bundle[name]={"model":m,"features":features}
    Path(model_path).parent.mkdir(parents=True,exist_ok=True);joblib.dump(bundle,model_path);Path(report_path).write_text(json.dumps(report,indent=2),encoding='utf-8');return report
def explain(record,model_path="models/dashboard.joblib",top=5):
    b=joblib.load(model_path)["Audited (group excluded)"]; frame=pd.DataFrame([[record[x] for x in b['features']]],columns=b['features']);m=b['model'];e=shap.TreeExplainer(m)(frame);v=np.asarray(e.values)
    if v.ndim==3:v=v[:,:,1]
    items=sorted([{"feature":f,"shap_value":float(s),"value":float(frame.iloc[0][f])} for f,s in zip(b['features'],v[0])],key=lambda x:abs(x['shap_value']),reverse=True)[:top]
    return {"classification":int(m.predict(frame)[0]),"probability":float(m.predict_proba(frame)[0][1]),"contributions":items}
