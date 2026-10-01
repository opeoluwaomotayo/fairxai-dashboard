from pathlib import Path
import pandas as pd
import streamlit as st
from generate_dataset import generate
from fairxai import train, explain
st.set_page_config(page_title="FairXAI Dashboard",page_icon="⚖️",layout="wide")
st.title("⚖️ FairXAI Responsible AI Dashboard")
st.caption("Performance • Fairness Audit • Explainable AI | Synthetic teaching data only")
if not Path("data/socioeconomic.csv").exists(): generate()
if not Path("models/dashboard.joblib").exists(): train()
report=train()
st.warning("Research/education prototype. Do not use this application to make consequential decisions about real people.")

tab1,tab2,tab3,tab4=st.tabs(["Overview","Fairness Audit","XAI Explanation","Methodology"])
with tab1:
    st.subheader("Model performance")
    rows=[]
    for name,r in report.items(): rows.append({"Model":name,**{k.title():round(v,3) for k,v in r['performance'].items()}})
    perf=pd.DataFrame(rows).set_index("Model");st.dataframe(perf,use_container_width=True);st.bar_chart(perf[["Accuracy","F1"]])
with tab2:
    st.subheader("Group fairness comparison")
    rows=[]
    for name,r in report.items():
        f=r['fairness'];rows.append({"Model":name,"Demographic parity difference":f['demographic_parity_difference'],"Equal opportunity difference":f['equal_opportunity_difference'],"Equalized odds gap":f['equalized_odds_gap']})
    fair=pd.DataFrame(rows).set_index("Model");st.dataframe(fair.round(3),use_container_width=True);st.bar_chart(fair)
    st.info("For difference metrics, values nearer zero indicate smaller measured group gaps. A metric near zero does not by itself establish ethical fairness.")
with tab3:
    st.subheader("Explain one audited-model prediction")
    c1,c2,c3=st.columns(3)
    age=c1.slider("Age",21,65,40);income=c2.number_input("Income",12000,200000,65000,1000);edu=c3.slider("Education years",10,20,16)
    emp=c1.slider("Employment years",0,45,12);debt=c2.slider("Debt ratio",0.02,0.95,0.25,0.01)
    r=explain({"age":age,"income":income,"education_years":edu,"employment_years":emp,"debt_ratio":debt})
    st.metric("Positive-class probability",f"{r['probability']:.1%}");st.metric("Classification",str(r['classification']))
    x=pd.DataFrame(r['contributions']);st.dataframe(x.round(4),use_container_width=True);st.bar_chart(x.set_index('feature')['shap_value'])
    st.caption("SHAP values explain this fitted model's behavior for the selected record. They do not establish causality.")
with tab4:
    st.markdown("""### Methodology\n1. Generate a reproducible synthetic socio-economic dataset with a deliberately introduced historical group effect.\n2. Train a baseline model that includes the group field.\n3. Train an audited model that excludes the group field.\n4. Compare predictive and group-fairness metrics.\n5. Explain individual audited-model predictions with SHAP.\n\n### Responsible interpretation\nRemoving a sensitive attribute is not a guarantee of fairness because other variables can act as proxies. Appropriate fairness definitions depend on the domain, law, harms, stakeholders, and decision context.""")
