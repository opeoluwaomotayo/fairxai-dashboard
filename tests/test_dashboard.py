from generate_dataset import generate
from fairxai import train, explain, fairness_metrics
def test_metrics_exist():
    x=fairness_metrics([1,1,0,0],[1,0,1,0],[0,1,0,1]);assert 'demographic_parity_difference' in x and 'equalized_odds_gap' in x
def test_model_and_explanation(tmp_path):
    d=tmp_path/'d.csv';m=tmp_path/'m.joblib';r=tmp_path/'r.json';generate(str(d),240);out=train(str(d),str(m),str(r));assert len(out)==2
    e=explain({"age":40,"income":65000,"education_years":16,"employment_years":12,"debt_ratio":.25},str(m),3);assert len(e['contributions'])==3 and 0<=e['probability']<=1
def test_streamlit_import():
    import app
    assert app is not None
