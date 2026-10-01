import csv, random
from pathlib import Path
FIELDS=["age","income","education_years","employment_years","debt_ratio","group","label"]
def generate(path="data/socioeconomic.csv",n=800,seed=42):
    rng=random.Random(seed); rows=[]
    for _ in range(n):
        group=rng.choice([0,1]); age=rng.randint(21,65); edu=rng.randint(10,20)
        emp=max(0,min(age-18,int(rng.gauss(10+edu/3,6))))
        income=max(12000,int(rng.gauss(22000+edu*2400+emp*900,15000)))
        debt=min(.95,max(.02,rng.betavariate(2.2,4.0)))
        score=.000035*income+.10*edu+.05*emp-2.2*debt-.55*group+rng.gauss(0,.7)
        rows.append([age,income,edu,emp,round(debt,3),group,int(score>3.15)])
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(FIELDS); w.writerows(rows)
    return len(rows)
if __name__=='__main__': print(f"Generated {generate()} synthetic records.")
