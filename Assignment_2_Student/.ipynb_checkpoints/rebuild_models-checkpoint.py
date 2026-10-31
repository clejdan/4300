"""Recovery only. Record Part I identities before opening this file."""
import joblib
from forensics_tools import ROOT, load_split, refit_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
X,y,_=load_split('part1','train')
for k in ['A','B','C']:
    joblib.dump(refit_regression(k,X,y),ROOT/'models'/f'regression_{k}.joblib')
X,y,_=load_split('part2','train')
m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=10000,random_state=42)).fit(X,y)
joblib.dump(m,ROOT/'models/logistic_probe.joblib')
X,y,_=load_split('part3','train')
for k,depth,leaf in [('A',4,35),('B',8,8)]:
    m=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,random_state=42).fit(X,y)
    joblib.dump(m,ROOT/'models'/f'tree_{k}.joblib')
print('Original models rebuilt from training data. No settings were tuned.')
