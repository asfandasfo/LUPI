import numpy as np
from sklearn.metrics import roc_auc_score as auc_roc
import pandas as pd
from sklearn.model_selection import StratifiedKFold


file=pd.read_csv('./data_files/labels.csv')
lbl=l=np.array(file.iloc[:,0],dtype=np.float64)
file=pd.read_csv('./data_files/genes.csv')
gene=np.array(file)

#%%

from sklearn import svm
fold_auc=[]
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(gene,lbl)
pr_auc=[]
fold=0
for tr_idx, ts_idx in skf.split(gene,lbl):
    fold+=1
    ytr, yts,pxtr,pxts=lbl[tr_idx],lbl[ts_idx],gene[tr_idx],gene[ts_idx]
   
    
    pmodel = svm.SVC(kernel='rbf',gamma='auto', C=100)
    pmodel.fit(pxtr,ytr)
    fold_auc.append(auc_roc(yts,pmodel.decision_function(pxts)))

print(np.mean(fold_auc))  
    
    
    