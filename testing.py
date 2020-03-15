
from ICIAR2018_master.src.networks import pret
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
from loader import loader
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score as auc_roc
import pandas as pd
prev=np.inf

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.m = pret(pretrained=True)
        for param in self.m.parameters():
            param.requires_grad =True
        self.model = torch.nn.Sequential(self.m.features)
        self.out = nn.Linear(4096,1)        


    def forward(self,x):
        x=self.model(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        
        return x
    

def getData(folders):
    data=[]
    for i in range(len(folders)):
        patient=[]
        dataset = loader(path=folders[i], stride=512,patch_size=512, augment=False)
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        print('loading '+str(i))
        for index, (image, file_name) in enumerate(data_loader):
            image = image.squeeze()
            patient.extend(image.numpy())
        data.append(patient)
        
    return data

file=pd.read_csv('./data_files/imgs.csv')
wsi_pth=file.iloc[:,0]
file=pd.read_csv('./data_files/labels.csv')
lbl=l=np.array(file.iloc[:,0],dtype=np.float64)
data=np.array(getData(wsi_pth))

#%%
fold_auc=[]
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(data,lbl)
fold=0
for tr_idx, ts_idx in skf.split(data,lbl):
    fold+=1
    ixtr, ixts, ytr, yts= data[tr_idx],data[ts_idx],lbl[tr_idx],lbl[ts_idx]
    test_auc=[]
   
    model=Net().cuda()
    #e.g. input_model
    model.load_state_dict(torch.load('model_name'+str(fold)+'.pth'))
    
    predictions=[]
    for ijk in range(len(ixts)):
        sc=[]
        tsbag=ixts[ijk]
       
        for p in range(len(tsbag)):
            patch=torch.FloatTensor(tsbag[p])
            
            patch = patch.cuda()
            patch=Variable(patch)
            patch=patch.reshape([1,3,512,512])
            sc.append(mlp.forward(patch).item())
        predictions.append(float(np.max(sc)))
    test_auc.append(auc_roc(yts, predictions))
fold_auc.append(max(test_auc))
print(fold_auc)



   
    
    
    
    
   