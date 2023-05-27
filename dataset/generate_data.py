#%%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.utils.data as Data
import pandas as pd
import numpy as np
from PIL import Image
import torch.nn as nn

import os
import random
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable

var_list = ['X','Y','rho','u','v','P','T']
type = "aveg"
measure_set = []
true_set = []

#%%

for idx in range(10000):
    print(idx)
    dir = r'E:/RDE_GAN_HR_dataset/p=7e5_17e5/'
    if not os.path.exists(os.path.join(dir, 'fig')):
        os.makedirs(os.path.join(dir, 'fig'))
    if not os.path.exists(os.path.join(dir, 'dataset')):
        os.makedirs(os.path.join(dir, 'dataset'))
    f=open(os.path.join(dir, 'data/'+str(idx)+'.dat'), encoding='utf-8')
    sentimentlist = []
    i = 0
    for line in f:
        i += 1
        if i <= 15:
            continue
        s = line.strip().split(' ')
        sentimentlist.append(s)
    f.close()
    df_data=pd.DataFrame(sentimentlist,columns=var_list)
    
    xmesh_num = df_data['X'].value_counts()[0]
    ymesh_num = df_data['Y'].value_counts()[0]
    
    D = np.array(sentimentlist).astype(float).reshape(xmesh_num, ymesh_num, len(var_list))
    D = D.transpose(2,0,1)
    D = D[2:6,:,:]

    D = torch.tensor(np.array(D))
    torch.save(D, os.path.join(dir, "dataset/"+str(idx)+".pt"))
