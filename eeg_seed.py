# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:13:49 2020

@author: Maitha_
"""
#%%
# store the code and text in the same directory as the .mat file
import sys

import numpy as np
from scipy.io import loadmat
np.set_printoptions(threshold=sys.maxsize)


    
    
#     return data

# def modify(data)
#%%   
# f = open('DATA.txt', 'r')
# exp =f.readlines()
# f.close()


file = open('DATA.txt', 'r')
for line in file.readlines():
    exp = line.rstrip().split(',') 
features=["de_LDS","psd_LDS"]
l = np.array([ 1,  0, -1, -1,  0,  1, -1,  0,  1,  1,  0, -1,  0,  1, -1],dtype=int)

#%%
label=[]
test=[]
label_test=[]
data_=[]
for i in range(len(exp)):
    for j in range(15):
        
        load_=0
        load_ = loadmat(exp[i])[features[0]+str(j+1)]
        x = np.transpose(load_, ((0,2,1)))
        y = x.reshape(x.shape[2],-1).T
        data_.append(y)
        label.append(l[j]*np.ones((len(y[1]),1)))
    if i>=11:
            for j in range(15):
                load_=0
                load_ = loadmat(exp[i])[features[0]+str(j+1)]
                x = np.transpose(load_, ((0,2,1)))
                y = x.reshape(x.shape[2],-1).T   
                test.append(y)
                label_test.append(l[j]*np.ones((len(y[1]),1)))
#%%
for i in range(len(data_)):
    if i==0:
        train_X = np.array([])
        train_X = np.vstack([train_X, data_[i]]) if train_X.size else data_[i]
        train_y = np.array([])
        train_y = np.vstack([train_y,label[i]]) if train_y.size else label[i]
        
    else:
        train_X = np.column_stack((train_X, data_[i]))
        train_y = np.vstack([train_y, label[i]])

    
for i in range(len(test)):
    if i==0:
        test_X = np.array([])
        test_X = np.vstack([test_X, test[i]]) if test_X.size else test[i]
        test_y = np.array([])
        test_y = np.vstack([test_y,label_test[i]]) if test_y.size else label_test[i]
        
    else:
        test_X = np.column_stack((test_X, test[i]))
        test_y = np.vstack([test_y, label_test[i]])
    
#%%    
test_X =test_X.T
train_X = train_X.T
train_y=np.where(train_y==-1, 2, train_y) 
test_y=np.where(test_y==-1, 2, test_y)         
        
np.save("train_X",train_X)
np.save("train_y",train_y)
np.save("test_X",test_X)
np.save("test_y",test_y)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    