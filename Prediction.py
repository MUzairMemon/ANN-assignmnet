import pandas as pd
import numpy as np
import Functions as fs
from scaling import Scalling 

def prediction (dataSet):
    
    DataSet = Scalling(path)
    thetas = np.load('thetas.npz')

    thi_1 = thetas['thi_1']
    th0_1 = thetas['th0_1']
    thi_2 = thetas['thi_2']
    th0_2 = thetas['th0_2']
    thi_3 = thetas['thi_3']
    th0_3 = thetas['th0_3']
    thi_4 = thetas['thi_4']
    th0_4 = thetas['th0_4']
    
    
    
    X=fs.ImageProc(path)
    
    X,Y,n,m = fs.PreProc(DataSet)
    Z1,A1,Z2,hyp = fs.forward_Propagation (thi_1,th0_1,thi_2,th0_2,thi_3,th0_3,thi_4,th0_4,X)
    Sample = X.shape[1]
    for i in range(0,Sample,1):
        if hyp[0][i] >=0.5:
            hyp[0][i] = 1
        elif hyp[0][i] < 0.5:
            hyp[0][i] = 0
       
    Accuracy,tp,tn,fp,fn = fs.accuracy(DataSet, thi_1, th0_1, thi_2, th0_2)

    return (hyp.T,Accuracy)


path=r'D:\RIME\Sem 3\DL\Project\nnProj\Test'
Prediction , Accuracy   = prediction(path)
print(Prediction, Accuracy)

