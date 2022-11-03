import numpy as np
import cv2
import os.path

CAT = r'D:\RIME\Sem 3\DL\Project\nnProj\Cat'
DOG = r'D:\RIME\Sem 3\DL\Project\nnProj\Dog'

def ImageProc(path):
    x=[] 
    for file in os.listdir(path):
               if (os.path.isfile(path + "/" + file)):
                   ROW=[]
                   img = cv2.imread(path + "/" + file)
                   img = cv2.resize(img, (300, 400)
                   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                   image = np.asarray(img)
                   nrows = image.shape[0]
                   ROW = image[0]
                   for i in np.arange(1,nrows,1):
                       row = image[i]
                       ROW=np.concatenate((ROW,row))   
                   x.append(ROW)
    X = np.array(x)
    return (X)

def PreProc(X_cat,X_dog,Y_cat,Y_dog):
    
    X_train = np.concatenate((X_cat,X_dog))
    Y_train = np.concatenate((Y_cat,Y_dog))
    
    X_train = X_train.T
    Y_train = Y_train.T
    
    print(X_train.shape)
    print(Y_train.shape)
    
    m = X_train.shape[1]    #no. of Examples
    n = X_train.shape[0]

    return(X_train,Y_train,n,m)
    
def thetas(n):
    thi_1 = np.random.rand(100,n)
    th0_1 = np.random.rand(100,1)
    thi_2 = np.random.rand(100,100)
    th0_2 = np.random.rand(100,1)
    thi_3 = np.random.rand(100,100)
    th0_3 = np.random.rand(100,1)
    thi_4 = np.random.rand(1,100)
    th0_4 = np.random.rand(1,1)
    
    return (thi_1,th0_1,thi_2,th0_2,thi_3,th0_3,thi_4,th0_4)

def sigmoid (z):
    return 1/(1+(np.exp(-z)))

def relu (z):
    return np.maximum(z,0)

def drelu (z):
    f=np.array(z>0 , dtype= np.float32)
    return f

def forward_Propagation (thi_1,th0_1,thi_2,th0_2,thi_3,th0_3,thi_4,th0_4,X_train):
    Z1 = np.dot(thi_1 , X_train ) + th0_1
    A1 = relu(Z1)
    Z2 = np.dot(thi_2 , A1) + th0_2
    A2 = relu(Z2)
    Z3 = np.dot(thi_3 , A2) + th0_3
    A3 = relu(Z3)
    Z4 = np.dot(thi_4 , A3) + th0_4
    hyp = sigmoid(Z4)
    return (Z1,A1,Z2,A2,Z3,A3,Z4,hyp)

def Reg_parameter(thi_1,thi_2,thi_3,thi_4,L,m):
    Reg = (L/(2*m))*(np.sum(thi_1*thi_1)+np.sum((thi_2*thi_2))+np.sum((thi_3*thi_3))+np.sum((thi_4*thi_4)))
    return Reg

def cost_function (hyp,Y,m):
    Y_1 = np.log10(hyp+0.0000000001)
    Y_0 = np.log10(1-hyp+0.000000000001)
    cost_fun = ((-1/m)*(np.sum((Y * Y_1) + ((1-Y)*Y_0))))
    return (cost_fun)

def backward_propagation (m,hyp,Y_train,A1,A2,A3,thi_2,thi_3,thi_4,X_train):
    dz4 = hyp-Y_train
    dthi_4 = (1/ m) * dz4.dot(A3.T)
    dth0_4 = (1/ m) * np.sum(dz4)
    
    # 3rd Hidden LaY_trainer
    dz3 = (thi_4.T.dot(dz4)) * drelu(A3)
    print(shape.dz3)
    dthi_3 = (1/ m) * thi_3.dot(dz3)
    dth0_3 = (1/ m) * np.sum(dz3)
    
    # 2nd Hidden LaY_trainer
    dz2 = thi_3.T.dot(dz3) * drelu(A2)
    #print(A1.shape)
    dthi_2 = (1/ m) * dz2.dot(A1.T)
    dth0_2 = (1/ m) * np.sum(dz2)
    
    # 1st Hidden LaY_trainer
    dz1 = thi_2.T.dot(dz2) * drelu(A1)
    dthi_1 = (1/ m) * dz1.dot(X_train.T)
    dth0_1 = (1/ m) * np.sum(dz1)
    
    print ((Y_train-hyp).shape)
    return (dthi_4,dth0_4,dthi_3,dth0_3,dthi_2,dth0_2,dthi_1,dth0_1)

def thetas_Update(dthi_4,dth0_4,dthi_3,dth0_3,dthi_2,dth0_2,dthi_1,dth0_1,thi_1,th0_1,thi_2,th0_2,thi_3,th0_3,thi_4,th0_4,a,L,m):
    thi_1 = thi_1 - (a*dthi_1 + (L/m)*thi_1)
    th0_1 = th0_1 - a*dth0_1
    thi_2 = thi_2 - (a*dthi_2 + (L/m)*thi_2)
    th0_2 = th0_2 - a*dth0_2
    thi_3 = thi_3 - (a*dthi_3 + (L/m)*thi_3)
    th0_3 = th0_3 - a*dth0_3
    thi_4 = thi_4 - (a*dthi_4 + (L/m)*thi_4)
    th0_4 = th0_4 - a*dth0_4
    return (thi_1,th0_1,thi_2,th0_2,thi_3,th0_3,thi_4,th0_4)

def accuracy (dataSet,thi_1,th0_1,thi_2,th0_2,thi_3,th0_3,thi_4,th0_4):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    X,Y,n,m = initial_parameters (dataSet)

    Z1,A1,Z2,hyp = forward_Propagation (thi_1,th0_1,thi_2,th0_2,X)
    
    for h in range(0,m,1):
        if hyp[0][h] >= 0.5 and Y[0][h] == 1:
            tp=tp+1
        elif hyp[0][h] < 0.5 and Y[0][h] == 0:
            tn=tn+1
        elif hyp[0][h] >= 0.5 and Y[0][h] == 0:
            fp=fp+1
        elif hyp[0][h] < 0.5 and Y[0][h] == 1:
            fn=fn+1
    Accuracy = (tp+tn)*(100/m)
    return Accuracy,tp,tn,fp,fn