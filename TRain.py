import pandas as pd
import Functions as fs
import numpy as np
import matplotlib.pyplot as plt



CAT = r'D:\RIME\Sem 3\DL\Project\nnProj\Cat'
DOG = r'D:\RIME\Sem 3\DL\Project\nnProj\Dog'

X_cat = fs.ImageProc(CAT)
X_dog = fs.ImageProc(DOG)
Y_cat =  np.zeros((10,1))
Y_dog =  np.ones((10,1))

X,Y,n,m = fs.PreProc(X_cat,X_dog,Y_cat,Y_dog)


thi_1,th0_1,thi_2,th0_2,thi_3,th0_3,thi_4,th0_4 = fs.thetas(n)
cost = []

L = 0.05
a = 0.05
ite = 300
    
for i in range(1,ite,1):
    Z1,A1,Z2,A2,Z3,A3,Z4,hyp = fs.forward_Propagation (thi_1,th0_1,thi_2,th0_2,thi_3,th0_3,thi_4,th0_4,X)
    Reg = fs.Reg_parameter(thi_1,thi_2,thi_3,thi_4,L,m)
    cost_fun = fs.cost_function (hyp,Y,m) + Reg
    dthi_4,dth0_4,dthi_3,dth0_3,dthi_2,dth0_2,dthi_1,dth0_1= fs.backward_propagation (m,hyp,Y,A1,A2,A3,thi_2,thi_3,thi_4,X)
    print (dthi_2.shape,dth0_2.shape,dthi_1.shape,dth0_1.shape) 

    thi_1,th0_1,thi_2,th0_2,thi_3,th0_3,thi_4,th0_4 = fs.thetas_Update(dthi_4,dth0_4,dthi_3,dth0_3,dthi_2,dth0_2,dthi_1,dth0_1,thi_1,th0_1,thi_2,th0_2,thi_3,th0_3,thi_4,th0_4,a,L,m)
    cost.append(cost_fun)
   
print (cost)
plt.plot(cost)

np.savez('thetas.npz' , thi_1=thi_1,thi_2=thi_2,th0_1=th0_1,th0_2=th0_2,thi_3=thi_3,thi_4=thi_4,th0_3=th0_3,th0_4=th0_4   )