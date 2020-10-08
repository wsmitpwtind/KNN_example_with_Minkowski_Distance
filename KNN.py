import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

Train_data = np.loadtxt('trainData.txt', dtype=np.float32)
Test_data = np.loadtxt('testData.txt', dtype=np.float32)



def Train_data_Print():
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   def ColorPrint(x):
       if x==1:
           return 'r'
       if x==2:
           return 'g'
       if x==3:
           return 'b'
   for i in Train_data:
       ax.scatter(i[0],i[1],i[2],c=ColorPrint(i[3]), label=' ')
   ax.legend()
   plt.show()

def Test_data_Print():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    def ColorPrint(x):
        if x==1:
            return 'r'
        if x==2:
            return 'g'
        if x==3:
            return 'b'
    for i in Test_data:
        ax.scatter(i[0],i[1],i[2],c=ColorPrint(i[3]), label=' ')
    ax.legend()
    plt.show()

def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma
    return x

def Minkowski_Distance(x,y,p):
    return (abs(x[0]-y[0])**p+abs(x[1]-y[1])**p+abs(x[2]-y[2])**p)**(1/p)

Accuracy=0
K_value_list=[]

Train_data_Print()
Test_data_Print()

Train_mean=Train_data.mean(axis = 0)
Train_std=Train_data.std(axis = 0)

Train_data[:,0]=Z_ScoreNormalization(Train_data[:,0],Train_data.mean(axis = 0)[0],Train_data.std(axis = 0)[0])
Train_data[:,1]=Z_ScoreNormalization(Train_data[:,1],Train_data.mean(axis = 0)[1],Train_data.std(axis = 0)[1])
Train_data[:,2]=Z_ScoreNormalization(Train_data[:,2],Train_data.mean(axis = 0)[2],Train_data.std(axis = 0)[2])

for Each_test_data in Test_data:
    Each_test_data[0] = Z_ScoreNormalization(Each_test_data[0], Train_mean[0], Train_std[0])
    Each_test_data[1] = Z_ScoreNormalization(Each_test_data[1], Train_mean[1], Train_std[1])
    Each_test_data[2] = Z_ScoreNormalization(Each_test_data[2], Train_mean[2], Train_std[2])
'''
for K_value in range(1,100):
   for Each_test_data in Test_data:
       Value_list=[]
       for Each_train_data in Train_data:
           Value_list.append([Minkowski_Distance(Each_test_data,Each_train_data,2),Each_train_data[3]])
       Value_list=sorted(Value_list, key=(lambda x: x[0]))
       Value_list=np.array(Value_list)
       Value_flag=[0,0,0,0]
       for i in range(K_value):
           Value_flag[int(Value_list[i][1])]+=1
       if int(Each_test_data[3])==Value_flag.index(max(Value_flag)):
           Accuracy+=1
   K_value_list.append(Accuracy/Test_data.shape[0])
   Accuracy=0

print(K_value_list)
K_value_list=np.array(K_value_list)
plt.plot(K_value_list)
plt.show()
'''
K_value=43
Accuracy=0
K_value_list=[]
P_find_X = np.linspace(1, 3, 200)
P_find_Y = np.linspace(1, 3, 200)
for P_find in P_find_X:
   for Each_test_data in Test_data:
       Value_list=[]
       for Each_train_data in Train_data:
           Value_list.append([Minkowski_Distance(Each_test_data,Each_train_data,P_find),Each_train_data[3]])
       Value_list=sorted(Value_list, key=(lambda x: x[0]))
       Value_list=np.array(Value_list)
       Value_flag=[0,0,0,0]
       for i in range(K_value):
           Value_flag[int(Value_list[i][1])]+=1
       if int(Each_test_data[3])==Value_flag.index(max(Value_flag)):
           Accuracy+=1
   K_value_list.append(Accuracy/Test_data.shape[0])
   Accuracy=0
P_list=[]
for i in range(200):
    P_list.append(i/100+1)
print(P_list)
print(K_value_list)
K_value_list=np.array(K_value_list)
plt.plot(P_list,K_value_list)
plt.show()

