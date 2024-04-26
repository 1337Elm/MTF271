import numpy as np
import torch 
import sys 
import time
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
#from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from random import randrange
from joblib import dump, load

name = "NN/"

class ThePredictionMachine(nn.Module):

    def __init__(self):
        
        super(ThePredictionMachine, self).__init__()

# 2 hidden layers
        self.input   = nn.Linear(2, 50) # axis 0: dimension of X
        self.hidden1 = nn.Linear(50, 50)
        self.hidden2 = nn.Linear(50, 3) # axis 1: dimension of y

# test with 4 hidden layers
#       self.input   = nn.Linear(2, 50) # axis 0: dimension of X
#       self.hidden1 = nn.Linear(50, 50)
#       self.hidden2 = nn.Linear(50, 50)
#       self.hidden3 = nn.Linear(50, 25)
#       self.hidden4 = nn.Linear(25, 2) # axis 1: dimension of y



    def forward(self, x):
        x = nn.functional.relu(self.input(x))
        x = nn.functional.relu(self.hidden1(x))
        x = self.hidden2(x)
        return x


neural_net = torch.load(name + "model-Re-10000.pth")
scaler_yplus = load(name + "scaler-yplus-Re-10000.bin")
scaler_pk = load(name + "scaler-pk-Re-10000.bin")

neural_net.eval()

DNS_mean=np.genfromtxt(name + "Re2000.dat",comments="%")
y_DNS=DNS_mean[:,0];
yplus_DNS=DNS_mean[:,1];
u_DNS=DNS_mean[:,2];
uu_DNS=DNS_mean[:,3]**2;
vv_DNS=DNS_mean[:,4]**2;
ww_DNS=DNS_mean[:,5]**2;
uv_DNS=DNS_mean[:,10];
dudy_DNS= np.gradient(u_DNS,yplus_DNS)
k_DNS=0.5*(uu_DNS+vv_DNS+ww_DNS)


mean = np.genfromtxt(name + "Re2000_bal_k.dat", comments="%")
yplus=mean[:,1];
pk = mean[:,3];
eps = mean[:,2];

tau = k_DNS/eps


yplus = yplus.reshape(-1,1)
pk =  pk.reshape(-1,1)

X=np.zeros((len(dudy_DNS),2))
X[:,0] = scaler_pk.transform(pk)[:,0]
X[:,1] = scaler_yplus.transform(yplus)[:,0]
X_tensor = torch.tensor(X,dtype=torch.float32)


predictions = neural_net(X_tensor)
c_NN = predictions.detach().numpy()

b1=c_NN[:,0]
b2=c_NN[:,1]
b4=c_NN[:,2]

a_11 = tau**2*dudy_DNS**2/12*(b2-6*b4)
uu_NN = (a_11+0.6666)*k_DNS

a_22 = tau**2*dudy_DNS**2/12*(b2+6*b4)
vv_NN = (a_22+0.6666)*k_DNS

a_33 = -tau**2*dudy_DNS**2/6*b2
ww_NN = (a_33+0.6666)*k_DNS

a_12 = b1*tau*dudy_DNS/2
uv_NN = a_12*k_DNS

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(uu_NN,yplus, 'b--',label="$\overline{u'^2}$")
plt.plot(uu_DNS,yplus, 'b',label="$\overline{u'^2}_{DNS}$")
plt.plot(uv_NN,yplus, 'r--',label="$\overline{u'v'}$")
plt.plot(uv_DNS,yplus, 'r',label="$\overline{u'v'}_{DNS}$")
plt.plot(vv_NN,yplus, 'k--',label="$\overline{v'^2}$")
plt.plot(vv_DNS,yplus, 'k',label="$\overline{v'^2}_{DNS}$")
plt.xlabel("$Amplitude$")
plt.ylabel("$y^+$")
plt.title("Predictions for " + "$Re_\\tau = 2000$")
plt.axis([-1,8,0,2000])
plt.legend(loc="best",fontsize=12)
plt.savefig(name + 'preds2000.png')
