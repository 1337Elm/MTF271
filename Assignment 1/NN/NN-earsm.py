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
import random

name = "NN/"
#########  The neural network modules: start ################################
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

# test with 4 hidden layers
#       x = nn.functional.relu(self.input(x))
#       x = nn.functional.relu(self.hidden1(x))
#       x = nn.functional.relu(self.hidden2(x))
#       x = nn.functional.relu(self.hidden3(x))
#       x = self.hidden4(x)

        return x

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print('in train_loop: len(dataloader)',len(dataloader))
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
# https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
#       optimizer.zero_grad(None)
        loss.backward()
        optimizer.step()

def test_loop(dataloader, model, loss_fn):
    global pred_numpy,pred1,size1
    size = len(dataloader.dataset)
    size1 = size
    num_batches = len(dataloader)
    test_loss = 0
    print('in test_loop: len(dataloader)',len(dataloader))

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
#transform from tensor to numpy
            pred_numpy = pred.detach().numpy()

    test_loss /= num_batches

    print(f"Avg loss: {test_loss:>.2e} \n")

    return test_loss

#########  The neural network modules: end ################################


plt.rcParams.update({'font.size': 22})
plt.interactive(True)
plt.close('all')

viscos = 1/10000

init_time = time.time()

# load DNS data, Re_tau =10 000
# %     y/h             y+             U+           u'+           v'+          w'+           uv'+         dU/dy+
DNS_mean=np.genfromtxt(name + "P10k.txt",comments="%")
y_DNS=DNS_mean[:,0];
yplus_DNS=DNS_mean[:,1];
u_DNS=DNS_mean[:,2];
uu_DNS=DNS_mean[:,3]**2;
vv_DNS=DNS_mean[:,4]**2;
ww_DNS=DNS_mean[:,5]**2;
uv_DNS=DNS_mean[:,6];
dudy_DNS= np.gradient(u_DNS,yplus_DNS)
k_DNS=0.5*(uu_DNS+vv_DNS+ww_DNS)
# %      y/h            y+         dissip        prod         p-strain       p-diff        T-diff        V-diff
DNS_uu = np.genfromtxt(name + "P10k.uu.txt",comments="%")
eps_DNS_uu = abs(DNS_uu[:,2])
visc_diff_uu =  DNS_uu[:,7]

DNS_vv = np.genfromtxt(name + "P10k.vv.txt",comments="%")
eps_DNS_vv = abs(DNS_vv[:,2])
visc_diff_vv =  DNS_vv[:,7]

DNS_ww = np.genfromtxt(name + "P10k.ww.txt",comments="%")
eps_DNS_ww = abs(DNS_ww[:,2])
visc_diff_ww =  DNS_ww[:,7]


eps_DNS= (eps_DNS_uu +eps_DNS_vv +eps_DNS_ww)/2
visc_diff = (visc_diff_uu +visc_diff_vv +visc_diff_ww)/2

dudy_DNS_non_plus  = np.gradient(u_DNS,y_DNS)
vist_DNS = abs(uv_DNS)/abs(dudy_DNS_non_plus)/viscos

#-----------------Data_manipulation--------------------

# don't use data near the wall or near the center
index_choose=np.nonzero((yplus_DNS > 5 )  & (yplus_DNS< 9200 ))

pk_DNS = -uv_DNS*dudy_DNS
dkdy=np.gradient(k_DNS,yplus_DNS)
d2kdy2=np.gradient(dkdy,yplus_DNS)
eps_DNS_org = eps_DNS
eps_DNS = np.maximum(eps_DNS-visc_diff,1e-10)

tau_DNS = k_DNS/eps_DNS

uu_DNS    =  uu_DNS[index_choose]
vv_DNS    =  vv_DNS[index_choose]
ww_DNS    =  ww_DNS[index_choose]
uv_DNS    =  uv_DNS[index_choose]
k_DNS     =  k_DNS[index_choose]
pk_DNS    =  pk_DNS[index_choose]
eps_DNS   =  eps_DNS[index_choose]
tau_DNS   =  tau_DNS[index_choose]
dudy_DNS  =  dudy_DNS[index_choose]
vist_DNS  =  vist_DNS[index_choose]
yplus_DNS =  yplus_DNS[index_choose]
y_DNS     =  y_DNS[index_choose]
u_DNS     =  u_DNS[index_choose]

# anisotropic stresses
a11_DNS=uu_DNS/k_DNS-0.66666
a22_DNS=vv_DNS/k_DNS-0.66666
a33_DNS=ww_DNS/k_DNS-0.66666
a12_DNS=uv_DNS/k_DNS

# Array for storing b1, b2, b4
b1_DNS=2*a12_DNS/tau_DNS/dudy_DNS  # b1
b2_DNS=6*(a11_DNS+a22_DNS)/tau_DNS**2/dudy_DNS**2  # b2
b4_DNS=(a22_DNS-a11_DNS)/tau_DNS**2/dudy_DNS**2  # b4

c = np.array([b1_DNS,b2_DNS,b4_DNS])

# transpose the target vector to make it a column vector  
y = c.transpose()

T = tau_DNS
pk_DNS_scaled = pk_DNS
# re-shape
pk_DNS_scaled = pk_DNS_scaled.reshape(-1,1)
yplus_DNS= yplus_DNS.reshape(-1,1)
# use MinMax scaler
scaler_pk = MinMaxScaler()
scaler_yplus = MinMaxScaler()
X=np.zeros((len(dudy_DNS),2))
X[:,0] = scaler_pk.fit_transform(pk_DNS_scaled)[:,0]
X[:,1] = scaler_yplus.fit_transform(yplus_DNS)[:,0]

# split the feature matrix and target vector into training and validation sets
# test_size=0.2 means we reserve 20% of the data for validation
# random_state=42 is a fixed seed for the random number generator, ensuring reproducibility

indices = np.arange(len(X))
X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(X, y, indices,test_size=0.2,shuffle=True,random_state=42)

# training data
dudy_DNS_train = dudy_DNS[index_train]
vist_DNS_train = vist_DNS[index_train]
k_DNS_train = k_DNS[index_train]
uu_DNS_train = uu_DNS[index_train]
vv_DNS_train = vv_DNS[index_train]
ww_DNS_train = ww_DNS[index_train]
uv_DNS_train = uv_DNS[index_train]
pk_DNS_train = pk_DNS[index_train]
tau_DNS_train = tau_DNS[index_train]
yplus_DNS_train = yplus_DNS[index_train]
b1_DNS_train = b1_DNS[index_train]
b2_DNS_train = b2_DNS[index_train]
b4_DNS_train = b4_DNS[index_train]

# testing (predicting)  data
dudy_DNS_train = dudy_DNS[index_train]
dudy_DNS_test = dudy_DNS[index_test]
vist_DNS_test = vist_DNS[index_test]
k_DNS_test = k_DNS[index_test]
uu_DNS_test = uu_DNS[index_test]
vv_DNS_test = vv_DNS[index_test]
ww_DNS_test = ww_DNS[index_test]
uv_DNS_test = uv_DNS[index_test]
pk_DNS_test = pk_DNS[index_test]
tau_DNS_test = tau_DNS[index_test]
yplus_DNS_test = yplus_DNS[index_test]
b1_DNS_test = b1_DNS[index_test]
b2_DNS_test = b2_DNS[index_test]
b4_DNS_test = b4_DNS[index_test]

# Set up hyperparameters
learning_rate = 0.07 # 

my_batch_size = 1
# number of epochs
epochs = 5000
epochs = 1200
epochs = 3

# convert the numpy arrays to PyTorch tensors with float32 data type
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# create PyTorch datasets and dataloaders for the training and validation sets
# a TensorDataset wraps the feature and target tensors into a single dataset
# a DataLoader loads the data in batches and shuffles the batches if shuffle=True
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=my_batch_size)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=my_batch_size)

start_time = time.time()

# Instantiate a neural network
neural_net = ThePredictionMachine()


# Initialize the loss function
loss_fn = nn.MSELoss()

# Choose loss function, check out https://pytorch.org/docs/stable/optim.html for more info
# In this case we choose Stocastic Gradient Descent
optimizer = torch.optim.SGD(neural_net.parameters(), lr=learning_rate)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=epochs)

loss_v = np.zeros(epochs)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, neural_net, loss_fn, optimizer)
    test_loss = test_loop(test_loader, neural_net, loss_fn)
    loss_v[t] = test_loss

print("Done!")

preds = neural_net(X_test_tensor)

print(f"{'time ML: '}{time.time()-start_time:.2e}")

#transform from tensor to numpy
c_NN = preds.detach().numpy()
 
c_NN_old = c_NN

b1=c_NN[:,0]
b2=c_NN[:,1]
b4=c_NN[:,2]

a_11 = tau_DNS_test**2*dudy_DNS_test**2/12*(b2-6*b4)
uu_NN = (a_11+0.6666)*k_DNS_test

a_22 = tau_DNS_test**2*dudy_DNS_test**2/12*(b2+6*b4)
vv_NN = (a_22+0.6666)*k_DNS_test

a_33 = -tau_DNS_test**2*dudy_DNS_test**2/6*b2
ww_NN = (a_33+0.6666)*k_DNS_test

a_12 = b1*tau_DNS_test*dudy_DNS_test/2
uv_NN = a_12*k_DNS_test

b1_std=np.std(b1-b1_DNS_test)/(np.mean(b1.flatten()**2))**0.5
b2_std=np.std(b2-b2_DNS_test)/(np.mean(b2.flatten()**2))**0.5
b4_std=np.std(b4-b4_DNS_test)/(np.mean(b4.flatten()**2))**0.5

print('\nb1_error_std',b1_std)
print('\nb2_error_std',b2_std)
print('\nb4_error_std',b4_std)

error_all_b1=abs(b1-b1_DNS_test)
error_index_b1= error_all_b1.argsort()
error_sorted_b1 = error_all_b1[error_index_b1]
# largest error:
largest_error_percent_b1 = error_all_b1[error_index_b1[-1]]/b1[error_index_b1[-1]]
print('largest_error_percent in uplus',largest_error_percent_b1)

error_all_b2=abs(b2-b2_DNS_test)
error_index_b2= error_all_b2.argsort()
error_sorted_b2 = error_all_b2[error_index_b2]
# largest error:
largest_error_percent_b2 = error_all_b2[error_index_b2[-1]]/b2[error_index_b2[-1]]
print('largest_error_percent in uplus',largest_error_percent_b2)

error_all_b4=abs(b4-b4_DNS_test)
error_index_b4= error_all_b4.argsort()
error_sorted_b4 = error_all_b4[error_index_b4]
# largest error:
largest_error_percent_b4 = error_all_b4[error_index_b4[-1]]/b4[error_index_b4[-1]]
print('largest_error_percent in uplus',largest_error_percent_b4)

np.savetxt(name + 'largest-Re-10000-error-percent.txt', [largest_error_percent_b1,largest_error_percent_b2,largest_error_percent_b4])

np.savetxt(name + 'error-loss.txt', [test_loss,b1_std,b2_std,b4_std] )

# save model to disk
filename = name + 'model-Re-10000.pth'
torch.save(neural_net, filename)
dump(scaler_yplus,name + 'scaler-yplus-Re-10000.bin')
dump(scaler_pk,name + 'scaler-pk-Re-10000.bin')

b1_min = np.min(b1)
b1_max = np.max(b1)
b2_min = np.min(b2)
b2_max = np.max(b2)
b4_min = np.min(b4)
b4_max = np.max(b4)
pk_max = np.max(pk_DNS)
pk_min = np.min(pk_DNS)
yplus_max = np.max(yplus_DNS)
yplus_min = np.min(yplus_DNS)

np.savetxt(name + 'min-max-re-10000.txt', [b1_min, b1_max, b2_min, b2_max, b4_min, b4_max, pk_min, pk_max, yplus_min, yplus_max] )

########################## b1
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(b1_DNS,yplus_DNS, 'r-',label='target')
plt.plot(b1,yplus_DNS_test, 'b.',label='NN')
plt.xlabel("$b_1$")
plt.ylabel("$y^+$")
plt.axis([-0.25,0,0,10000])
plt.legend(loc="best",fontsize=12)
plt.savefig(name + 'b1-Re-10000.png')

########################## b2
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(b2_DNS,yplus_DNS, 'r-',label='target')
plt.plot(b2,yplus_DNS_test, 'b.',label='NN')
plt.axis([0,0.1,0,10000])
plt.xlabel("$b_2$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig(name + 'b2-Re-10000.png')


########################## b4
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(b4_DNS,yplus_DNS, 'r-',label='target')
plt.plot(b4,yplus_DNS_test, 'b.',label='NN')
plt.axis([-0.25,0,0,10000])
plt.xlabel("$b_4$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig(name + 'b4-Re-10000.png')




########################## uu
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.scatter(uu_NN,yplus_DNS_test, marker="o", s=10, c="red", label="Neural Network")
ax1.plot(uu_DNS,yplus_DNS,'b-', label="Target")
plt.xlabel("$\overline{u'u'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig(name + 'uu-Re-10000.png')


########################## vv
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.scatter(vv_NN,yplus_DNS_test, marker="o", s=10, c="red", label="Neural Network")
ax1.plot(vv_DNS,yplus_DNS,'b-', label="Target")
plt.xlabel("$\overline{v'v'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig(name + 'vv-Re-10000.png')

########################## ww
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.scatter(ww_NN,yplus_DNS_test, marker="o", s=10, c="red", label="Neural Network")
ax1.plot(ww_DNS,yplus_DNS,'b-', label="Target")
plt.xlabel("$\overline{w'w'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig(name + 'ww-Re-10000.png')


########################## uv
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.scatter(uv_NN,yplus_DNS_test, marker="o", s=10, c="red", label="Neural Network")
ax1.plot(uv_DNS,yplus_DNS,'b-', label="Target")
plt.xlabel("$\overline{u'v'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig(name + 'uv-Re-10000.png')

########################## loss, error
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(loss_v, 'b-')
epoch_start = min(1000,epochs)-1
# plot 'loss' from epoch_start
# plt.axis([epoch_start, len(loss_v),min(loss_v),loss_v[epoch_start]])
plt.xlabel("$y^+$")
plt.ylabel("loss")
plt.savefig(name + 'loss-Re-10000.png')




# load k-omega data
komega=np.loadtxt(name + 'y_u_k_eps_uv_channel-10000-k-omega.txt')
y_komega=komega[:,0];
yplus_komega= y_komega/viscos
u_komega=komega[:,1];
k_komega=komega[:,2];
eps_komega=komega[:,3]*viscos;
uv_komega=komega[:,4];


index_choose=np.nonzero((yplus_komega > 5 )  & (yplus_komega< 9200 ))

y_komega=y_komega[index_choose]
yplus_komega= yplus_komega[index_choose]
u_komega=u_komega[index_choose]
k_komega=k_komega[index_choose]
eps_komega=eps_komega[index_choose]
uv_komega=uv_komega[index_choose]

# anisotropic stresses
a11_DNS=uu_DNS/k_DNS-0.66666
a22_DNS=vv_DNS/k_DNS-0.66666
a33_DNS=ww_DNS/k_DNS-0.66666
a12_komega=uv_komega/k_komega

# Array for storing b1, b2, b4
b1_komega=2*a12_komega/(k_komega/eps_komega)/np.gradient(u_komega,yplus_komega)  # b1
b2_DNS=6*(a11_DNS+a22_DNS)/tau_DNS**2/dudy_DNS**2  # b2
b4_DNS=(a22_DNS-a11_DNS)/tau_DNS**2/dudy_DNS**2  # b4

shorten = random.sample(range(0,len(b2_DNS)), len(b1_komega))
b2_DNS = b2_DNS[shorten]
b4_DNS = b4_DNS[shorten]

c = np.array([b1_komega,b2_DNS,b4_DNS])

# transpose the target vector to make it a column vector  
y = c.transpose()

T = tau_DNS
pk_komega = -uv_komega*np.gradient(u_komega,yplus_komega)
# re-shape
pk_komega_scaled = pk_komega.reshape(-1,1)
yplus_komega_scaled = yplus_komega.reshape(-1,1)
# use MinMax scaler
scaler_pk = MinMaxScaler()
scaler_yplus = MinMaxScaler()
X=np.zeros((len(np.gradient(np.gradient(u_komega,yplus_komega))),2))
X[:,0] = scaler_pk.fit_transform(pk_komega_scaled)[:,0]
X[:,1] = scaler_yplus.fit_transform(yplus_komega_scaled)[:,0]

indices = np.arange(len(X))
X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(X, y, indices,test_size=0.2,shuffle=True,random_state=42)

y_komega_train=y_komega[index_train]
yplus_komega_train= yplus_komega[index_train]
u_komega_train=u_komega[index_train]
k_komega_train=k_komega[index_train]
eps_komega_train=eps_komega[index_train]
uv_komega_train=uv_komega[index_train]

y_komega_test=y_komega[index_test]
yplus_komega_test= yplus_komega[index_test]
u_komega_test=u_komega[index_test]
k_komega_test=k_komega[index_test]
eps_komega_test=eps_komega[index_test]
uv_komega_test=uv_komega[index_test]


dudy_DNS_train = dudy_DNS[index_train]
dudy_DNS_test = dudy_DNS[index_test]
vist_DNS_test = vist_DNS[index_test]
k_DNS_test = k_DNS[index_test]
uu_DNS_test = uu_DNS[index_test]
vv_DNS_test = vv_DNS[index_test]
ww_DNS_test = ww_DNS[index_test]
uv_DNS_test = uv_DNS[index_test]
pk_DNS_test = pk_DNS[index_test]
tau_DNS_test = tau_DNS[index_test]
yplus_DNS_test = yplus_DNS[index_test]
b1_DNS_test = b1_DNS[index_test]
b2_DNS_test = b2_DNS[index_test]
b4_DNS_test = b4_DNS[index_test]


# convert the numpy arrays to PyTorch tensors with float32 data type
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# create PyTorch datasets and dataloaders for the training and validation sets
# a TensorDataset wraps the feature and target tensors into a single dataset
# a DataLoader loads the data in batches and shuffles the batches if shuffle=True
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=my_batch_size)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=my_batch_size)

start_time = time.time()

# Instantiate a neural network
neural_net = ThePredictionMachine()


# Initialize the loss function
loss_fn = nn.MSELoss()

# Choose loss function, check out https://pytorch.org/docs/stable/optim.html for more info
# In this case we choose Stocastic Gradient Descent
optimizer = torch.optim.SGD(neural_net.parameters(), lr=learning_rate)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=epochs)

loss_v = np.zeros(epochs)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, neural_net, loss_fn, optimizer)
    test_loss = test_loop(test_loader, neural_net, loss_fn)
    loss_v[t] = test_loss

print("Done!")

preds = neural_net(X_test_tensor)
c_NN = preds.detach().numpy()
 
c_NN_old = c_NN

b1=c_NN[:,0]
b2=c_NN[:,1]
b4=c_NN[:,2]

a_11 = tau_DNS_test**2*dudy_DNS_test**2/12*(b2-6*b4)
uu_NN = (a_11+0.6666)*k_DNS_test

a_22 = tau_DNS_test**2*dudy_DNS_test**2/12*(b2+6*b4)
vv_NN = (a_22+0.6666)*k_DNS_test

a_33 = -tau_DNS_test**2*dudy_DNS_test**2/6*b2
ww_NN = (a_33+0.6666)*k_DNS_test

a_12 = b1*(k_komega_test/eps_komega_test)*np.gradient(u_komega_test,yplus_komega_test)/2
uv_NN = a_12*k_komega_test

b1_std=np.std(b1-b1_DNS_test)/(np.mean(b1.flatten()**2))**0.5
b2_std=np.std(b2-b2_DNS_test)/(np.mean(b2.flatten()**2))**0.5
b4_std=np.std(b4-b4_DNS_test)/(np.mean(b4.flatten()**2))**0.5

print('\nb1_error_std',b1_std)
print('\nb2_error_std',b2_std)
print('\nb4_error_std',b4_std)


fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.scatter(uu_NN,yplus_komega_test, marker="o", s=10, c="red", label="Neural Network")
ax1.plot(uu_DNS,yplus_DNS,'b-', label="Target")
plt.xlabel("$\overline{u'u'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig(name + 'uu_pred_re_2000.png')


fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.scatter(uv_NN,yplus_komega_test, marker="o", s=10, c="red", label="Neural Network")
ax1.plot(uv_DNS,yplus_DNS,'b-', label="Target")
plt.xlabel("$\overline{u'v'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig(name + 'uv_pred_re_2000.png')

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax1.scatter(vv_NN,yplus_komega_test, marker="o", s=10, c="red", label="Neural Network")
ax1.plot(vv_DNS,yplus_DNS,'b-', label="Target")
plt.xlabel("$\overline{v'^2}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
plt.savefig(name + 'vv_pred_re_2000.png')
