import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys,os
import data as data
import architecture as architecture


####################################### INPUT ####################################
# k-values
kmin  = 7e-3 #h/Mpc
#kmaxs = [0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][::-1] #h/Mpc
kmax = 0.9 #h/Mpc

# model parameters
kpivot    = 2.0
predict_C = False
#suffix    = 'BN_100x100x100x100_15000_2500-5000-7500_10000_12500_kpivot=0.5_noC'
#suffix    = 'BN_100x100x100x100_kpivot=2.0_noC'
suffix    = 'BN_100x100x100_kpivot=2.0_noC'
fout      = 'results/results_%s.txt'%suffix

# architecture parameters
hidden1 = 100
hidden2 = 100
hidden3 = 100
hidden4 = 100

# training parameters
batch_size_test  = 70000

plot_results = True
##################################################################################

# find the fundamental frequency, the number of bins up to kmax and the k-array
kF     = kmin
k_bins = int((kmax-kmin)/kF)
k      = np.arange(2,k_bins+2)*kF #avoid k=kF as we will get some negative values
Nk     = 4.0*np.pi*k**2*kF/kF**3  #number of modes in each k-bin

# find the number of neurons in the output layer and define loss
if predict_C:  last_layer = 3
else:          last_layer = 2

# get the test dataset
test_data, test_label = data.dataset(k, Nk, kpivot, batch_size_test, predict_C)

# get architecture and load best-model
net = architecture.Model(k.shape[0],hidden1,hidden2,hidden3,hidden4,last_layer)
fmodel = 'results/best_model_%s_kmax=%.2f.pt'%(suffix,kmax)
net.load_state_dict(torch.load(fmodel))

net.eval()
with torch.no_grad():
    pred = net(test_data)
print('kmax = %.2f'%kmax)

A_pred, B_pred = pred[:,0]*9.9 + 0.1, pred[:,1]
A_test, B_test = test_label[:,0]*9.9 + 0.1, test_label[:,1]

dA = np.mean(((A_pred - A_test)**2).numpy())
dB = np.mean(((B_pred - B_test)**2).numpy())

print('error A = %.3e'%dA)       
print('error B = %.3e'%dB)
        
if predict_C:
    dC = np.mean(((pred[:,2]-test_label[:,2])**2).numpy())        
    print('error C = %.3e'%dC)


    
