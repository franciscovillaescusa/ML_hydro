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
kmaxs = [0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][::-1] #h/Mpc

# model parameters
kpivot    = 2.0
predict_C = False
suffix    = '100x100x100_kpivot=%.2f_noC'%kpivot

# architecture parameters
hidden1 = 100
hidden2 = 100
hidden3 = 100
hidden4 = 100

# training parameters
batch_size_test  = 70000

fout = 'errors.txt'
##################################################################################

# verbose
if predict_C:  print('kmax\t   dA   \t   dB   \t   dC')
else:          print('kmax\t   dA   \t   dB')

f = open(fout, 'w')
for kmax in kmaxs:

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

    # get NN prediction
    net.eval()
    with torch.no_grad():
        pred = net(test_data)

    A_pred, B_pred = pred[:,0]*9.9 + 0.1,       pred[:,1]
    A_test, B_test = test_label[:,0]*9.9 + 0.1, test_label[:,1]

    dA = np.sqrt(np.mean(((A_pred - A_test)**2).numpy()))
    dB = np.sqrt(np.mean(((B_pred - B_test)**2).numpy()))
        
    if predict_C:
        dC = np.sqrt(np.mean(((pred[:,2]-test_label[:,2])**2).numpy()))
        print('%.2f\t%.3e\t%.3e\t%.3e'%(kmax,dA,dB,dC))
        f.write('%.2f %.4e %.4e %.4e\n'%(kmax,dA,dB,dC))
    else:
        print('%.2f\t%.3e\t%.3e'%(kmax,dA,dB))
        f.write('%.2f %.4e %.4e\n'%(kmax,dA,dB))
f.close()
