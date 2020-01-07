# This scripts is an example of a MCMC fit. We generate points from a linear law
# y = a*x+b and we add noise to them. We then try to obtain the values of a and b
# from the data

import numpy as np
from scipy.optimize import minimize
import sys,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import architecture
import data


# This function returns the predicted y-values for the particular model considered
# x --------> array with the x-values of the data
# theta ----> array with the value of the model parameters [A,B]
def model_theory(x,theta):
    A,B = theta
    return A*x**B

# This functions returns the log-likelihood for the theory model
# theta -------> array with parameters to fit [A,B]
# x,y,dy ------> data to fit
def lnlike_theory(theta,x,y,dy):
    # put priors here
    #if (theta[0]<0.1) or (theta[0]>10.0) or (theta[1]<-1.0) or (theta[1]>0.0):
    #    return -np.inf
    #else:
    y_model = model_theory(x,theta)
    chi2 = -np.sum(((y-y_model)/dy)**2, dtype=np.float64)
    return chi2


####################################### INPUT #######################################
# k-bins
kmin  = 7e-3 #h/Mpc
kmaxs = [0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][::-1] #h/Mpc

# model parameters
kpivot    = 2.0
predict_C = False
#suffix    = '20x20x20_BS=128_noSche_Adam1_kpivot=%.2f_noC'%kpivot
#suffix    = '100x100x100_BS=128_noSche_Adam1_kpivot=%.2f_noC'%kpivot
#suffix    = '50x50x50_kpivot=%.2f_noC'%kpivot
suffix    = 'Pk-30-30-30-2_BS=256_batches=32_noSche_Adam_lr=1e-3_kpivot=%.2f'%kpivot

# architecture parameters
model   = 'model1' #'model1', 'model0'
hidden1 = 100
hidden2 = 100
hidden3 = 100
hidden4 = 100
hidden5 = 100
hidden  = 30

# training parameters
test_set_size = 6400

fout = 'errors_fit_30x30x30.txt'
#####################################################################################

# find the number of neurons in the output layer and define loss
if predict_C:  last_layer = 3
else:          last_layer = 2

# define the arrays containing the errors on the parameters
dA_LS = np.zeros(len(kmaxs), dtype=np.float64)
dB_LS = np.zeros(len(kmaxs), dtype=np.float64)
dA_NN = np.zeros(len(kmaxs), dtype=np.float64)
dB_NN = np.zeros(len(kmaxs), dtype=np.float64)

# define the chi2 function
chi2_func = lambda *args: -lnlike_theory(*args)

# do a loop over the different kmax
for l,kmax in enumerate(kmaxs):

    # define the arrays containing the values of the chi2
    chi2_LS, chi2_NN = np.zeros(test_set_size), np.zeros(test_set_size)
    
    # find the fundamental frequency and the number of bins up to kmax
    kF     = kmin
    k_bins = int((kmax-kmin)/kF)
    k      = np.arange(2,k_bins+2)*kF
    Nk     = 4.0*np.pi*k**2*kF/kF**3
    ndof   = len(k)-2.0 #number of degrees of freedom
    
    # get a test dataset
    test_data, test_label = data.dataset(k, Nk, kpivot, test_set_size, predict_C)
    
    # get the architecture
    if   model=='model0':
        net = architecture.Model(k.shape[0], hidden1, hidden2, hidden3, hidden4, 
                                 hidden5, last_layer)
    elif model=='model1':
        net = architecture.Model1(k.shape[0], hidden, last_layer)
    else:  raise Exception('Wrong model!')


    net.load_state_dict(torch.load('results/best_model_%s_kmax=%.2f.pt'%(suffix,kmax)))
    net.eval()
    
    # fit results to a power law
    for i in range(test_set_size):

        # read data and compute errorbars
        log10_Pk_data = test_data[i].numpy()
        Pk_data       = 10**(log10_Pk_data)
        A, B          = test_label[i].numpy()
        A_true        = A*9.9 + 0.1
        B_true        = B
        Pk_true       = (A_true*k**B_true)
        dPk_true      = np.sqrt(Pk_true**2/Nk)

        ######## LEAST SQUARES ########
        # fit data to function
        best_fit  = minimize(chi2_func, [A_true,B_true], args=(k, Pk_data, dPk_true),
                             method='Powell')
        theta_best_fit = best_fit["x"]
        A_LS, B_LS = theta_best_fit

        # compute chi2 and accumulated error
        chi2_LS[i] = chi2_func(theta_best_fit, k, Pk_data, dPk_true)*1.0/ndof
        dA_LS[l] += (A_true - A_LS)**2
        dB_LS[l] += (B_true - B_LS)**2
        ###############################
        
        ####### NEURAL NETWORK ########
        with torch.no_grad():
            A_NN, B_NN = net(test_data[i])
        A_NN, B_NN = A_NN.numpy()*9.9 + 0.1, B_NN.numpy()
        
        # compute chi2 and accumulated error
        chi2_NN[i] = chi2_func([A_NN, B_NN], k, Pk_data, dPk_true)*1.0/ndof
        dA_NN[l] += (A_true - A_NN)**2
        dB_NN[l] += (B_true - B_NN)**2
        ###############################

    dA_LS[l] = np.sqrt(dA_LS[l]/test_set_size)
    dB_LS[l] = np.sqrt(dB_LS[l]/test_set_size)
    dA_NN[l] = np.sqrt(dA_NN[l]/test_set_size)
    dB_NN[l] = np.sqrt(dB_NN[l]/test_set_size)
    print('%.2f %.3e %.3e'%(kmax, dA_LS[l], dB_LS[l]))
    print('%.2f %.3e %.3e'%(kmax, dA_NN[l], dB_NN[l]))
    print('%.2f %.3f %.3f\n'%(kmax, dA_NN[l]/dA_LS[l], dB_NN[l]/dB_LS[l]))

np.savetxt(fout, np.transpose([kmaxs, dA_LS, dB_LS, dA_NN, dB_NN]))








