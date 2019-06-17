# This scripts is an example of a MCMC fit. We generate points from a linear law
# y = a*x+b and we add noise to them. We then try to obtain the values of a and b
# from the data

import numpy as np
#import emcee,corner
from scipy.optimize import minimize
import sys,os
import torch
import torch.nn as nn
import torch.nn.functional as F



# define the arquitecture of the network
class Model(nn.Module):
    def __init__(self,k_bins,hidden1,hidden2,hidden3,last_layer):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(k_bins,  hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, last_layer)

    def forward(self,k):
        out = F.relu(self.fc1(k))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

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

# create the data to train the network
def dataset(k, Nk, kpivot, batch_size, predict_gamma):
    data  = np.empty((batch_size, k.shape[0]), dtype=np.float32)
    if predict_gamma:  label = np.empty((batch_size,3), dtype=np.float32)
    else:              label = np.empty((batch_size,2), dtype=np.float32)
    indexes = np.where(k>kpivot)[0]
    i = 0
    while(i<batch_size):
        alpha  = (9.9*np.random.random()+0.1)
        beta   = -1.0 + 1.0*np.random.random()
        gamma  = -0.5 + np.random.random()
        Pk     = alpha*k**beta
        if len(indexes)>0:
            A      = Pk[indexes[0]]/k[indexes[0]]**gamma
            Pk[indexes] = A*k[indexes]**gamma
        dPk    = np.sqrt(Pk**2/Nk)
        Pk     = np.random.normal(loc=Pk, scale=dPk)
        if np.any(Pk<0):
            continue
        #data[i] = np.log10(Pk)
        data[i] = Pk
        if predict_gamma:  label[i] = [alpha, beta, gamma]
        else:              label[i] = [alpha, beta]
        i += 1
    return data,label


####################################### INPUT #########################################
kmin   = 7e-3 #h/Mpc
kmaxs  = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][::-1] #h/Mpc
kpivot = 2.0

hidden1 = 70
hidden2 = 70
hidden3 = 70
last_layer = 2
predict_gamma = False

batch_size_valid = 64*100
#######################################################################################

dA = np.zeros(len(kmaxs), dtype=np.float64)
dB = np.zeros(len(kmaxs), dtype=np.float64)

# do a loop over the different kmax
for l,kmax in enumerate(kmaxs):

    # find the fundamental frequency and the number of bins up to kmax
    kF     = kmin
    k_bins = int((kmax-kmin)/kF)

    # define the k-array
    k = np.arange(1,k_bins+2)*kF

    # find the number of modes in each k-bin: Nk = 2*k^2*dk/kF^3
    Nk = 4.0*np.pi*k**2*kF/kF**3

    # find the degrees of freedom
    ndof = len(k)-2.0
    
    # get a validation dataset
    valid_data, valid_label = dataset(k, Nk, kpivot, batch_size_valid, predict_gamma)

    # fit results to a power law
    for i in xrange(batch_size_valid):

        # read data and compute errorbars
        Pk     = valid_data[i]
        A1, B1 = valid_label[i]
        dPk    = np.sqrt(Pk**2/Nk)
        
        # fit data to function
        chi2_func = lambda *args: -lnlike_theory(*args)
        best_fit  = minimize(chi2_func, [A1,B1], args=(k,Pk,dPk), method='Powell')
        theta_best_fit = best_fit["x"]
        chi2 = chi2_func(theta_best_fit, k, Pk, dPk)*1.0/ndof

        # get results
        A2, B2 = theta_best_fit

        # compute accumulated error
        dA[l] = dA[l] + (A1-A2)**2
        dB[l] = dB[l] + (B1-B2)**2

    dA[l] = dA[l]/batch_size_valid
    dB[l] = dB[l]/batch_size_valid
    print '%.2f %.3e %.3e'%(kmax, dA[l], dB[l])


    # get the neural network arquitecture
    net = Model(k.shape[0], hidden1, hidden2, hidden3, last_layer)
    net.load_state_dict(torch.load('../results/best_model_kmax=%.2f.pt'%kmax))
    net.eval()
        
    pred = net(torch.tensor(np.log10(valid_data)))
    pred = pred.detach().numpy()
    dA2, dB2 = 0.0, 0.0
    for i in xrange(batch_size_valid):
        dA2 = dA2 + (pred[i,0]-valid_label[i,0])**2
        dB2 = dB2 + (pred[i,1]-valid_label[i,1])**2
    dA2 = dA2/batch_size_valid
    dB2 = dB2/batch_size_valid
    print '%.2f %.3e %.3e'%(kmax, dA2, dB2)
    print '%.2f %.2f %.2f'%(kmax, dA2/dA[l], dB2/dB[l])
    
np.savetxt('results_fit.txt', np.transpose([kmaxs, dA, dB]))

