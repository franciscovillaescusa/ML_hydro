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
    def __init__(self,k_bins,hidden1,hidden2,hidden3,hidden4,last_layer):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(k_bins,  hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, hidden4)
        self.fc5 = nn.Linear(hidden4, last_layer)

    def forward(self,Pk):
        out = F.relu(self.fc1(Pk))
        #out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
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
def dataset(k, Nk, kpivot, batch_size, predict_C):
    data  = np.empty((batch_size, k.shape[0]), dtype=np.float32)
    if predict_C:  label = np.empty((batch_size,3), dtype=np.float32)
    else:          label = np.empty((batch_size,2), dtype=np.float32)
    indexes = np.where(k>kpivot)[0]
    i = 0
    while(i<batch_size):
        A = (9.9*np.random.random()+0.1)
        B = -1.0 + 1.0*np.random.random()
        C = -0.5 + np.random.random()
        Pk = A*k**B
        if len(indexes)>0:
            A_value = Pk[indexes[0]]/k[indexes[0]]**C
            Pk[indexes] = A_value*k[indexes]**C
        dPk = np.sqrt(Pk**2/Nk)
        Pk  = np.random.normal(loc=Pk, scale=dPk)
        #if np.any(Pk<0):
        #    continue
        #data[i] = np.log10(Pk)
        data[i] = Pk
        if predict_C:  label[i] = [A, B, C]
        else:          label[i] = [A, B]
        i += 1
    return data,label


####################################### INPUT #########################################
kmin   = 7e-3 #h/Mpc
kmaxs  = [0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #h/Mpc
kpivot = 2.0

hidden1 = 50
hidden2 = 50
hidden3 = 50
hidden4 = 50
last_layer = 2
predict_C = False

batch_size_valid = 64*100

suffix = '50x50x50_40000_5000-10000-15000_20000_kpivot=2.0_noC'
#######################################################################################

dA1 = np.zeros(len(kmaxs), dtype=np.float64)
dB1 = np.zeros(len(kmaxs), dtype=np.float64)
dA2 = np.zeros(len(kmaxs), dtype=np.float64)
dB2 = np.zeros(len(kmaxs), dtype=np.float64)

# do a loop over the different kmax
for l,kmax in enumerate(kmaxs):

    # find the fundamental frequency and the number of bins up to kmax
    kF     = kmin
    k_bins = int((kmax-kmin)/kF)
    k      = np.arange(2,k_bins+2)*kF
    Nk     = 4.0*np.pi*k**2*kF/kF**3
    ndof   = len(k)-2.0 #number of degrees of freedom
    
    # get a validation dataset
    valid_data, valid_label = dataset(k, Nk, kpivot, batch_size_valid, predict_C)

    
    # get the neural network arquitecture
    net = Model(k.shape[0], hidden1, hidden2, hidden3, hidden4, last_layer)
    net.load_state_dict(torch.load('../results/best_model_%s_kmax=%.2f.pt'%(suffix,kmax)))
    net.eval()
    
    
    # fit results to a power law
    for i in xrange(batch_size_valid):

        # read data and compute errorbars
        Pk      = valid_data[i]
        A, B    = valid_label[i]
        Pk_true = A*k**B
        dPk     = np.sqrt(Pk_true**2/Nk)
        
        # fit data to function
        chi2_func = lambda *args: -lnlike_theory(*args)
        best_fit  = minimize(chi2_func, [A,B], args=(k,Pk,dPk), method='Powell')
        theta_best_fit = best_fit["x"]
        chi2 = chi2_func(theta_best_fit, k, Pk, dPk)*1.0/ndof

        # get results
        A1, B1 = theta_best_fit

        # compute accumulated error
        dA1[l] = dA1[l] + (A-A1)**2
        dB1[l] = dB1[l] + (B-B1)**2

        # get prediction from neural network
        A2,B2 = net(torch.tensor(np.log10(valid_data[i])))
        A2,B2 = A2.detach().numpy(), B2.detach().numpy()
        chi2_NN = chi2_func([A2, B2], k, Pk, dPk)*1.0/ndof

        # compute accumulated error
        dA2[l] = dA2[l] + (A-A2)**2
        dB2[l] = dB2[l] + (B-B2)**2
        
        #print i
        #print 'A=%.5f \t B=%.5f'%(A1,B1)
        #print 'A=%.5f \t B=%.5f \t chi2=%.5f'%(A2,B2,chi2)
        #print 'A=%.5f \t B=%.5f \t chi2=%.5f'%(A3,B3,chi2_NN)
        #print ' '

        #if chi2_NN<chi2:
        #    print i, chi2, chi2_NN
        #    raise Exception('chi2 of NN is better than standard chi2')

        #np.savetxt('borrar.txt', np.transpose([k,Pk,dPk]))

    dA1[l] = dA1[l]/batch_size_valid
    dB1[l] = dB1[l]/batch_size_valid
    dA2[l] = dA2[l]/batch_size_valid
    dB2[l] = dB2[l]/batch_size_valid
    print '%.2f %.3e %.3e'%(kmax, dA1[l], dB1[l])
    print '%.2f %.3e %.3e'%(kmax, dA2[l], dB2[l])
    print ''


    """
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
    """
    
np.savetxt('results_fit.txt', np.transpose([kmaxs, dA1, dB1, dA2, dB2]))








