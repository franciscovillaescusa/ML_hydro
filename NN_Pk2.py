import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys,os
import matplotlib.pyplot as plt
from torchcontrib.optim import SWA


def model_ls(x,A,B):
    return A*x+B

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
        data[i] = np.log10(Pk)
        #data[i] = Pk
        if predict_gamma:  label[i] = [alpha, beta, gamma]
        else:              label[i] = [alpha, beta]
        i += 1
    return torch.tensor(data),torch.tensor(label)


####################################### INPUT #########################################
kmin   = 7e-3 #h/Mpc
kmaxs  = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][::-1] #h/Mpc
kpivot = 2.0

hidden1 = 50
hidden2 = 50
hidden3 = 50

predict_gamma = False

epochs           = 1000
batch_size_train = 16
batch_size_valid = 64*5
batch_size_test  = 10000
batches          = 100

fout = 'results/new_results_kpivot=2.0_no-gamma.txt'
#######################################################################################


# define the arrays containing the error on each parameter
dalpha = np.zeros(len(kmaxs), dtype=np.float64)
dbeta  = np.zeros(len(kmaxs), dtype=np.float64)
dgamma = np.zeros(len(kmaxs), dtype=np.float64)

# do a loop over the different kmax
for l,kmax in enumerate(kmaxs):

    print '\nWorking with kmax = %.2f'%kmax
    
    # find the fundamental frequency and the number of bins up to kmax
    kF   = kmin
    k_bins = int((kmax-kmin)/kF)

    # define the k-array
    k = np.arange(1,k_bins+2)*kF

    # find the number of modes in each k-bin: Nk = 2*k^2*dk/kF^3
    Nk = 4.0*np.pi*k**2*kF/kF**3    
    
    # get a validation dataset
    valid_data, valid_label = dataset(k, Nk, kpivot, batch_size_valid, predict_gamma)
    print valid_data
    print valid_data.shape
    print k.shape
    #data = np.empty((valid_data.shape[1], valid_data.shape[0]+1))
    #data[:,0] = np.log10(k)
    #for i in xrange(valid_data.shape[0]):
    #    data[:,i+1] = valid_data[i].numpy()
    #np.savetxt('borrar.txt', data)

    """
    # fit with least squares (no errorbars)
    M = np.vstack([np.log10(k), np.ones(len(k))]).T
    dA, dB = 0.0, 0.0
    for i in xrange(batch_size_valid):
        B1,A1 = np.linalg.lstsq(M, valid_data[i].numpy(), rcond=None)[0]
        A1 = 10**A1
        A2,B2 = valid_label[i,0].numpy(), valid_label[i,1].numpy()
        print '%.5f %.5f'%(A1,B1)
        print '%.5f %.5f'%(A2,B2)
        print ''
        dA += (A1-A2)**2
        dB += (B1-B2)**2
    print dA/batch_size_valid
    print dB/batch_size_valid

    # fit the least squares (errorbars)
    dA, dB = 0.0, 0.0
    for i in xrange(batch_size_valid):
        dPk = np.log10(np.sqrt(10**(valid_data[i].numpy()**2)/Nk))
        popt, pcov = curve_fit(model_ls, np.log10(k), valid_data[i].numpy(),
                               sigma=dPk, p0=valid_label[i].numpy())
        B1,A1 = popt[0], 10**(popt[1])
        A2,B2 = valid_label[i,0].numpy(), valid_label[i,1].numpy()
        print '%.5f %.5f'%(A1,B1)
        print '%.5f %.5f'%(A2,B2)
        print ''
        dA += (A1-A2)**2
        dB += (B1-B2)**2
    print dA/batch_size_valid
    print dB/batch_size_valid
    sys.exit()
    """

    
    # define the network, loss and optimizer
    if predict_gamma:  last_layer = 3
    else:              last_layer = 2
    net = Model(k.shape[0], hidden1, hidden2, hidden3, last_layer)
    loss_func = nn.MSELoss()

    #base_opt = torch.optim.SGD(net.parameters(), lr=0.01)
    #base_opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999),
    #                      eps=1e-8,amsgrad=False)

    #optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.002)
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.95, nesterov=True)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99),
                           eps=1e-8,amsgrad=False)


    # do a loop over the different epochs
    loss_train = np.zeros(epochs, dtype=np.float64)
    loss_valid = np.zeros(epochs, dtype=np.float64)
    done1, done2, done3 = False, False, False
    min_train, min_eval = 1e7, 1e7

    # do a loop over the different epochs
    for epoch in xrange(epochs): 

        total_loss = 0
        for batch in xrange(batches):
    
            data, label = dataset(k, Nk, kpivot, batch_size_train, predict_gamma)
            pred = net(data)
            loss = loss_func(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
        
        loss_train[epoch] = total_loss/batches
        
        # compute the loss for the validation set
        pred = net(valid_data)
        loss_valid[epoch] = loss_func(pred, valid_label).detach()

        # save model if it is better
        if loss_valid[epoch]<min_eval:
            print 'saving model; epoch %d; %.3e %.3e'\
                %(epoch,loss_train[epoch],loss_valid[epoch])
            torch.save(net.state_dict(), 'results/best_model_kmax=%.2f.pt'%kmax)
            min_train, min_eval = loss_train[epoch], loss_valid[epoch]

        """
        if loss_valid[epoch].item()<1e-3 and not(done1):
            #lr = 5e-4
            lr = 0.002/5.0
            for g in optimizer.param_groups:
                g['lr'] = lr
            done1 = True
            print lr

        if loss_valid[epoch].item()<1e-4 and not(done2):
            #lr = 2e-4
            lr /= 5.0
            for g in optimizer.param_groups:
                g['lr'] = lr
            done2 = True
            print lr

        if loss_valid[epoch].item()<1e-5 and not(done3):
            #lr = 1e-4
            lr /= 5.0
            for g in optimizer.param_groups:
                g['lr'] = lr
            done3 = True
            print lr
        """

        plt.cla() #clear axes
        plt.yscale('log')
        plt.plot(loss_train[:epoch])
        plt.plot(loss_valid[:epoch])
        plt.pause(0.0001)
    
        #np.savetxt('results/loss_kmax=%.2f.txt'%kmax, np.transpose([loss_train,loss_test]))



    ###### evaluate the performance of the model ######
    test_data, test_label = dataset(k, Nk, kpivot, batch_size_test,predict_gamma)

    net = Model(k.shape[0],hidden1,hidden2,hidden3,last_layer)
    net.load_state_dict(torch.load('results/best_model_kmax=%.2f.pt'%kmax))
    net.eval()
        
    pred = net(test_data)

    dalpha[l] = np.mean(((pred[:,0]-test_label[:,0])**2).detach().numpy())
    print 'error alpha = %.3e'%dalpha[l]
        
    dbeta[l]  = np.mean(((pred[:,1]-test_label[:,1])**2).detach().numpy())
    print 'error beta  = %.3e'%dbeta[l]
        
    if predict_gamma:
        dgamma[l] = np.mean(((pred[:,2]-test_label[:,2])**2).detach().numpy())        
        print 'error gamma = %.3e'%dgamma[l]
    ###################################################

# save results to file
if predict_gamma:  np.savetxt(fout, np.transpose([kmaxs, dalpha, dbeta, dgamma]))
else:              np.savetxt(fout, np.transpose([kmaxs, dalpha, dbeta]))
