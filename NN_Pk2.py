# This script trains the model and save the best model to file.
# That model can later be read and evaluated its accuracy
from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys,os
import matplotlib.pyplot as plt
#from torchcontrib.optim import SWA


###### MPI DEFINITIONS ###### 
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()


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
hidden4 = 50

predict_gamma = False

epochs           = 20000
batch_size_train = 16
batch_size_valid = 64*5
batch_size_test  = 64*100
batches          = 100

plot_results = False

suffix = '50x50x50_20000_5000-10000-15000_kpivot=2.0_no-gamma'
fout = 'results/results_%s.txt'%suffix
#######################################################################################

# find the numbers that each cpu will work with
numbers = np.where(np.arange(len(kmaxs))%nprocs==myrank)[0]

# define the arrays containing the error on each parameter
dalpha   = np.zeros(len(kmaxs), dtype=np.float64)
dalpha_t = np.zeros(len(kmaxs), dtype=np.float64)
dbeta    = np.zeros(len(kmaxs), dtype=np.float64)
dbeta_t  = np.zeros(len(kmaxs), dtype=np.float64)
dgamma   = np.zeros(len(kmaxs), dtype=np.float64)
dgamma_t = np.zeros(len(kmaxs), dtype=np.float64)

# do a loop over the different kmax
for l in numbers:
    
    kmax = kmaxs[l]
    print '\nWorking with kmax = %.2f'%kmax
    
    # find the fundamental frequency, the number of bins up to kmax and the k-array
    kF     = kmin
    k_bins = int((kmax-kmin)/kF)
    k      = np.arange(1,k_bins+2)*kF
    Nk     = 4.0*np.pi*k**2*kF/kF**3  #number of modes in each k-bin
    
    # get a validation dataset
    valid_data, valid_label = dataset(k, Nk, kpivot, batch_size_valid, predict_gamma)
    #data = np.empty((valid_data.shape[1], valid_data.shape[0]+1))
    #data[:,0] = np.log10(k)
    #for i in xrange(valid_data.shape[0]):
    #    data[:,i+1] = valid_data[i].numpy()
    #np.savetxt('borrar.txt', data)

    
    # define the network and loss
    if predict_gamma:  last_layer = 3
    else:              last_layer = 2
    net = Model(k.shape[0], hidden1, hidden2, hidden3, hidden4, last_layer)
    loss_func = nn.MSELoss()

    # do a loop over the different epochs
    loss_train = np.zeros(epochs, dtype=np.float64)
    loss_valid = np.zeros(epochs, dtype=np.float64)
    done1, done2, done3 = False, False, False
    min_train, min_eval = 1e7, 1e7

    # get the name of the model file
    fmodel = 'results/best_model_%s_kmax=%.2f.pt'%(suffix,kmax)

    # do a loop over the different epochs
    for epoch in xrange(epochs): 

        # define the optimizer here
        if epoch==0:
            #base_opt = torch.optim.SGD(net.parameters(), lr=0.01)
            #base_opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999),
            #                      eps=1e-8,amsgrad=False)
            #optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.002)
            #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.95,
            #                      nesterov=True)
            optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999),
                                   eps=1e-8,amsgrad=False)            
        
        # after 1000 epochs load the best model and decrease the learning rate
        if epoch==4999:
            net = Model(k.shape[0],hidden1,hidden2,hidden3,hidden4,last_layer)
            net.load_state_dict(torch.load(fmodel))
            optimizer = optim.Adam(net.parameters(), lr=0.001/2.0, betas=(0.9, 0.999),
                                   eps=1e-8,amsgrad=False)

        # after 2000 epochs load the best model and decrease the learning rate
        if epoch==9999:
            net = Model(k.shape[0],hidden1,hidden2,hidden3,hidden4,last_layer)
            net.load_state_dict(torch.load(fmodel))
            optimizer = optim.Adam(net.parameters(), lr=0.001/4.0, betas=(0.9, 0.999),
                                   eps=1e-8,amsgrad=False)
            #optimizer = optim.SGD(net.parameters(), lr=0.001/5.0, momentum=0.95,
            #                      nesterov=True)
        
        # after 2000 epochs load the best model and decrease the learning rate
        if epoch==14999:
            net = Model(k.shape[0],hidden1,hidden2,hidden3,hidden4,last_layer)
            net.load_state_dict(torch.load(fmodel))
            optimizer = optim.Adam(net.parameters(), lr=0.001/8.0, betas=(0.9, 0.999),
                                   eps=1e-8,amsgrad=False)

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
            print 'saving model; kmax %.2f epoch %d; %.3e %.3e'\
                %(kmax,epoch,loss_train[epoch],loss_valid[epoch])
            torch.save(net.state_dict(), fmodel)
            min_train, min_eval = loss_train[epoch], loss_valid[epoch]

        # plot the losses
        if plot_results:
            plt.cla() #clear axes
            plt.yscale('log')
            plt.plot(loss_train[:epoch])
            plt.plot(loss_valid[:epoch])
            plt.pause(0.0001)



    ###### evaluate the performance of the model ######
    test_data, test_label = dataset(k, Nk, kpivot, batch_size_test, predict_gamma)

    net = Model(k.shape[0],hidden1,hidden2,hidden3,hidden4,last_layer)
    net.load_state_dict(torch.load(fmodel))
    net.eval()
        
    pred = net(test_data)
    print 'kmax = %.2f'%kmax
    
    dalpha[l] = np.mean(((pred[:,0]-test_label[:,0])**2).detach().numpy())
    print 'error alpha = %.3e'%dalpha[l]
        
    dbeta[l]  = np.mean(((pred[:,1]-test_label[:,1])**2).detach().numpy())
    print 'error beta  = %.3e'%dbeta[l]
        
    if predict_gamma:
        dgamma[l] = np.mean(((pred[:,2]-test_label[:,2])**2).detach().numpy())        
        print 'error gamma = %.3e'%dgamma[l]
    ###################################################

# reduce the results
comm.Reduce(dalpha, dalpha_t, root=0)
comm.Reduce(dbeta,  dbeta_t,  root=0)
comm.Reduce(dgamma, dgamma_t, root=0)
    
# save results to file
if myrank>0:  sys.exit()
if predict_gamma:  np.savetxt(fout, np.transpose([kmaxs, dalpha_t, dbeta_t, dgamma_t]))
else:              np.savetxt(fout, np.transpose([kmaxs, dalpha_t, dbeta_t]))

