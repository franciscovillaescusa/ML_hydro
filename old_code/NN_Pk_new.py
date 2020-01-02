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
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.bn4 = nn.BatchNorm1d(hidden4)

    def forward(self,Pk):
        out = self.bn1(F.leaky_relu(self.fc1(Pk)))
        out = self.bn2(F.leaky_relu(self.fc2(out)))
        out = self.bn3(F.leaky_relu(self.fc3(out)))
        out = self.bn4(F.leaky_relu(self.fc4(out)))
        out = self.fc5(out)
        return out
    
# create the data to train/validate/test the network
def dataset(k, Nk, kpivot, batch_size, predict_C):

    # get the values of the parameters/labels
    A  = (9.9*np.random.random(batch_size)+0.1)
    B  = -1.0 + 1.0*np.random.random(batch_size)
    C  = -0.5 + np.random.random(batch_size)
    if predict_C:
        label = np.array([A, B, C], dtype=np.float32)
    else:
        label = np.array([A, B],    dtype=np.float32)

    # compute Pk
    Pk = np.zeros((batch_size, k.shape[0]), dtype=np.float32)
    for i in xrange(batch_size):
        Pk[i] = A[i]*k**B[i]

    # get the hydro Pk part
    indexes = np.where(k>kpivot)[0]
    if len(indexes)>0:
        A_value = Pk[:,indexes[0]]/k[indexes[0]]**C
        for i in xrange(batch_size):
            Pk[i,indexes] = A_value[i]*k[indexes]**C[i]

    # add cosmic variance
    dPk = np.sqrt(Pk**2/Nk)
    Pk  = np.random.normal(loc=Pk, scale=dPk)

    # save data to make plots
    #Pk_plot = np.zeros((batch_size+1,k.shape[0]), dtype=np.float32)
    #Pk_plot[0]  = k
    #Pk_plot[1:] = Pk
    #np.savetxt('borrar.txt', np.transpose(Pk_plot))
    
    # return data
    data = np.log10(Pk).astype(np.float32) #Pk
    return torch.tensor(data),torch.tensor(label)


####################################### INPUT #########################################
kmin   = 7e-3 #h/Mpc
kmaxs  = [0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][::-1] #h/Mpc
kpivot = 0.5

hidden1 = 100
hidden2 = 100
hidden3 = 100
hidden4 = 100

predict_C = False

epochs           = 15000
batch_size_train = 32
batch_size_valid = 64*10
batch_size_test  = 64*100
batches          = 100

plot_results = False

suffix = 'BN_100x100x100x100_15000_2500-5000-7500_10000_12500_kpivot=2.0_noC'
fout   = 'results/results_%s.txt'%suffix
#######################################################################################

# find the numbers that each cpu will work with
numbers = np.where(np.arange(len(kmaxs))%nprocs==myrank)[0]

# define the arrays containing the error on each parameter
dA   = np.zeros(len(kmaxs), dtype=np.float64)
dA_t = np.zeros(len(kmaxs), dtype=np.float64)
dB   = np.zeros(len(kmaxs), dtype=np.float64)
dB_t = np.zeros(len(kmaxs), dtype=np.float64)
dC   = np.zeros(len(kmaxs), dtype=np.float64)
dC_t = np.zeros(len(kmaxs), dtype=np.float64)

# do a loop over the different kmax
for l in numbers:
    
    kmax = kmaxs[l]
    print '\nWorking with kmax = %.2f'%kmax
    
    # find the fundamental frequency, the number of bins up to kmax and the k-array
    kF     = kmin
    k_bins = int((kmax-kmin)/kF)
    k      = np.arange(2,k_bins+2)*kF #avoid k=kF as we will get some negative values
    Nk     = 4.0*np.pi*k**2*kF/kF**3  #number of modes in each k-bin
    
    # get a validation dataset
    valid_data, valid_label = dataset(k, Nk, kpivot, batch_size_valid, predict_C)
    
    # find the number of neurons in the output layer and define loss
    if predict_C:  last_layer = 3
    else:          last_layer = 2
    loss_func = nn.MSELoss()

    # do a loop over the different epochs
    loss_train = np.zeros(epochs, dtype=np.float64)
    loss_valid = np.zeros(epochs, dtype=np.float64)
    min_train, min_eval = 1e7, 1e7

    # get the name of the model file
    fmodel = 'results/best_model_%s_kmax=%.2f.pt'%(suffix,kmax)

    # do a loop over the different epochs
    for epoch in xrange(epochs): 

        # define the optimizer here
        if epoch==0:
            net = Model(k.shape[0], hidden1, hidden2, hidden3, hidden4, last_layer)
            #base_opt = torch.optim.SGD(net.parameters(), lr=0.01)
            #base_opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999),
            #                      eps=1e-8,amsgrad=False)
            #optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.002)
            #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.95,
            #                      nesterov=True)
            if os.path.exists(fmodel):
                net.load_state_dict(torch.load(fmodel))
            optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999),
                                   eps=1e-8,amsgrad=False)            
        
        # after 1000 epochs load the best model and decrease the learning rate
        if epoch==2500:
            net = Model(k.shape[0],hidden1,hidden2,hidden3,hidden4,last_layer)
            net.load_state_dict(torch.load(fmodel))
            optimizer = optim.Adam(net.parameters(), lr=0.001/2.0, betas=(0.9, 0.999),
                                   eps=1e-8,amsgrad=False)

        # after 2000 epochs load the best model and decrease the learning rate
        if epoch==5000:
            net = Model(k.shape[0],hidden1,hidden2,hidden3,hidden4,last_layer)
            net.load_state_dict(torch.load(fmodel))
            optimizer = optim.Adam(net.parameters(), lr=0.001/4.0, betas=(0.9, 0.999),
                                   eps=1e-8,amsgrad=False)
        
        # after 2000 epochs load the best model and decrease the learning rate
        if epoch==7500:
            net = Model(k.shape[0],hidden1,hidden2,hidden3,hidden4,last_layer)
            net.load_state_dict(torch.load(fmodel))
            optimizer = optim.Adam(net.parameters(), lr=0.001/8.0, betas=(0.9, 0.999),
                                   eps=1e-8,amsgrad=False)

        # after 2000 epochs load the best model and decrease the learning rate
        if epoch==10000:
            net = Model(k.shape[0],hidden1,hidden2,hidden3,hidden4,last_layer)
            net.load_state_dict(torch.load(fmodel))
            optimizer = optim.Adam(net.parameters(), lr=0.001/16.0, betas=(0.9, 0.999),
                                   eps=1e-8,amsgrad=False)

        # after 2000 epochs load the best model and decrease the learning rate
        if epoch==12500:
            net = Model(k.shape[0],hidden1,hidden2,hidden3,hidden4,last_layer)
            net.load_state_dict(torch.load(fmodel))
            optimizer = optim.Adam(net.parameters(), lr=0.001/32.0, betas=(0.9, 0.999),
                                   eps=1e-8,amsgrad=False)

        total_loss = 0
        for batch in xrange(batches):
    
            data, label = dataset(k, Nk, kpivot, batch_size_train, predict_C)
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
    test_data, test_label = dataset(k, Nk, kpivot, batch_size_test, predict_C)

    net = Model(k.shape[0],hidden1,hidden2,hidden3,hidden4,last_layer)
    net.load_state_dict(torch.load(fmodel))
    net.eval()
        
    pred = net(test_data)
    print 'kmax = %.2f'%kmax
    
    dA[l] = np.mean(((pred[:,0]-test_label[:,0])**2).detach().numpy())
    print 'error A = %.3e'%dA[l]
        
    dB[l] = np.mean(((pred[:,1]-test_label[:,1])**2).detach().numpy())
    print 'error B = %.3e'%dB[l]
        
    if predict_C:
        dC[l] = np.mean(((pred[:,2]-test_label[:,2])**2).detach().numpy())        
        print 'error C = %.3e'%dC[l]
    ###################################################

# reduce the results
comm.Reduce(dA, dA_t, root=0)
comm.Reduce(dB, dB_t, root=0)
comm.Reduce(dC, dC_t, root=0)
    
# save results to file
if myrank>0:  sys.exit()
if predict_C:  np.savetxt(fout, np.transpose([kmaxs, dA_t, dB_t, dC_t]))
else:          np.savetxt(fout, np.transpose([kmaxs, dA_t, dB_t]))
