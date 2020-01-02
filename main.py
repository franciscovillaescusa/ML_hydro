# This script trains the model and save the best model to file.
# That model can later be read and evaluated its accuracy
from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys,os
import matplotlib.pyplot as plt
import data as data
import architecture as architecture


###### MPI DEFINITIONS ###### 
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()


####################################### INPUT #########################################
# k-values
kmin  = 7e-3 #h/Mpc
kmaxs = [0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][::-1] #h/Mpc

# model parameters
kpivot    = 2.0
predict_C = False
suffix    = '100x100x100_kpivot=%.2f_noC'%kpivot
fout      = 'results/results_%s.txt'%suffix

# architecture parameters
hidden1 = 100
hidden2 = 100
hidden3 = 100
hidden4 = 100

# training parameters
epochs           = 50000
batch_size_train = 16
batch_size_valid = 64*300
batches          = 200
max_lr           = 1e-4
min_lr           = 1e-6

plot_results = False
#######################################################################################

# find the numbers that each cpu will work with
numbers = np.where(np.arange(len(kmaxs))%nprocs==myrank)[0]

# do a loop over the different kmax
for l in numbers:
    
    kmax = kmaxs[l]
    print('\nWorking with kmax = %.2f'%kmax)
    
    # find the fundamental frequency, the number of bins up to kmax and the k-array
    kF     = kmin
    k_bins = int((kmax-kmin)/kF)
    k      = np.arange(2,k_bins+2)*kF #avoid k=kF as we will get some negative values
    Nk     = 4.0*np.pi*k**2*kF/kF**3  #number of modes in each k-bin

    # find the number of neurons in the output layer and define loss
    if predict_C:  last_layer = 3
    else:          last_layer = 2
    
    # get the validation dataset and the loss function
    valid_data, valid_label = data.dataset(k, Nk, kpivot, batch_size_valid, predict_C)
    loss_func = nn.MSELoss()
    
    # do a loop over the different epochs
    loss_train = np.zeros(epochs, dtype=np.float64)
    loss_valid = np.zeros(epochs, dtype=np.float64)
    min_train  = 1e7

    # get the name of the model file
    fmodel = 'results/best_model_%s_kmax=%.2f.pt'%(suffix,kmax)

    # get the architecture and load (if exists) best-model
    net = architecture.Model(k.shape[0], hidden1, hidden2, hidden3, hidden4, last_layer)
    if os.path.exists(fmodel):
        net.load_state_dict(torch.load(fmodel))
        net.eval() 
        with torch.no_grad():
            pred = net(valid_data)
        min_valid = loss_func(pred, valid_label)
        print('min valid = %.3e'%min_valid)
    else:
        min_valid = 1e7
    
    optimizer = optim.Adam(net.parameters(), lr=max_lr, betas=(0.5, 0.999),
                           eps=1e-8, amsgrad=True)            
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.4, 
                                                           patience=500, verbose=True,
                                                           eps=min_lr)
    
    # do a loop over the different epochs
    for epoch in range(epochs): 

        # training
        train_data, train_label = data.dataset(k, Nk, kpivot, batch_size_train*batches, predict_C)
        total_loss = 0
        net.train()
        for batch in range(batches):
            data_train = train_data[batch*batch_size_train:(batch+1)*batch_size_train]
            label      = train_label[batch*batch_size_train:(batch+1)*batch_size_train]
            pred = net(data_train)
            loss = loss_func(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
        loss_train[epoch] = total_loss/batches
        
        # validation
        net.eval() 
        with torch.no_grad():
            pred = net(valid_data)
        loss_valid[epoch] = loss_func(pred, valid_label)
         
        # save model if it is better
        if loss_valid[epoch]<min_valid:
            print('saving model; kmax %.2f epoch %d; %.3e %.3e'\
                %(kmax,epoch,loss_train[epoch],loss_valid[epoch]))
            torch.save(net.state_dict(), fmodel)
            min_train, min_valid = loss_train[epoch], loss_valid[epoch]

        # plot the losses
        if plot_results:
            plt.cla() #clear axes
            plt.yscale('log')
            plt.plot(loss_train[:epoch])
            plt.plot(loss_valid[:epoch])
            plt.pause(0.0001)

        # update learning rate
        scheduler.step(loss_valid[epoch])

