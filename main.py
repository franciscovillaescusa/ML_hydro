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
kmaxs = [0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #h/Mpc

# model parameters
kpivot       = 0.2
predict_C    = False
fix_A_value  = False #whether fix A_value for kpivot or not
suffix       = 'Pk-30-30-30-2_BS=128_batches=64_noSche_Adam_lr=1e-3'
model_folder = 'Pk-30-30-30-2_kpivot=%.2f'%kpivot 

# architecture parameters
model   = 'model1' #'model1', 'model0'
hidden1 = 100 #for model0
hidden2 = 100 #for model0
hidden3 = 100 #for model0
hidden4 = 100 #for model0
hidden5 = 100 #for model0
hidden  = 30  #for model1

# training parameters
epochs           = 500000
batch_size_train = 256
batch_size_valid = 64*3000
batches          = 32 #64
max_lr           = 1e-4
min_lr           = 1e-7  #only for scheduler
wd               = 0.0

plot_results = False
#######################################################################################

# find suffix
suffix = '%s_kpivot=%.2f'%(suffix,kpivot)
if not(fix_A_value):  suffix = '%s_varied_A'%suffix

# find the numbers that each cpu will work with
numbers = np.where(np.arange(len(kmaxs))%nprocs==myrank)[0]

# find the number of neurons in the output layer and define loss
if predict_C:  last_layer = 3
else:          last_layer = 2

# do a loop over the different kmax
for l in numbers:
    
    kmax = kmaxs[l]
    
    # find the fundamental frequency, the number of bins up to kmax and the k-array
    kF     = kmin
    k_bins = int((kmax-kmin)/kF)
    k      = np.arange(2,k_bins+2)*kF #avoid k=kF as we will get some negative values
    Nk     = 4.0*np.pi*k**2*kF/kF**3  #number of modes in each k-bin
    
    # get the validation dataset and the loss function
    valid_data, valid_label = \
                data.dataset(k, Nk, kpivot, batch_size_valid, predict_C, fix_A_value)
    loss_func = nn.MSELoss()
    #loss_func = nn.L1Loss()

    # get the architecture
    if   model=='model0':
        net = architecture.Model(k.shape[0], hidden1, hidden2, hidden3, hidden4, 
                                 hidden5, last_layer)
    elif model=='model1':
        net = architecture.Model1(k.shape[0], hidden, last_layer)
    else:  raise Exception('Wrong model!')

    # load best model if it exists
    fmodel = 'results/models/%s/best_model_%s_kmax=%.2f.pt'%(model_folder,suffix,kmax)
    if os.path.exists(fmodel):
        net.load_state_dict(torch.load(fmodel))
        net.eval() 
        with torch.no_grad():
            pred = net(valid_data)
        min_valid = loss_func(pred, valid_label)
        print('kmax=%.2f ---> min valid = %.3e'%(kmax,min_valid))
    else:
        min_valid = 1e7
    
    # define the optimizer
    optimizer = optim.Adam(net.parameters(), lr=max_lr, betas=(0.9, 0.999),
                           eps=1e-8, amsgrad=False, weight_decay=wd)            
    #optimizer = optim.SGD(net.parameters(), lr=max_lr, momentum=0.99, nesterov=True)

    # define the scheduler
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.4, 
    #                                                       patience=100, verbose=True,
    #                                                       eps=min_lr)


    # find the name of the output file and read it
    fout = 'results/validation_losses/valid_losses_%s_kmax=%.2f.txt'%(suffix,kmax)
    if os.path.exists(fout):
        offset, dumb = np.loadtxt(fout, unpack=True)
        offset = int(offset[-1])
    else:
        offset = 0

    # do a loop over the different epochs
    for epoch in range(offset,offset+epochs): 

        # training
        train_data, train_label = \
            data.dataset(k, Nk, kpivot, batch_size_train*batches, predict_C, fix_A_value)
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
        loss_train = total_loss/batches
        
        # validation
        net.eval() 
        with torch.no_grad():
            pred = net(valid_data)
        loss_valid = loss_func(pred, valid_label)
         
        # save model if it is better
        if loss_valid<min_valid:
            print('saving model; kmax %.2f epoch %d; %.4e %.4e'\
                %(kmax,epoch,loss_train,loss_valid))
            torch.save(net.state_dict(), fmodel)
            min_valid = loss_valid

            f = open(fout, 'a')
            f.write('%d %.4e\n'%(epoch, loss_valid))
            f.close()

        """
        # plot the losses
        if plot_results:
            plt.cla() #clear axes
            plt.yscale('log')
            plt.plot(loss_train[:epoch])
            plt.plot(loss_valid[:epoch])
            plt.pause(0.0001)
        """

        # update learning rate
        #scheduler.step(loss_valid[epoch])

