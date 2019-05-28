import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys,os
import matplotlib.pyplot as plt



# define the arquitecture of the network
class Model(nn.Module):
    def __init__(self,k_bins,hidden1,hidden2,hidden3):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(k_bins,  hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 2)

    def forward(self,k):
        out = F.relu(self.fc1(k))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out
    
# create the data to train the network
def dataset(k, Nk, batch_size):
    data  = np.empty((batch_size, k.shape[0]), dtype=np.float32)
    label = np.empty((batch_size,2),           dtype=np.float32)
    i = 0
    while(i<batch_size):
        alpha  = (9.9*np.random.random()+0.1)
        beta   = -1.0 + 1.0*np.random.random()
        Pk     = alpha*k**beta
        dPk    = np.sqrt(Pk**2/Nk)
        Pk     = np.random.normal(loc=Pk, scale=dPk)
        if np.any(Pk<0):
            continue
        data[i] = np.log10(Pk)
        label[i] = [alpha, beta]
        i += 1
    return torch.tensor(data),torch.tensor(label)


####################################### INPUT ##########################################
kmin = 7e-3 #h/Mpc
kmax = 0.05  #h/Mpc

hidden1 = 30
hidden2 = 30
hidden3 = 30

epochs     = 500
batch_size = 32
batches    = 100
########################################################################################

# find the fundamental frequency and the number of bins up to kmax
kF   = kmin
k_bins = int((kmax-kmin)/kF)

# define the k-array
k = np.arange(1,k_bins+2)*kF

# find the number of modes in each k-bin: Nk = 2*k^2*dk/kF^3
Nk = 4.0*np.pi*k**2*kF/kF**3

# get a test dataset
test_data, test_label = dataset(k,Nk,batch_size*5)

# define the network, loss and optimizer
net = Model(k.shape[0],hidden1,hidden2,hidden3)
loss_func = nn.MSELoss()
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.95)
optimizer = optim.Adam(net.parameters(), lr=0.002)


# do a loop over the different epochs
loss_train = np.zeros(epochs, dtype=np.float32)
loss_test  = np.zeros(epochs, dtype=np.float32)
done1, done2, done3 = False, False, False
min_train, min_eval = 1e7, 1e7

for epoch in xrange(epochs): 

    total_loss = 0
    for batch in xrange(batches):
    
        data, label = dataset(k,Nk,batch_size)
        pred = net(data)
        loss = loss_func(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()
        
    loss_train[epoch] = total_loss/batches
        
    # compute the loss for the test set
    pred = net(test_data)
    loss_test[epoch] = loss_func(pred, test_label).detach()

    # save model if it is better
    if loss_train[epoch]<min_train and loss_test[epoch]<min_eval:
        print 'saving model; epoch %d; %.3e %.3e'%(epoch,loss_train[epoch],loss_test[epoch])
        torch.save(net.state_dict(), 'results/best_model_kmax=%.2f.pt'%kmax)
        min_train, min_eval = loss_train[epoch], loss_test[epoch]
    
    if loss.item()<1e-3 and not(done1):
        lr = 5e-4
        for g in optimizer.param_groups:
            g['lr'] = lr
        done1 = True
        print lr

    if loss.item()<1e-4 and not(done2):
        lr = 2e-4
        for g in optimizer.param_groups:
            g['lr'] = lr
        done2 = True
        print lr

    if loss.item()<1e-5 and not(done3):
        lr = 1e-4
        for g in optimizer.param_groups:
            g['lr'] = lr
        done3 = True
        print lr

    #plt.cla() #clear axes
    #plt.yscale('log')
    #plt.plot(loss_train[:epoch])
    #plt.plot(loss_test[:epoch])
    #plt.pause(0.0001)
    
np.savetxt('results/loss_kmax=%.2f.txt'%kmax, np.transpose([loss_train,loss_test]))



###### evaluate the performance of the model ######
test_data, test_label = dataset(k,Nk,batch_size=10000)

net = Model(k.shape[0],hidden1,hidden2,hidden3)
net.load_state_dict(torch.load('results/best_model_kmax=%.2f.pt'%kmax))
net.eval()

pred = net(test_data)
loss = loss_func(pred, test_label).detach()

print 'MSL for 10000 samples = %.3e'%loss
###################################################
    
