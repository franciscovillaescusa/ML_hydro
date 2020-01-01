import numpy as np
import torch
import sys,os


# create the data to train/validate/test the network
def dataset(k, Nk, kpivot, batch_size, predict_C):

    # A goes from  0.1 to 10
    # B goes from -1.0 to 0.0
    # C goes from -0.5 to 0.5
    
    # get the values of the parameters/labels
    A  = (9.9*np.random.random(batch_size)+0.1)
    A2 = (A-0.1)/9.9
    B  = -1.0 + 1.0*np.random.random(batch_size)
    C  = -0.5 + np.random.random(batch_size)
    if predict_C:  label = np.array([A2, B, C], dtype=np.float32)
    else:          label = np.array([A2, B],    dtype=np.float32)

    # compute Pk
    Pk = np.zeros((batch_size, k.shape[0]), dtype=np.float32)
    for i in range(batch_size):
        Pk[i] = A[i]*k**B[i]

    # get the hydro Pk part
    indexes = np.where(k>kpivot)[0]
    if len(indexes)>0:
        A_value = Pk[:,indexes[0]]/k[indexes[0]]**C
        for i in range(batch_size):
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
    data = np.log10(Pk, dtype=np.float32) #Pk
    return torch.tensor(data), torch.tensor(label).T
