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
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.bn4 = nn.BatchNorm1d(hidden4)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,Pk):
        out = self.dropout(F.leaky_relu(self.bn1(self.fc1(Pk))))
        out = self.dropout(F.leaky_relu(self.bn2(self.fc2(out))))
        out = self.dropout(F.leaky_relu(self.bn3(self.fc3(out))))
        out = self.dropout(F.leaky_relu(self.bn4(self.fc4(out))))
        out = self.fc5(out)
        return out
