import torch.nn as nn
import torch.nn.functional as F


# define the arquitecture of the network
class Model(nn.Module):
    def __init__(self,k_bins,hidden1,hidden2,hidden3,hidden4,hidden5,last_layer):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(k_bins,  hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, hidden4)
        #self.fc5 = nn.Linear(hidden4, hidden5)
        self.fc5 = nn.Linear(hidden4, last_layer)
        #self.fc6 = nn.Linear(hidden5, last_layer)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.bn4 = nn.BatchNorm1d(hidden4)
        self.dropout = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self,Pk):
        #out = self.dropout(F.leaky_relu(self.bn1(self.fc1(Pk))))
        #out = self.dropout(F.leaky_relu(self.bn2(self.fc2(out))))
        #out = self.dropout(F.leaky_relu(self.bn3(self.fc3(out))))
        #out = self.dropout(F.leaky_relu(self.bn4(self.fc4(out))))
        #out = self.fc5(out)

        out = F.leaky_relu(self.fc1(Pk))
        out = F.leaky_relu(self.fc2(out))
        out = F.leaky_relu(self.fc3(out))
        #out = F.leaky_relu(self.fc4(out))
        #out = F.leaky_relu(self.fc5(out))
        out = self.fc5(out)
        #out = self.fc6(out)

        """
        out = F.elu(self.fc1(Pk))
        out = F.elu(self.fc2(out))
        out = F.elu(self.fc3(out))
        #out = F.leaky_relu(self.fc4(out))
        #out = F.leaky_relu(self.fc5(out))
        out = self.fc5(out)
        #out = self.fc6(out)
        """

        return out



# define the arquitecture of the network
# This model seems, for hidden=20, seems to work pretty well
class Model1(nn.Module):
    def __init__(self,k_bins, hidden, last_layer):
        super(Model1,self).__init__()
        self.fc1 = nn.Linear(k_bins, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, last_layer)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self,Pk):
        out = F.leaky_relu(self.fc1(Pk))
        out = F.leaky_relu(self.fc2(out))
        out = F.leaky_relu(self.fc2(out))
        out = self.fc3(out)
        return out
