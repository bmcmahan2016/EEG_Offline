import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, bin_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(bin_size * 16, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

net = Net(10)
print(net)

