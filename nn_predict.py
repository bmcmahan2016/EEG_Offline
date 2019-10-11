import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_size, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

