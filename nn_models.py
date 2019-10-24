import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_size, 320)
        self.fc2 = nn.Linear(320, 640)
        self.fc3 = nn.Linear(640, 120)
        self.fc4 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.c1 = nn.Conv2d(1, 10, (4, 1))
        self.bn1 = nn.BatchNorm2d(10)
        self.c2 = nn.Conv2d(10, 10, (1, 16))
        self.c3 = nn.Conv2d(10, 20, (6, 1))
        self.c4 = nn.Conv2d(20, 40, (6, 1))
        self.c5 = nn.Conv2d(40, 40, (6, 1))
        self.m5 = nn.MaxPool2d((2, 1))
        self.lf = nn.Linear(640, num_classes)
        self.drop_out = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.c1(x)
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = F.relu(self.c5(x))
        x = self.m5(x)
        x = x.view(-1, 640)
        x = self.drop_out(x)
        x = self.lf(x)
        return F.log_softmax(x, dim=1)

