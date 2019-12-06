import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_size, 320)
        self.fc2 = nn.Linear(320, 640)
        self.fc3 = nn.Linear(640, 120)
        self.fc4 = nn.Linear(120, num_classes)
        self.lr = 0.01
        self.momentum = 0.9
        self.name = "DenseNet"

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class ConvNet(nn.Module):
    def __init__(self, num_classes, collection_type, combine_features):
        super(ConvNet, self).__init__()
        if combine_features:
            in_channel = 3
        else:
            in_channel = 1
        self.c1 = nn.Conv2d(in_channel, 10, (4, 1))
        self.c2 = nn.Conv2d(10, 10, (1, 16))
        self.c3 = nn.Conv2d(10, 20, (6, 1))
        self.c4 = nn.Conv2d(20, 40, (6, 1))
        self.c5 = nn.Conv2d(40, 40, (6, 1))
        self.m5 = nn.MaxPool2d((2, 1))
        self.lf = nn.Linear(640, num_classes)
        self.drop_out = nn.Dropout(p=0.5)
        self.momentum = 0.9
        if collection_type == "gel":
            self.lr = 0.0001
        else:
            self.lr = 0.01
        self.name = "ConvNet"
    
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

class ConvNet2(nn.Module):
    def __init__(self, num_classes, collection_type, combine_features):
        super(ConvNet2, self).__init__()
        if combine_features:
            in_channel = 3
        else:
            in_channel = 1
        self.c1 = nn.Conv2d(in_channel, 8, (31, 1), padding=(15, 0))
        self.bn1 = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 16, (1, 16), groups=8)
        self.bn2 = nn.BatchNorm2d(16)
        self.a3 = nn.AvgPool2d((2, 1))
        self.d3 = nn.Dropout(p=0.5)
        self.c4 = nn.Conv2d(16, 16, (15, 1), groups=16, padding=(7, 0))
        self.c5 = nn.Conv2d(16, 16, 1)
        self.bn5 = nn.BatchNorm2d(16)
        self.a5 = nn.AvgPool2d((2, 1))
        self.d5 = nn.Dropout(p=0.5)
        self.lf = nn.Linear(320, num_classes)
        self.momentum = 0.9
        if collection_type == "gel":
            self.lr = 0.01
        else:
            self.lr = 0.01
        self.name = "ConvNet2"
    
    def forward(self, x):
        x = self.c1(x)
        x = self.bn1(x)
        x = self.c2(x)
        x = F.elu(self.bn2(x))
        x = self.a3(x)
        x = self.d3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = F.elu(self.bn5(x))
        x = self.a5(x)
        x = self.d5(x)
        x = x.view(-1, 320)
        x = self.lf(x)
        return F.log_softmax(x, dim=1)

