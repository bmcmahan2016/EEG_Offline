import numpy as np
from load_data import DataManager
from nn_models import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import argparse
import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as utils

def UpdateClassificationTask(training_data, training_classes, include_classes):
    include_classes = set(include_classes)
    class_map = {}
    id_map = {}
    for idx, name in enumerate(include_classes):
        class_map[name] = idx
        id_map[idx] = name

    include_indices = []
    for idx in range(len(training_classes)):
        name = DataManager.CLASS_MAP[training_classes[idx]]
        if name in include_classes:
            training_classes[idx] = class_map[name]
            include_indices.append(idx)

    return training_data[include_indices], training_classes[include_indices], id_map

def GetClassFrequency(training_classes, class_map):
    uniques, counts = np.unique(training_classes, return_counts=True)
    total_trials = np.sum(counts)
    trial_freq = {}
    for idx in range(len(uniques)):
        trial_freq[class_map[uniques[idx]]] = (counts[idx], float(counts[idx]/total_trials))

    print("Relative trial frequency", trial_freq)
    print()
    return len(uniques)

def ClassifySVM(training_data, training_classes):
    X_train, X_test, y_train, y_test = train_test_split(training_data, training_classes, test_size=0.2, random_state=0) 
    # scaler = preprocessing.StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    clf = SVC(C=1.0, gamma='scale')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    train_acc = 100. * clf.score(X_train, y_train)
    print("train accuracy:", train_acc)
    test_acc = 100. * clf.score(X_test, y_test)
    print("test accuracy:", test_acc)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

def TrainNN(net, train_loader):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(20):
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            pred = net_out.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), 100. * correct / total, 
                            loss.item()))

def TestNN(net, test_loader):
    net.eval()
    criterion = nn.NLLLoss(reduction='sum')
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def ClassifyNN(training_data, training_classes, num_classes=4):
    X_train, X_test, y_train, y_test = train_test_split(training_data, training_classes, test_size=0.2, random_state=0) 
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train).long()
    train_dataset = utils.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = utils.DataLoader(train_dataset, batch_size=100)

    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.Tensor(y_test).long()
    test_dataset = utils.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = utils.DataLoader(test_dataset, batch_size=100)

    # net = Net(training_data.shape[1], num_classes)
    net = ConvNet(num_classes)
    TrainNN(net, train_loader)
    TestNN(net, test_loader)

def main():
    parser = argparse.ArgumentParser(description='Predict reach directions')
    parser.add_argument("-l", "--experiment_limit", type=int, required=True, help="number of db files to use")
    parser.add_argument("-m", "--classification_method", type=str, required=False, default="SVM",
        help="Type of classifier: SVM or NN (SVM is default)")
    args = parser.parse_args()

    data_manager = DataManager(bin_size=50, lowcut=30.0, highcut=60.0, include_center=False) # set parameters for data filtering and collection
    training_data, training_classes = data_manager.GetData(args.experiment_limit) # get data for specified number of experiments

    class_map = DataManager.CLASS_MAP
    include_classes = ['Left', 'Right'] # update classification task to include only these classes
    training_data, training_classes, class_map = UpdateClassificationTask(training_data, training_classes, include_classes)
    
    num_classes = GetClassFrequency(training_classes, class_map) # print the class frequency in the data set

    # each data sample has 2D array of bin_size x eeg_channels
    # we must flatten this into a 1D vector for SVM classification
    n,h,c = training_data.shape
    # training_data = training_data.reshape((n, -1))
    training_data = training_data.reshape((n, 1, h, c))
    if args.classification_method == "SVM":
        ClassifySVM(training_data, training_classes) # train and evaluate SVM on this data set
    elif args.classification_method == "NN":
        ClassifyNN(training_data, training_classes, num_classes=num_classes) # train and evaluate neural network on this data set

if __name__ == '__main__':
    main()