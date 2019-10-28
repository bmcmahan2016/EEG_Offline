import numpy as np
from load_data import DataManager
from load_data import PlotData
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

def UpdateClassificationTask(training_data, training_classes, include_classes, equalize_proportions=False):
    include_classes = set(include_classes)
    class_map = {}
    id_map = {}
    class_count = {}
    for idx, name in enumerate(include_classes):
        class_map[name] = idx
        id_map[idx] = name
        class_count[name] = 0

    include_indices = []
    for idx in range(len(training_classes)):
        name = DataManager.CLASS_MAP[training_classes[idx]]
        if name in include_classes:
            class_count[name] += 1
            if (len(include_classes) < 4):
                training_classes[idx] = class_map[name]
                include_indices.append(idx)

    if (len(include_classes) < 4): # update classification task to include only these classes
        training_data = training_data[include_indices]
        training_classes = training_classes[include_indices]
    else:
        id_map = DataManager.CLASS_MAP

    if equalize_proportions:
        include_indices = []
        nums = [val for _, val in class_count.items()]
        min_count = min(nums)
        for key in class_count:
            class_count[key] = 0
        for idx in range(len(training_classes)):
            if class_count[id_map[training_classes[idx]]] < min_count:
                class_count[id_map[training_classes[idx]]] += 1
                include_indices.append(idx)

        training_data = training_data[include_indices]
        training_classes = training_classes[include_indices]

    return training_data, training_classes, id_map

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
    print("Train accuracy:", train_acc)
    test_acc = 100. * clf.score(X_test, y_test)
    print("Test accuracy:", test_acc)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n",cm)

def TrainNN(net, train_loader):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=net.lr, momentum=net.momentum)
    for epoch in range(50):
        correct = 0
        train_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            train_loss += loss.item() * len(data)
            loss.backward()
            optimizer.step()
            pred = net_out.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss /= len(train_loader.dataset)
        print('Train Epoch: {} - Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            epoch, train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))

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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def ClassifyNN(training_data, training_classes, net):
    X_train, X_test, y_train, y_test = train_test_split(training_data, training_classes, test_size=0.2, random_state=0) 
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train).long()
    train_dataset = utils.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = utils.DataLoader(train_dataset, batch_size=100)

    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.Tensor(y_test).long()
    test_dataset = utils.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = utils.DataLoader(test_dataset, batch_size=100)

    TrainNN(net, train_loader)
    TestNN(net, test_loader)

def GetArgs():
    parser = argparse.ArgumentParser(description='Predict reach directions')
    parser.add_argument("-e", "--experiment_limit", type=int, required=False, default=5,
        help="Number of db files to use")
    parser.add_argument("-c", "--collection_type", type=str, required=False, default="non_gel",
        help="Type of data collection to use: gel or non_gel (non_gel is default)")
    parser.add_argument("-b", "--bin_size", type=int, required=False, default=10,
        help="Number of time ticks per data sample")
    parser.add_argument("-l", "--lowcut", type=float, required=False, default=30.0,
        help="Lowcut frequency for bandpass filter")
    parser.add_argument("-t", "--highcut", type=float, required=False, default=60.0,
        help="Highcut frequency for bandpass filter")
    parser.add_argument("-i", "--include_center", action="store_true",
        help="Include reaches back to the center in the data set")
    parser.add_argument("-d", "--reach_directions", type=str, nargs='+', required=False, 
        default=["Down", "Left", "Up", "Right"],
        help="Any combination of reach directions to include in data set: Down Left Up Right")
    parser.add_argument("-m", "--model", type=str, required=False, default="SVM",
        help="Type of classifier model: SVM, Dense, or Conv (SVM is default)")
    parser.add_argument("-p", "--equalize_proportions", action="store_true",
        help="Equalize the proportion of target reach classes")
    return parser.parse_args()

def VisualizeData(training_data, training_classes):
    plot_map = {0 : False, 1 : False, 2 : False, 3 : False}
    plot_data = []
    plot_labels = []
    for i in range(len(training_classes)):
        if plot_map[training_classes[i]]:
            continue
        plot_data.append(training_data[i])
        plot_labels.append(training_classes[i])
        plot_map[training_classes[i]] = True
        if len(plot_data) == 4:
            break

    PlotData(2, 2, plot_data, plot_labels)


def main():
    args = GetArgs()
    data_manager = DataManager(collection_type=args.collection_type, bin_size=args.bin_size,
        lowcut=args.lowcut, highcut=args.highcut, include_center=args.include_center) # set parameters for data filtering and collection
    training_data, training_classes = data_manager.GetData(args.experiment_limit, plot_freq=True) # get data for specified number of experiments

    training_data, training_classes, class_map = UpdateClassificationTask(training_data, training_classes,
        args.reach_directions, equalize_proportions=args.equalize_proportions)
    # VisualizeData(training_data, training_classes)
    # exit(1)
    num_classes = GetClassFrequency(training_classes, class_map) # print the class frequency in the data set

    # each data sample has 2D array of bin_size x eeg_channels
    # we must flatten this into a 1D vector for SVM classification
    n,h,c = training_data.shape
    if args.model == "SVM":
        training_data = training_data.reshape((n, -1))
        ClassifySVM(training_data, training_classes) # train and evaluate SVM on this data set
    else:
        if args.model == "Dense":
            training_data = training_data.reshape((n, -1))
            net = Net(training_data.shape[1], num_classes)
        elif args.model == "Conv": # currently only works for bin size = 50
            training_data = training_data.reshape((n, 1, h, c))
            net = ConvNet(num_classes, args.collection_type)
        ClassifyNN(training_data, training_classes, net) # train and evaluate neural network on this data set

if __name__ == '__main__':
    main()