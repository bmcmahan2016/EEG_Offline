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
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as utils
import matplotlib.pyplot as plt
from datetime import datetime
import os

def GetClassFrequency(training_classes, class_map):
    uniques, counts = np.unique(training_classes, return_counts=True)
    total_trials = np.sum(counts)
    trial_freq = {}
    for idx in range(len(uniques)):
        trial_freq[class_map[uniques[idx]]] = (counts[idx], float(counts[idx]/total_trials))

    print("Relative trial frequency", trial_freq)
    print()
    return len(uniques)

def ClassifySVM(X_train, X_test, y_train, y_test):
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

def TrainNN(net, train_loader, optimizer, epoch):
    criterion = nn.NLLLoss()
    correct = 0
    train_loss = 0
    net.train()
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
    acc = 100. * correct / len(train_loader.dataset)
    print('Train Epoch: {} - Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, train_loss, correct, len(train_loader.dataset), acc))
    return train_loss, acc

def TestNN(net, test_loader, last_run=False):
    net.eval()
    criterion = nn.NLLLoss(reduction='sum')
    test_loss = 0
    correct = 0
    acc_per_class = {}
    with torch.no_grad():
        for data, target in test_loader:
            output = net(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if last_run:
                for idx, item in enumerate(target):
                    item = item.item()
                    if item not in acc_per_class:
                        acc_per_class[item] = [0, 0]
                    if pred[idx] == item:
                        acc_per_class[item][0] += 1
                    acc_per_class[item][1] += 1


    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    if last_run:
        for key in acc_per_class:
            correct = acc_per_class[key][0]
            total = acc_per_class[key][1]
            print("Target {} has accuracy: {}/{} ({:.2f}%)".format(key, correct, total, 100*float(correct)/total))
    return test_loss, acc

def PlotNNResults(epochs, loss, accuracy, plot_type):
    fig, axs = plt.subplots(2, sharex=True)
    axs[0].plot(epochs, accuracy)
    axs[0].set_ylabel(plot_type + " Accuracy")
    axs[1].plot(epochs, loss)
    axs[1].set_ylabel(plot_type + " Loss")

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("Epoch")
    fig.suptitle(plot_type + " Results")

def ClassifyNN(X_train, X_test, y_train, y_test, net, num_epochs, model_name, load_model):
    base_epoch = 1
    if load_model != "":
        state = torch.load("saved_models/" + load_model)
        net.load_state_dict(state['model'])
        base_epoch += state['epoch']

    epochs = range(base_epoch, base_epoch + num_epochs)
    plot_results = True
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train).long()
    train_dataset = utils.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = utils.DataLoader(train_dataset, batch_size=40)

    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.Tensor(y_test).long()
    test_dataset = utils.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = utils.DataLoader(test_dataset, batch_size=100)

    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    optimizer = optim.SGD(net.parameters(), lr=net.lr, momentum=net.momentum)
    if load_model != "":
        optimizer.load_state_dict(state['optimizer'])
    # scheduler = StepLR(optimizer, 10, gamma=0.9)
    min_loss = 1
    saved_model = {}
    saved_opt = {}
    model_epoch = 1
    for epoch in epochs:
        train_l, train_a = TrainNN(net, train_loader, optimizer, epoch)
        last_run = True
        if epoch == epochs[-1]:
            last_run = True
        test_l, test_a = TestNN(net, test_loader, last_run)
        if test_l < min_loss:
            min_loss = test_l
            saved_model = net.state_dict()
            saved_opt = optimizer.state_dict()
            model_epoch = epoch
        # scheduler.step()
        train_loss.append(train_l)
        train_acc.append(train_a)
        test_loss.append(test_l)
        test_acc.append(test_a)

    if plot_results:
        PlotNNResults(epochs, train_loss, train_acc, "Train")
        PlotNNResults(epochs, test_loss, test_acc, "Test")
        plt.show()
    
    if model_name != "":
        now = datetime.now()
        timestamp = now.strftime("%d_%H_%M")
        filename = os.path.join("saved_models", model_name + "_" + net.name + "_" + timestamp + ".pt")
        torch.save({
            'model' : saved_model,
            'epoch' : model_epoch,
            'optimizer' : saved_opt,
            'loss' : min_loss,
            }, filename)
        print("Saved model from epoch {} with test loss {}".format(model_epoch, min_loss))

def GetArgs():
    parser = argparse.ArgumentParser(description='Predict reach directions')
    parser.add_argument("-e", "--experiment_limit", type=int, required=False, default=5,
        help="Number of db files to use")
    parser.add_argument("-c", "--collection_type", type=str, required=False, default="gel",
        help="Type of data collection to use: gel or non_gel (gel is default)")
    parser.add_argument("-b", "--bin_size", type=int, required=False, default=80,
        help="Number of time ticks per data sample (80 is default)")
    parser.add_argument("-l", "--lowcut", type=float, required=False, default=30.0,
        help="Lowcut frequency for bandpass filter (30 is default)")
    parser.add_argument("-t", "--highcut", type=float, required=False, default=60.0,
        help="Highcut frequency for bandpass filter (60 is default)")
    parser.add_argument("-i", "--include_center", action="store_true",
        help="Include reaches back to the center in the data set")
    parser.add_argument("-d", "--reach_directions", type=str, nargs='+', required=False, 
        default=["Down", "Left", "Up", "Right"],
        help="Any combination of reach directions to include in data set: Down Left Up Right")
    parser.add_argument("-m", "--model", type=str, required=False, default="Conv",
        help="Type of classifier model: SVM, Dense, or Conv (Conv is default)")
    parser.add_argument("-p", "--equalize_proportions", action="store_false",
        help="Don't equalize the proportion of target reach classes")
    parser.add_argument("-s", "--sliding_window", action="store_true",
        help="Apply a sliding window to the data set")
    parser.add_argument("-f", "--combine_features", action="store_true",
        help="Create 3 channels of data for 5-12hz, 12-30hz, and 30-60hz")
    parser.add_argument("-r", "--num_epochs", type=int, required=False, default=60,
        help="Number of epochs for training neural nets (60 is default)")
    parser.add_argument("-n", "--model_name", type=str, required=False, default="",
        help="Non-empty name will save trained saved model with specified name")
    parser.add_argument("-z", "--load_model", type=str, required=False, default="",
        help="Non-empty name will continue training the specified model")
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
    data_manager = DataManager(collection_type=args.collection_type, bin_size=args.bin_size, trial_delay=15,
        lowcut=args.lowcut, highcut=args.highcut, include_center=args.include_center,
        sliding_window=args.sliding_window, equalize_proportions=args.equalize_proportions,
        include_classes=args.reach_directions, combine_features=args.combine_features,
        start_exp=1, inc=40, augment_training=False) # set parameters for data filtering and collection
    X_train, X_test, y_train, y_test, class_map = data_manager.GetData(args.experiment_limit, plot_freq=False) # get data for specified number of experiments
    print("Class map:", class_map)
    print(X_train.shape)

    # VisualizeData(X_train, y_test)
    # exit(1)

    num_classes = GetClassFrequency(y_train, class_map) # print the class frequency in the data set

    # each data sample has 2D array of bin_size x eeg_channels
    # we must flatten this into a 1D vector for SVM classification
    if args.model == "SVM":
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))
        ClassifySVM(X_train, X_test, y_train, y_test) # train and evaluate SVM on this data set
    else:
        if args.model == "Dense":
            X_train = X_train.reshape((X_train.shape[0], -1))
            X_test = X_test.reshape((X_test.shape[0], -1))
            net = Net(X_train.shape[1], num_classes)
        elif args.model == "Conv": # currently only works for bin size = 80
            if not args.combine_features:
                X_train = np.expand_dims(X_train, axis=1)
                X_test = np.expand_dims(X_test, axis=1)
            if args.collection_type != 'ant':
                net = ConvNet2(num_classes, args.collection_type, args.combine_features)
            else:
                net = ConvNetANT(num_classes)
        ClassifyNN(X_train, X_test, y_train, y_test, net, args.num_epochs, args.model_name, args.load_model) # train and evaluate neural network on this data set

if __name__ == '__main__':
    main()