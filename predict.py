import numpy as np
from load_data import DataManager
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import argparse

def main():
    parser = argparse.ArgumentParser(description='Predict reach directions')
    parser.add_argument("-f", "--file_limit", type=int, help="number of db files to use")
    args = parser.parse_args()

    data_manager = DataManager(bin_size=10, lowcut=5.0, highcut=50.0)
    training_data, training_classes = data_manager.GetData(args.file_limit)
    n,b,c = training_data.shape
    training_data = training_data.reshape((n, b*c))
    print("Training data shape", training_data.shape)
    print("Training class shape", training_classes.shape)

    uniques, counts = np.unique(training_classes, return_counts=True)
    total_trials = np.sum(counts)
    trial_freq = {DataManager.CLASS_MAP[uniques[0]] : (counts[0], float(counts[0])/total_trials),
                  DataManager.CLASS_MAP[uniques[1]] : (counts[1], float(counts[1])/total_trials),
                  DataManager.CLASS_MAP[uniques[2]] : (counts[2], float(counts[2])/total_trials),
                  DataManager.CLASS_MAP[uniques[3]] : (counts[3], float(counts[3])/total_trials)}
    print("Relative trial frequency", trial_freq)
    print()

    X_train, X_test, y_train, y_test = train_test_split(training_data, training_classes, random_state = 0) 
    # scaler = preprocessing.StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    clf = SVC(C=1.0, gamma='scale')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    train_acc = clf.score(X_train, y_train)
    print("train accuracy:", train_acc)
    test_acc = clf.score(X_test, y_test)
    print("test accuracy:", test_acc)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)


if __name__ == '__main__':
    main()