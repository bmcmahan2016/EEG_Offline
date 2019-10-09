import numpy as np
from load_data import GetData
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from load_data import CLASS_MAP
from sklearn import preprocessing

def main():
    db_file_limit = 5
    training_data, training_classes = GetData(db_file_limit, bin_size=10)
    n,b,c = training_data.shape
    training_data = training_data.reshape((n, b*c))
    print("Training data shape", training_data.shape)
    print("Training class shape", training_classes.shape)

    uniques, counts = np.unique(training_classes, return_counts=True)
    total_trials = np.sum(counts)
    trial_freq = {CLASS_MAP[uniques[0]] : (counts[0], float(counts[0])/total_trials),
                  CLASS_MAP[uniques[1]] : (counts[1], float(counts[1])/total_trials),
                  CLASS_MAP[uniques[2]] : (counts[2], float(counts[2])/total_trials),
                  CLASS_MAP[uniques[3]] : (counts[3], float(counts[3])/total_trials)}
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