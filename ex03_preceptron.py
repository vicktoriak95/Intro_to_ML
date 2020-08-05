#################################
# Your name: Vicktoria Kraslavski
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing


"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""


def helper():
    # Fetching the database - data and labels
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    # Reading examples labeled 0,8
    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    # Dividing into train, validation and test
    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos)*2-1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos)*2-1

    test_data_unscaled = data[60000+test_idx, :].astype(float)
    test_labels = (labels[60000+test_idx] == pos)*2-1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def perceptron(data, labels):
    """
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """
    # Normalizing data and initiating w
    normalized_data = sklearn.preprocessing.normalize(data)
    w = np.zeros((data.shape[1],))

    for index, x in enumerate(normalized_data):
        # Predicting label of x
        pred_label = predict(w, x)

        # Comparing with true label and updating w if necessary
        true_label = labels[index]
        if pred_label != true_label:
            w = w + true_label * x

    return w


#################################

# Place for additional code

def q_a():
    # Defying test and train data, and initiating array of n's
    train_data, train_labels, _, _, test_data, test_labels = helper()
    n_array = np.array([5, 10, 50, 100, 500, 1000, 5000])

    accur_5_array = []
    accur_95_array = []
    mean_array = []

    for n in n_array:
        n_train_data = train_data[:n, :]
        n_train_labels = train_labels[:n]

        accur_array = []

        for i in range(100):
            # Shuffling data
            indices = np.arange(n_train_data.shape[0])
            np.random.shuffle(indices)
            n_train_data = n_train_data[indices]
            n_train_labels = n_train_labels[indices]

            # Training on train data and calculating w
            w = perceptron(n_train_data, n_train_labels)

            # Calculating accuracy
            accur_array.append(accuracy_calc(test_data, test_labels, w))

        # Calculating mean, 5% ans 95% percentiles
        accur_array = np.array(accur_array)
        accur_5_array.append(np.percentile(accur_array, 5))
        accur_95_array.append(np.percentile(accur_array, 95))
        mean_array.append(np.mean(accur_array))

    # Plotting table
    colLabels = n_array
    rowLabels = ['Mean accuracy', '5th percentile', '95th percentile']
    cellText = [mean_array, accur_5_array, accur_95_array]

    hcell, wcell = 0.1, 0.1
    hpad, wpad = 0.5, 0.5
    fig = plt.figure(dpi=80, figsize=(len(colLabels) * wcell + wpad, 3 * hcell + hpad))
    ax = fig.add_subplot(1, 1, 1)
    table = ax.table(cellText=cellText,
                     rowLabels=rowLabels,
                     colLabels=colLabels,
                     loc='center')
    ax.axis('off')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    #plt.show()


def q_b():
    train_data, train_labels, _, _, _, _ = helper()
    w = perceptron(train_data, train_labels)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    #plt.show()


def q_c():
    train_data, train_labels, _, _, test_data, test_labels = helper()
    w = perceptron(train_data, train_labels)
    accur = accuracy_calc(test_data, test_labels, w)
    #print(accur)


def q_d():
    train_data, train_labels, _, _, test_data, test_labels = helper()
    w = perceptron(train_data, train_labels)
    counter = 0
    eight = None
    zero = None
    for index, x in enumerate(test_data):
        predict_y = predict(w, x)
        true_y = test_labels[index]
        if true_y > predict_y:
            eight = x
            counter += 1
        if true_y < predict_y:
            zero = x
            counter += 1
        if counter == 2:
            break
    # Printing mistaken pictures
    """
    print("Show mistake on eight ")
    plt.imshow(np.reshape(eight, (28, 28)), interpolation='nearest')
    plt.show()
    print("Show mistake on zero ")
    plt.imshow(np.reshape(zero, (28, 28)), interpolation='nearest')
    plt.show()
    """

def predict(w, x):
    pred_label = np.sign(w @ x)
    if pred_label == 0:
        pred_label = 1
    return pred_label


def accuracy_calc(data, labels, w):
    error_count = 0
    for index, x in enumerate(data):
        pred_y = predict(w, x)
        true_y = labels[index]
        if true_y != pred_y:
            error_count += 1
    return 1 - (error_count / data.shape[0])

#q_a()
#q_b()
#q_c()
#q_d()

#################################



