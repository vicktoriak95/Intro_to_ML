#################################
# Your name: Vicktoria Kraslavski
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper_hinge():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def helper_ce():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = labels[train_idx[:6000]]

    validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
    validation_labels = labels[train_idx[6000:8000]]

    test_data_unscaled = data[8000 + test_idx, :].astype(float)
    test_labels = labels[8000 + test_idx]

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    """
    # Initiating w
    w = np.zeros((len(data[0]),))
    for t in range(T):
        i = int(numpy.random.uniform(1, len(labels)))
        eta = eta_0 / (t + 1)
        value = labels[i] * w @ data[i]
        if value < 1:
            w = (1 - eta) * w + eta * C * labels[i] * data[i]
        else:
            w = (1 - eta) * w

    return w


def SGD_ce(data, labels, eta_0, T):
    """
    Implements multi-class cross entropy loss using SGD.
    """
    W = np.zeros((10, len(data[0])))  # W is the matrix of Weights, rows are w0,...,w9
    for t in range(T):
        i = np.random.randint(0, len(data))
        x = data[i]
        y = int(labels[i])
        eta = eta_0
        for index in range(10):
            if index == y:
                gradient = (soft_max(index, x, W) - 1) * x
            else:
                gradient = soft_max(index, x, W) * x
            W[index] = W[index] - eta * gradient

    return W

#################################


def q_1a():
    train_data, train_labels, validation_data, validation_labels, _, _ = helper_hinge()
    #print("got data")
    T = 1000
    C = 1
    eta_vector = np.array([10.0**p for p in numpy.arange(-5, 5, 0.2)])
    accur_vector = []
    best_eta = -1
    best_accur = -1

    for eta in eta_vector:
        #print("eta is", eta)
        w = SGD_hinge(train_data, train_labels, C, eta, T)
        #print("w is", w)
        avg_accur = 0
        for i in range(10):
            avg_accur += find_accur(validation_data, validation_labels, w)
        avg_accur /= 10
        #print("avg_accur is ", avg_accur)
        accur_vector.append(avg_accur)
        if(avg_accur > best_accur):
            best_accur = avg_accur
            best_eta = eta

    plt.plot(eta_vector, accur_vector, label="Average accuracy", marker=".", color="blue")
    plt.xscale("log")
    plt.ylim((-0.1, 1.1))
    plt.xlabel('eta')
    plt.axvline(best_eta, color='yellow', label="best_eta: {}".format(best_eta))
    plt.ylabel('Average accuracy')
    plt.legend()
    #plt.show()

    #print("best eta is", best_eta)
    #return best_eta


def q_1b():
    train_data, train_labels, validation_data, validation_labels, _, _ = helper_hinge()
    #print("got data")
    T = 1000
    eta_0 = 1
    c_vector = np.array([10.0**p for p in numpy.arange(-5, 6, 0.5)])
    accur_vector = []
    best_c = -1
    best_accur = -1

    for c in c_vector:
        #print("c is", c)
        w = SGD_hinge(train_data, train_labels, c, eta_0, T)
        #print("w is", w)
        avg_accur = 0
        for i in range(10):
            avg_accur += find_accur(validation_data, validation_labels, w)
        avg_accur /= 10
        #print("avg_accur is ", avg_accur)
        accur_vector.append(avg_accur)
        if(avg_accur > best_accur):
            best_accur = avg_accur
            best_c = c

    plt.plot(c_vector, accur_vector, label="Average accuracy", marker=".", color="blue")
    plt.xscale("log")
    plt.xlabel('C')
    plt.ylim((0.8, 1.1))
    plt.ylabel('Average accuracy')
    plt.axvline(best_c, color='yellow', label="best_C: {}".format(best_c))
    plt.legend()
    #plt.show()

    #print("best c is", best_c)
    #return best_c


def q_1cd():
    train_data, train_labels, _, _, test_data, test_labels = helper_hinge()
    C = 0.0001
    eta_0 = 1.2589
    T = 20000
    w = SGD_hinge(train_data, train_labels, C, eta_0, T)
    accur = find_accur(test_data, test_labels, w)
    #print("accuracy is ", accur)

    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    #plt.show()


def find_accur(data, labels, w):
    accur = 0

    for index, x in enumerate(data):
        pred_y = predict(w, x)
        true_y = labels[index]
        if pred_y == true_y:
            accur += 1

    return accur / len(labels)

def predict(w, x):
    pred_label = np.sign(w @ x)
    if pred_label == 0:
        pred_label = 1
    return pred_label


def predict_multi_class(W, x):
    optional_y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_score = []
    for y in optional_y:
        score = soft_max(y, x, W)
        y_score.append(score)
    y_pred = np.argmax(y_score)
    return y_pred


def find_accur_multi_class(data, labels, W):
    accur = 0

    for index, x in enumerate(data):
        pred_y = predict_multi_class(W, x)
        true_y = int(labels[index])
        if pred_y == true_y:
            accur += 1

    return accur / float(len(labels))


def soft_max(y, x, W):
    mult = W @ x
    #mult = mult - np.max(mult)  # Normalized
    mult = np.exp(mult)
    numerator = mult[y]  # Mone
    denominator = np.sum(mult)  # Mechane
    return numerator / denominator

def q_2a():
    train_data, train_labels, validation_data, validation_labels, _, _ = helper_ce()
    #print("got data")
    T = 1000
    eta_vector = np.array([10.0 ** p for p in np.arange(-9, 5.5, 0.5)])
    accur_vector = []
    best_eta = -1
    best_accur = -1

    for eta in eta_vector:
        print("eta is", eta)
        avg_accur = 0
        for i in range(10):
            W = SGD_ce(train_data, train_labels, eta, T)
            avg_accur += find_accur_multi_class(validation_data, validation_labels, W)
        avg_accur /= 10
        print("avg_accur is ", avg_accur)
        accur_vector.append(avg_accur)
        if (avg_accur > best_accur):
            best_accur = avg_accur
            best_eta = eta

    plt.plot(eta_vector, accur_vector, label="Average accuracy", marker=".", color="blue")
    plt.xscale("log")
    plt.ylim((-0.1, 1.1))
    plt.xlabel('eta')
    plt.ylabel('Average accuracy')
    plt.axvline(best_eta, color='yellow', label="best_eta: {}".format(best_eta))
    plt.legend()
    plt.show()

    #print("best eta is", best_eta)
    #return best_eta

def q2_bc():
    train_data, train_labels, _, _, test_data, test_labels = helper_ce()
    eta_0 = 3.162277660168379e-07
    T = 20000
    W = SGD_ce(train_data, train_labels, eta_0, T)
    accur = find_accur_multi_class(test_data, test_labels, W)
    print("accuracy is ", accur)
    fig = plt.figure()

    for index, w in enumerate(W):
        fig.add_subplot(2, 5, index + 1)
        plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()

#################################


#q_1a()
#q_1b()
#q_1cd()
#q_2a()
q2_bc()