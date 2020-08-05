#################################
# Your name: Vicktoria Kraslavski
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# Generate points in 2D
# Return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # Plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    # Defying all classifiers
    linear_clf = svm.SVC(kernel='linear', C=1000)
    quadratic_clf = svm.SVC(kernel='poly', degree=2, C=1000)
    rbf_clf = svm.SVC(kernel='rbf', C=1000)

    # Training on train data
    linear_clf.fit(X_train, y_train)
    quadratic_clf.fit(X_train, y_train)
    rbf_clf.fit(X_train, y_train)

    # Getting SV
    numof_linear_sv = linear_clf.n_support_[0] + linear_clf.n_support_[1]  # TODO: Check if that is what i need
    numof_quadratic_sv = quadratic_clf.n_support_[0] + quadratic_clf.n_support_[1]
    numof_rbf_sv = rbf_clf.n_support_[0] + rbf_clf.n_support_[1]

    # Plotting classifiers
    create_plot(X_train, y_train, linear_clf)
    plt.title("Linear Kernel, %d support vectors" % numof_linear_sv)
    plt.show()
    create_plot(X_train, y_train, quadratic_clf)
    plt.title("Quadratic Kernel, %d support vectors" % numof_quadratic_sv)
    plt.show()
    create_plot(X_train, y_train, rbf_clf)
    plt.title("RBF Kernel, %d support vectors" % numof_rbf_sv)
    plt.show()

    return np.array([linear_clf.n_support_], [quadratic_clf.n_support_], [rbf_clf.n_support_])

def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    c_vector = np.array([10.0**p for p in np.arange(-5, 6)])
    accuracy_vector = []
    c_check_points = np.array([10**-4, 10**-2, 10**2])

    for c in c_vector:
        linear_clf = svm.SVC(kernel='linear', C=c)
        linear_clf.fit(X_train, y_train)
        mean_accuracy = linear_clf.score(X_val, y_val)
        accuracy_vector.append(mean_accuracy)
        if c in c_check_points:
            create_plot(X_val, y_val, linear_clf)
            plt.title("Linear Kernel, C = %f, " % c + "accuracy = %f" % mean_accuracy)
            plt.show()

    best_c = c_vector[np.argmax(accuracy_vector)]
    plt.plot(c_vector, accuracy_vector, label="Average accuracy", marker=".", color="blue")
    plt.xscale("log")
    plt.ylim((-0.1, 1.1))
    plt.xlabel('C')
    plt.axvline(best_c, color='yellow', label="best C: {}".format(best_c))
    plt.ylabel('Average accuracy')
    plt.legend()
    plt.show()

    return accuracy_vector

def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    gamma_vector = np.array([10.0**p for p in np.arange(-5, 6)])
    C = 10
    accuracy_vector = []
    gamma_check_points = np.array([10 ** -1, 10 ** 0, 10, 10 ** 2])

    for gamma in gamma_vector:
        rbf_clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
        rbf_clf.fit(X_train, y_train)
        mean_accuracy = rbf_clf.score(X_val, y_val)
        accuracy_vector.append(mean_accuracy)
        if gamma in gamma_check_points:
            create_plot(X_val, y_val, rbf_clf)
            plt.title("Linear Kernel, gamma = %f, " % gamma + "accuracy = %f" % mean_accuracy)
            plt.show()

    best_gamma = gamma_vector[np.argmax(accuracy_vector)]
    print(accuracy_vector)
    plt.plot(gamma_vector, accuracy_vector, label="Average accuracy", marker=".", color="blue")
    plt.xscale("log")
    plt.ylim((-0.1, 1.1))
    plt.xlabel('gamma')
    plt.axvline(best_gamma, color='yellow', label="best gamma: {}".format(best_gamma))
    plt.ylabel('Average accuracy')
    plt.legend()
    plt.show()

X_train, y_train, X_val, y_val = get_points()
#train_three_kernels(X_train, y_train, X_val, y_val)
#linear_accuracy_per_C(X_train, y_train, X_val, y_val)
rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val)