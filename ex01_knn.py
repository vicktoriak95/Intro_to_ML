import numpy.random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
data = mnist['data']
labels = mnist['target']

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]
print("done training ")


def find_majority(labels_vector):
    labels_dict = {}
    for label in labels_vector:  # Counting the labels
        if label in labels_dict:  # If label in labels_dict
            labels_dict[label] += 1
        else:
            labels_dict[label] = 1
    major_label, max_appearances = (None, 0)
    for label in labels_dict:  # Finding the majority label
        if labels_dict[label] > max_appearances:
            major_label, max_appearances = (label, labels_dict[label])
    return major_label


def k_nearest_neighbors(t_images, t_labels, query_image, k):
    distances = []
    for i in range(len(t_images)):  # Calculating distances
        distance = numpy.linalg.norm(t_images[i] - query_image)
        distances.append((distance, t_labels[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]  # K nearest neighbors
    labels_vector = [neighbor[1] for neighbor in neighbors]
    return find_majority(labels_vector)


def knn_accuracy_prediction(k, n):
    sum_success = 0
    for i in range(len(test)):
        created_label = k_nearest_neighbors(train[:n], train_labels[:n], test[i], k)
        real_label = test_labels[i]
        if created_label == real_label:
            sum_success += 1
    accuracy_rate = sum_success / len(test)
    print("Your accuracy rate is:", accuracy_rate * 100, "%")
    return accuracy_rate


def k_accuracy():
    results = []
    for k in range(1, 101):
        results.append(knn_accuracy_prediction(k, 1000))
    plt.plot([i for i in range(1, 101)], results)
    plt.show()


def n_accuracy():
    results = []
    for i in range(100, 5001, 100):
        results.append(knn_accuracy_prediction(1, i))
    plt.plot([i for i in range(1, 101)], results)
    plt.show()




