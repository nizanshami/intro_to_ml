from numpy.linalg.linalg import norm
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

#(a)
def k_nn(training_images, lebals, query, k):
    distances = [(np.linalg.norm(image - query), lebals[i]) for i, image in enumerate(training_images)]
    k_nearest = sorted(distances, key=lambda item: item[0])[:k]
    
    return majority_lable(k_nearest)

def majority_lable(k_nearest):
    d = {}
    for neighbor in k_nearest:
        if d.get(neighbor[1]) == None:
            d[neighbor[1]] = 1
        else: 
            d[neighbor[1]] += 1
    return max(d, key=d.get)

def calculate_accuracy(n, train_data, train_labels, test_data, test_labels, k):
    successes = 0 
    for i, image in enumerate(test_data):
        if(test_labels[i] == k_nn(train_data[:n], train_labels[:n], image, k)):
            successes += 1
    accuracy = successes/len(test) * 100
    return accuracy

if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']
    idx = np.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]

    #(b)
    n = 1000
    k = 10
    print(calculate_accuracy(1000, train[:n], train_labels[:n], test, test_labels, k))
    

    #(c)
    y = []
    max_accuracy = 0
    for k in range(1,101):
        accuracy = calculate_accuracy(n, train[:n], train_labels[:n], test, test_labels, k)
        y.append(accuracy)
        if accuracy > y[max_accuracy]:
            max_accuracy = k

    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.plot(range(1,101), y)
    plt.show()

    
    #(d)
    y = []
    for n in range(100,5001,100):
        y.append(calculate_accuracy(n, train[:n], train_labels[:n], test, test_labels, 1))

    plt.xlabel("n")
    plt.ylabel("accuracy")
    plt.plot(range(100,5001,100), y)
    plt.show()