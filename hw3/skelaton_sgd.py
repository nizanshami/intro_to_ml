#################################
# Your name: nizan shami
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper_hinge():
    mnist = fetch_openml('mnist_784', as_frame=False)
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
    mnist = fetch_openml('mnist_784',as_frame=False)
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
    w = np.zeros(len(data[0]))
    for t in range(1, T+1):
        i = np.random.randint(0, len(data))
        y = (np.dot(w, data[i])) * labels[i]
        eta = eta_0/t
        if(y - 1 < 1e-10):
            w = (1 - eta) * w + eta * C * labels[i] * data[i]
        else:
            w = (1 - eta) * w
    return w


def SGD_ce(data, labels, eta_0, T):
    wights = np.zeros((10, len(data[0])))
    labels = labels.astype(int)
    for t in range(1, T+1):
        i = np.random.randint(0, len(data))
        eta_mul_gradient = eta_0 * gradient(wights, data[i], labels[i])
        for i in range(10):
            wights[i] = wights[i] - eta_mul_gradient[i]
    return wights
       
        


#################################

def one_a(train_data, train_labels, validation_data, validation_labels):
    eta_range = np.linspace(0.1, 10, 10)
    average_accuracy = []
    for eta in eta_range:
        accuracy = []
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, 1, eta, 1000)
            accuracy.append(cal_accuracy( lambda x : np.sign(np.dot(w, x)) , validation_data, validation_labels))
        average_accuracy.append(np.average(accuracy))
        

    plt.plot(eta_range, average_accuracy ,marker = 'o', color = 'blue')
    plt.xlabel('eta')
    plt.ylabel('average accuracy')
    plt.xscale('log')
    plt.show()

def one_b(train_data, train_labels, validation_data, validation_labels):
    C_range = np.concatenate([np.linspace(0.00001, 0.0001, 5) ,np.linspace(0.0001, 0.001, 5), np.linspace(0.001, 0.1,5)]) 
    average_accuracy = []
    count = 0
    for C in C_range:
        accuracy = []
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, C, 0.85, 1000)
            accuracy.append(cal_accuracy( lambda x : np.sign(np.dot(w, x)) , validation_data, validation_labels))
        average_accuracy.append(np.average(accuracy))
        

    plt.plot(C_range, average_accuracy ,marker = 'o', color = 'blue')
    plt.xlabel('eta')
    plt.ylabel('average accuracy')
    plt.xscale('log')
    plt.show()

def one_c(train_data, train_labels):
    c = 0.00011 
    eta = 0.85
    w = SGD_hinge(train_data, train_labels, c, eta, 20000)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()

def cal_accuracy(classifier , validation_data, validation_lebal):
    count = 0
    for i,x in enumerate(validation_data):
        y_hat = classifier(x)
        if(y_hat != validation_lebal[i]):
            count += 1
    return 1 - count/len(validation_data)
                
def cal_softmax(wights, x):
    softmax = [np.dot(x, w) for w in wights]
    max_value = np.amax(softmax)
    softmax = softmax - max_value #log-sum-exp trick
    softmax = np.exp(softmax)
    return softmax/np.sum(softmax)

def gradient(wights, x, y):
    softmaxes = cal_softmax(wights, x)
    softmaxes[y] = softmaxes[y] - 1
    grad = [np.dot(softmax,x) for softmax in softmaxes]
    return np.array(grad)

def two_a(train_data, train_labels, validation_data, validation_labels):
    eta_range = np.concatenate([np.linspace(1e-8, 1e-7, 5), np.linspace(1e-7, 1e-6, 10),np.linspace(1e-6, 1e-5, 5)])
    average_accuracy = []
    count = 0
    for eta in eta_range:
        accuracy = []
        for i in range(10):
            wights = SGD_ce(train_data, train_labels, eta, 1000)
            accuracy.append(cal_accuracy( lambda x : np.argmax([np.dot(w, x) for w in wights]) , validation_data, validation_labels.astype(int)))
        average_accuracy.append(np.average(accuracy))
        count += 1
    
    plt.plot(eta_range, average_accuracy ,marker = 'o', color = 'blue')
    plt.xlabel('eta')
    plt.ylabel('average accuracy')
    plt.xscale('log')
    plt.show()

def two_b(train_data, train_labels):
    eta = 5e-7
    wights = SGD_ce(train_data, train_labels, eta, 20000)
    fig = plt.figure()
    for i in range(1,11):
        fig.add_subplot(2,5, i)
        plt.imshow(np.reshape(wights[i-1], (28, 28)), interpolation='nearest')
    plt.show()

def two_c(train_data, train_labels, test_data, test_labels):
    eta = 5e-7
    wights = SGD_ce(train_data, train_labels, eta, 20000)
    return cal_accuracy( lambda x : np.argmax([np.dot(w, x) for w in wights]) , test_data, test_labels.astype(int))

#################################
