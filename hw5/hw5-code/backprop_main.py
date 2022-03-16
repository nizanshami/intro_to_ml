from random import randbytes
import matplotlib.pyplot as plt
import numpy as np
import backprop_data

import skeleton_backprop_network

def a():
    training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)
    net = skeleton_backprop_network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)



def b():
    training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)
    rates = [0.001, 0.01, 0.1, 1, 10, 100]
    train_accuracy = [0 for i in rates]
    train_loss = [0 for i in rates]
    test_accuracy = [0 for i in rates]

    for i in range(6):
        net = skeleton_backprop_network.Network([784, 40, 10])
        train_accuracy[i], train_loss[i], test_accuracy[i] = net.SGD(training_data, epochs=30, mini_batch_size=10,
                                                                     learning_rate=rates[i], test_data=test_data)

    for i in range(len(rates)):
        plt.plot(np.arange(30), train_accuracy[i], label="rate = {}".format(rates[i]))

    plt.xlabel('epochs')
    plt.ylabel('Train Accuracy')
    plt.legend()
    plt.show()

    for i in range(len(rates)):
        plt.plot(np.arange(30), train_loss[i], label="rate = {}".format(rates[i]))

    plt.xlabel('epochs')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.show()

    for i in range(len(rates)):
        plt.plot(np.arange(30), test_accuracy[i], label="rate = {}".format(rates[i]))

    plt.xlabel('epochs')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.show()

def c():
    training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
    net = skeleton_backprop_network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

c()