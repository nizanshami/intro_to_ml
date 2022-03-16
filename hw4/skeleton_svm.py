#################################
# Your name: nizan shami
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    linear_clf = svm.SVC(C=1000, kernel='linear')
    quadratic_clf = svm.SVC(C=1000, kernel='poly', degree=2)
    rbf_clf = svm.SVC(C=1000, kernel='rbf')

    linear_clf.fit(X_train, y_train)
    quadratic_clf.fit(X_train, y_train)
    rbf_clf.fit(X_train, y_train)

    create_plot(X_train, y_train, linear_clf)
    plt.show()
    create_plot(X_train, y_train, quadratic_clf)
    plt.show()
    create_plot(X_train, y_train, rbf_clf)
    plt.show()
    

    return np.array([linear_clf.n_support_, quadratic_clf.n_support_, rbf_clf.n_support_]) 
    


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    
    c_range = np.logspace(-5,5, num=30)
    accuracy_train = []
    accuracy_val= []
    for c in c_range:
        linear_clf = svm.SVC(C=c, kernel='linear')
        linear_clf.fit(X_train, y_train)
        accuracy_train.append(linear_clf.score(X_train, y_train))
        accuracy_val.append(linear_clf.score(X_val, y_val))
    plt.plot(np.linspace(-5, 5, num=30), accuracy_val, label="validation accuracy")
    plt.plot(np.linspace(-5, 5, num=30), accuracy_train, label="'train accuracy")
    plt.axis([-5, 5, 0, 1.2])
    plt.xlabel("C= 10^i")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    gamma_range = np.logspace(-5,5, num=30)
    accuracy_train = []
    accuracy_val= []
    for gamma in gamma_range:
        linear_clf = svm.SVC(gamma=gamma, kernel='rbf', C=10)
        linear_clf.fit(X_train, y_train)
        accuracy_train.append(linear_clf.score(X_train, y_train))
        accuracy_val.append(linear_clf.score(X_val, y_val))
    plt.plot(np.linspace(-5, 5, num=30), accuracy_val, label="validation accuracy")
    plt.plot(np.linspace(-5, 5, num=30), accuracy_train, label="'train accuracy")
    plt.axis([-5, 5, 0, 1.2])
    plt.xlabel("gamma = 10^i")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
