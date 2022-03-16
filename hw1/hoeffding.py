import numpy as np
import matplotlib.pyplot as plt

N = 20000
n = 20

mat = np.random.randint(2, size=(N,n))
eps = np.linspace(0,1, 50)
err = np.abs(np.mean(mat, axis=1) - 0.5)
empirical = [np.sum(err > eps[i])/N for i in range(50)]

plt.plot(eps, empirical, "-b", label="empirical")
plt.plot(eps, 2*np.exp(-2*n*(eps**2)), "-r", label="hoeffding")
plt.legend()

plt.show()


