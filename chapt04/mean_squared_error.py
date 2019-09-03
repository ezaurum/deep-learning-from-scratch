import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt



def mean_squared_error(y, t):
    return .5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape
    return -np.sum(t * np.log(y[np.arange(batch_size), t])) / batch_size


def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def function_2(x):
    return x[0] ** 2 + x[1]**2


mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
x1 = np.arange(-10, 10, 0.1)
x0 = np.arange(-10, 10, 0.1)
y = np.sum(x1**2)
ax.plot(x1, x1, y, label='parametric curve')
ax.legend()

plt.show()
