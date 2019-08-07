import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def init_network():
    return {'W1': np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]), 'b1': np.array([[0.1, 0.2, 0.3]]),
            'W2': np.array([[0.1, 0.4], [0.2, 0.5], [0.3, .6]]), 'b2': np.array([[0.1, 0.2]]),
            'W3': np.array([[0.1, 0.3], [0.2, 0.4]]), 'b3': np.array([[0.1, 0.2]])}


def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    return a3


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


nw = init_network()
xx = np.array([1, .5])
print(forward(nw, xx))

print(softmax(np.array([0.2, 3, 2])))

