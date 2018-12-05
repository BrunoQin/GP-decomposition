import numpy as np

X = np.random.rand(100)[:, None] * 10 - 5  # N x D
G = np.hstack((0.5 * np.sin(3 * X) + X, 3.0 * np.cos(X) - X))  # N x L
Ptrue = np.array([[0.5, -0.3, 1.5], [-0.4, 0.43, 0.0]])  # L x P
Y = np.matmul(G, Ptrue)  # N x P
Y += np.random.randn(*Y.shape) * [0.2, 0.2, 0.2]
