import numpy as np
import gpflow
from matplotlib import pyplot as plt
np.random.seed(0)


# sample from the model
Q = 3
D = 4
k = gpflow.kernels.Matern32(1, lengthscales=2.)
xx = np.linspace(-5, 5, 200).reshape(200, 1)
K = k.compute_K_symm(xx)
f_samples = np.dot(np.linalg.cholesky(K), np.random.randn(200, Q))
W = np.random.randn(Q, D)
rates = np.exp(np.dot(f_samples, W))
Y = np.random.poisson(rates)

# plot the data
fig, axes = plt.subplots(2, 2)
for i, ax in enumerate(axes.flatten()):
    ax.plot(xx, Y[:, i], 'kx')


# build a sparse GP model
M = 30
k = gpflow.kernels.Matern32(1, active_dims=[0]) * gpflow.kernels.Coregion(input_dim=1, output_dim=D, rank=Q, active_dims=[1])
# k.coregion.W = np.random.randn(D, Q)  # random initialization for the weights
X = np.vstack([np.hstack([xx, np.ones((200, 1)) * i]) for i in range(4)])  # repeat the x matrix for each output
Y = Y.reshape(-1, 1, order='F')
Z = np.vstack([np.hstack([np.random.randn(M, 1)*3, np.ones((M, 1)) * i]) for i in range(4)])  # inducing points in every output
lik = gpflow.likelihoods.Poisson()
m = gpflow.models.SVGP(X=X, Y=Y.astype(np.float64), kern=k, likelihood=lik, Z=Z)
m.feature.trainable = False
opt = gpflow.train.ScipyOptimizer()
opt.minimize(m)

# make some predictions and plot
Xtest = np.linspace(-6, 6, 300).reshape(-1, 1)
for i, ax in enumerate(axes.flatten()):
    Xtest_i = np.hstack([Xtest, np.ones((300, 1)) * i])
    mu, var = m.predict_y(Xtest_i)
    print(mu.shape)
    ax.plot(Xtest, mu, 'C0', label='posterior mean')
    ax.plot(xx, rates[:, i], 'C3--', label='ground truth' )
    ax.set_xlim(-6, 6)
plt.legend()
plt.show()
