import gpflow
import numpy as np
import matplotlib.pyplot as plt

X1 = np.random.rand(100, 1)
X2 = np.random.rand(100, 1) * 0.5
Y1 = np.sin(6*(X1 + X2)) + np.random.standard_t(3, X1.shape)*0.03
Y2 = np.sin(6*(X1 + X2) + 0.7) + np.random.standard_t(3, X2.shape)*0.03

k1 = gpflow.kernels.RBF(2, active_dims=[0, 1])
coreg = gpflow.kernels.Coregion(1, output_dim=2, rank=2, active_dims=[2])
kern = k1 * coreg

# build a variational model. This likelihood switches between Student-T noise with different variances:
lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Gaussian()])

# Augment the time data with ones or zeros to indicate the required output dimension
X_augmented = np.vstack((np.hstack((np.hstack((X1, X2)), np.zeros_like(X1))), np.hstack((np.hstack((X1, X2)), np.ones_like(X2)))))
print(X_augmented.shape)
# Augment the Y data to indicate which likeloihood we should use
Y_augmented = np.vstack((np.hstack((Y1, np.zeros_like(X1))), np.hstack((Y2, np.ones_like(X2)))))
print(Y_augmented.shape)
# now buld the GP model as normal
m = gpflow.models.VGP(X_augmented, Y_augmented, kern=kern, likelihood=lik, num_latent=1)

gpflow.train.ScipyOptimizer().minimize(m, maxiter=2000, disp=True)


def plot_gp(x, mu, var, color='k'):
    plt.plot(x - mu, color='g', lw=2)
    plt.show()
    # plt.plot(x, mu + 2*np.sqrt(var), '--', color=color)
    # plt.plot(x, mu - 2*np.sqrt(var), '--', color=color)


def plot(m):
    # xtest = np.linspace(0, 1, 100)[:,None]
    # line, = plt.plot(X1, Y1, 'x', mew=2)
    mu, var = m.predict_f(np.hstack((np.hstack((X1, X2)), np.zeros_like(X1))))
    plot_gp(Y1, mu, var)
    print(mu)

    # line, = plt.plot(X2, Y2, 'x', mew=2)
    mu, var = m.predict_f(np.hstack((np.hstack((X1, X2)), np.ones_like(X1))))
    plot_gp(Y2, mu, var)
    print(mu)


plot(m)


