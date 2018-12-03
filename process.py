import gpflow
import cv2
import numpy as np

import util


def stage1(l_r, scale=3, overlap=1/3, sl=20, sh=40):
    h_b = cv2.resize(l_r, (1080, 600), interpolation = cv2.INTER_CUBIC)
    # h_b = np.ones((1000, 1000))
    l_r = np.array(l_r)
    h_b = np.array(h_b)

    lh, lw = l_r.shape
    hh, hw = h_b.shape
    stride = sh * overlap
    h_patch_n = int((hh - sh) / stride + 1)
    w_patch_n = int((hw - sh) / stride + 1)
    h_b_upper_left = util.get_upper_left_coordinate(hh, hw, sh, h_patch_n, w_patch_n)
    h_b_patches = util.get_patches(h_b, h_b_upper_left, sh, h_patch_n, w_patch_n)
    l_upper_left = util.get_upper_left_coordinate(lh, lw, sl, h_patch_n, w_patch_n)
    l_patches = util.get_patches(l_r, l_upper_left, sl, h_patch_n, w_patch_n)

    X_train = None
    Y_train = None
    ii = 0
    h_patch_n = 2
    w_patch_n = 2
    for i in range(h_patch_n):
        for j in range(w_patch_n):
            l_patch = l_patches[i][j]
            X, y = util.get_set(l_patch)

            if X_train is None:
                X_train = np.hstack((X, np.zeros((X.shape[0], 1))))
                Y_train = np.hstack((y, np.zeros_like(y)))
            else:
                X_train = np.vstack((X_train, np.hstack((X, np.ones((X.shape[0], 1)) * ii))))
                Y_train = np.vstack((Y_train, np.hstack((y, np.ones_like(y) * ii))))

            ii = ii + 1
            print(ii)

    lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gaussian() for _ in range(h_patch_n * w_patch_n)])
    k1 = gpflow.kernels.Matern32(8, active_dims=[0, 1, 2, 3, 4, 5, 6, 7])
    # what is rank?
    # rank of coreg matrix - number of latent functions that share the same covariance kernel k(x,x')
    coreg = gpflow.kernels.Coregion(input_dim=1, output_dim=h_patch_n * w_patch_n, rank=1, active_dims=[8])
    kern = k1 * coreg
    M = 50
    Z = np.vstack([np.hstack([np.random.randn(M, 8), np.ones((M, 1)) * i]) for i in range(h_patch_n * w_patch_n)])
    # lik = gpflow.likelihoods.Gaussian()
    m = gpflow.models.SVGP(X_train, Y_train, kern=kern, likelihood=lik, num_latent=1, Z=Z)
    m.feature.trainable = False
    gpflow.train.ScipyOptimizer().minimize(m, maxiter=10000, disp=True)

    ii = 0
    for i in range(h_patch_n):
        for j in range(w_patch_n):
            h_patch = h_b_patches[i][j]
            Xt, yt = util.get_set(h_patch)
            X_test = np.hstack((Xt, np.ones((Xt.shape[0], 1)) * ii))
            print(X_test.shape)
            mu, var = m.predict_f(X_test)
            print(np.sum(mu - yt))
            mu = np.reshape(mu, (sh - 2, sh - 2))
            h_patch[1:-1, 1:-1] = mu
            h_b_patches[i][j] = h_patch
            ii = ii + 1
            print(ii)

    h_r_blur = np.zeros((hh, hw))
    h_r_blur = util.construct_patch(h_r_blur, h_b_patches, h_b_upper_left, sh, h_patch_n, w_patch_n)
    # np.testing.assert_array_equal(h_b, h_r_blur)

    return h_r_blur


def stage2(h_r_blur, l_r, scale=3, overlap=1/3, sl=20, sh=40):
    l_r_blur = cv2.resize(h_r_blur, (360, 200), interpolation = cv2.INTER_CUBIC)
    l_r_blur = util.de_blur(l_r_blur)
    l_r_blur = np.array(l_r_blur)
    h_r_blur = np.array(h_r_blur)

    hh, hw = h_r_blur.shape
    lh, lw = l_r.shape
    stride = sh * overlap
    h_patch_n = int((hh - sh) / stride + 1)
    w_patch_n = int((hw - sh) / stride + 1)
    h_blur_upper_left = util.get_upper_left_coordinate(hh, hw, sh, h_patch_n, w_patch_n)
    h_blur_patches = util.get_patches(h_r_blur, h_blur_upper_left, sh, h_patch_n, w_patch_n)
    l_upper_left = util.get_upper_left_coordinate(lh, lw, sl, h_patch_n, w_patch_n)
    l_patches = util.get_patches(l_r, l_upper_left, sl, h_patch_n, w_patch_n)
    l_blur_upper_left = l_upper_left.copy()
    l_blur_patches = util.get_patches(l_r_blur, l_blur_upper_left, sl, h_patch_n, w_patch_n)

    X_train = None
    Y_train = None
    ii = 0
    for i in range(h_patch_n):
        for j in range(w_patch_n):
            l_patch = l_patches[i][j]
            l_blur_patch = l_blur_patches[i][j]
            X, y = util.get_set(l_patch)
            X_blur, y_blur = util.get_set(l_blur_patch)
            if X_train is None:
                X_train = np.hstack((X_blur, np.zeros((X_blur.shape[0], 1))))
                Y_train = np.hstack((y, np.zeros_like(y)))
            else:
                X_train = np.vstack((X_train, np.hstack((X_blur, np.ones((X_blur.shape[0], 1)) * ii))))
                Y_train = np.vstack((Y_train, np.hstack((y, np.ones_like(y) * ii))))

            ii = ii + 1
            print(ii)

    lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gaussian() for _ in range(h_patch_n * w_patch_n)])
    k1 = gpflow.kernels.Matern32(8, active_dims=[0, 1, 2, 3, 4, 5, 6, 7])
    # what is rank?
    coreg = gpflow.kernels.Coregion(1, output_dim=h_patch_n*w_patch_n, rank=1, active_dims=[8])
    kern = k1 * coreg
    m = gpflow.models.VGP(X_train, Y_train, kern=kern, likelihood=lik, num_latent=1)
    gpflow.train.ScipyOptimizer().minimize(m, maxiter=30000, disp=True)

    ii = 0
    for i in range(h_patch_n):
        for j in range(w_patch_n):
            h_blur_patch = h_blur_patches[i][j]
            Xt, yt = util.get_set(h_blur_patch)
            X_test = np.hstack((Xt, np.ones((Xt.shape[0], 1)) * ii))

            mu, var = m.predict_f(X_test)
            mu = np.reshape(mu, (sh - 2, sh - 2))
            h_blur_patch[1:-1, 1:-1] = mu
            h_blur_patches[i][j] = h_blur_patch
            ii = ii + 1

    h_r = np.zeros((hh, hw))
    h_r = util.construct_patch(h_r, h_blur_patches, h_blur_upper_left, sh, h_patch_n, w_patch_n)
    # np.testing.assert_array_equal(h_r_blur, h_r)

    return h_r