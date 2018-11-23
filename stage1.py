import gpflow
import cv2
import numpy as np

import util


def stage1(l_r, scale=3, overlap=1/3, sl=20, sh=40):
    h_b = cv2.resize(l_r, (1080, 600), interpolation = cv2.INTER_CUBIC)
    l_r = np.array(l_r)
    h_b = np.array(h_b)

    hh, hw = h_b.shape
    h_patch_n = round(hh/(sh*overlap))
    w_patch_n = round(hw/(sh*overlap))
    h_b_patches = util.get_patches(h_b, sh, h_patch_n, w_patch_n)
    l_patches = util.get_patches(l_r, sl, h_patch_n, w_patch_n)

    for i in range(len(h_b_patches)):
        l_patch = l_patches[i]
        h_patch = h_b_patches[i]
        X, y = util.get_set(l_patch)
        Xt, yt = util.get_set(h_patch)

        M = 50
        kern = gpflow.kernels.RBF(X.shape[1], 1)
        Z = X[:M, :].copy()
        m = gpflow.models.SVGP(X, y, kern, gpflow.likelihoods.Gaussian(), Z, minibatch_size=len(X))

        gpflow.train.ScipyOptimizer().minimize(m)
        mu, var = m.predict_y(Xt)

        mu = np.reshape(mu, (sh - 2, sh - 2))
        h_patch[1:-2, 1:-2] = mu

        h_b_patches[i] = h_patch

    h_r_blur = util.construct_patch(h_b_patches, h_b.shape[0], h_b.shape[1])
    np.testing.assert_array_equal(h_b, h_r_blur)

    return h_r_blur
