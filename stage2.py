import gpflow
import cv2
import numpy as np

import util


def stage2(h_r_blur, l_r, scale=3, overlap=1/3, sl=20, sh=40):
    l_r_blur = cv2.resize(h_r_blur, (360, 200), interpolation = cv2.INTER_CUBIC)
    # todo deblur l_r_blur
    l_r_blur = np.array(l_r_blur)
    h_r_blur = np.array(h_r_blur)

    hh, hw = h_r_blur.shape
    h_patch_n = round(hh/(sh*overlap))
    w_patch_n = round(hw/(sh*overlap))
    h_blur_patches = util.get_patches(h_r_blur, sh, h_patch_n, w_patch_n)
    l_patches = util.get_patches(l_r, sl, h_patch_n, w_patch_n)
    l_blur_patches = util.get_patches(l_r_blur, sl, h_patch_n, w_patch_n)

    for i in range(len(h_blur_patches)):
        l_patch = l_patches[i]
        l_blur_patch = l_blur_patches[i]
        h_blur_patch = h_blur_patches[i]
        X, y = util.get_set(l_patch)
        X_blur, y_blur = util.get_set(l_blur_patch)
        Xt, yt = util.get_set(h_blur_patch)

        M = 50
        kern = gpflow.kernels.RBF(X_blur.shape[1], 1)
        Z = X_blur[:M, :].copy()
        m = gpflow.models.SVGP(X_blur, y, kern, gpflow.likelihoods.Gaussian(), Z, minibatch_size=len(X))

        gpflow.train.ScipyOptimizer().minimize(m)
        mu, var = m.predict_y(Xt)

        mu = np.reshape(mu, (sh - 2, sh - 2))
        h_blur_patch[1:-2, 1:-2] = mu

        h_blur_patches[i] = h_blur_patch

    h_r = util.construct_patch(h_blur_patches, h_r_blur.shape[0], h_r_blur.shape[1])
    np.testing.assert_array_equal(h_r_blur, h_r)

    return h_r
