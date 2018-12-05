import gpflow
import cv2
import numpy as np
import tensorflow as tf
from concurrent import futures
from threading import Lock

import util


def _predict(x, y, xt, lock, graph=None):
    # lock is created in parent thread and passed to all children
    with tf.Session(graph=tf.Graph()) as sess:
        #Lock on the AutoBuild global
        lock.acquire()
        try:
            with gpflow.defer_build():
                kern = gpflow.kernels.RBF(8)
                model = gpflow.models.VGP(x, y, kern, likelihood=gpflow.likelihoods.Gaussian(), num_latent=1)
        finally:
            lock.release()
        model.compile()
        # training etc here
        gpflow.train.ScipyOptimizer().minimize(model)
        ystar, varstar = model.predict_f(xt)
    return ystar


def async(X, Y, Xt, max_workers=20):
    ###
    # Possibly do some GP optimization etc.
    # then run some gpflow operations in threads
    jobs = []
    graph = tf.Graph
    lock = Lock()
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for id, (x, y, xt) in enumerate(zip(X, Y, Xt)):
            jobs.append(executor.submit(_predict, x, y, xt, lock, graph=graph))
        futures.wait(jobs)
        results = [j.result() for j in jobs]
    return results


def stage1(l_r, scale=3, overlap=1/3, sl=20, sh=40):
    l_r = np.array(l_r)
    lh, lw = l_r.shape
    h_b = cv2.resize(l_r, (lh*scale, lw*scale), interpolation = cv2.INTER_CUBIC)
    # h_b = np.ones((1000, 1000))
    h_b = np.array(h_b)

    hh, hw = h_b.shape
    stride = sh * overlap
    h_patch_n = int((hh - sh) / stride + 1)
    w_patch_n = int((hw - sh) / stride + 1)
    h_b_upper_left = util.get_upper_left_coordinate(hh, hw, sh, h_patch_n, w_patch_n)
    h_b_patches = util.get_patches(h_b, h_b_upper_left, sh, h_patch_n, w_patch_n)
    l_upper_left = util.get_upper_left_coordinate(lh, lw, sl, h_patch_n, w_patch_n)
    l_patches = util.get_patches(l_r, l_upper_left, sl, h_patch_n, w_patch_n)

    X_train = []
    Y_train = []
    X_test = []
    for i in range(h_patch_n):
        for j in range(w_patch_n):
            l_patch = l_patches[i][j]
            h_patch = h_b_patches[i][j]
            X, y = util.get_set(l_patch)
            Xt, yt = util.get_set(h_patch)
            X_train.append(X)
            Y_train.append(y)
            X_test.append(Xt)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)

    results = async(X_train, Y_train, X_test, 8)

    ii = 0
    for i in range(h_patch_n):
        for j in range(w_patch_n):
            h_patch = h_b_patches[i][j]
            mu = np.reshape(results[ii], (sh - 2, sh - 2))
            h_patch[1:-1, 1:-1] = mu
            h_b_patches[i][j] = h_patch
            ii = ii + 1
            print(ii)

    h_r_blur = np.zeros((hh, hw))
    h_r_blur = util.construct_patch(h_r_blur, h_b_patches, h_b_upper_left, sh, h_patch_n, w_patch_n)
    # np.testing.assert_array_equal(h_b, h_r_blur)

    return h_r_blur


def stage2(h_r_blur, l_r, scale=3, overlap=1/3, sl=20, sh=40):
    h_r_blur = np.array(h_r_blur)
    hh, hw = h_r_blur.shape
    l_r_blur = cv2.resize(h_r_blur, (hh/scale, hw/scale), interpolation = cv2.INTER_CUBIC)
    l_r_blur = util.de_blur(l_r_blur)
    l_r_blur = np.array(l_r_blur)

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

    X_train = []
    Y_train = []
    X_test = []
    h_patch_n = 2
    w_patch_n = 2
    for i in range(h_patch_n):
        for j in range(w_patch_n):
            l_patch = l_patches[i][j]
            l_blur_patch = l_blur_patches[i][j]
            h_blur_patch = h_blur_patches[i][j]
            X1, y1 = util.get_set(l_patch)
            X2, y2 = util.get_set(l_blur_patch)
            Xt, yt = util.get_set(h_blur_patch)
            X_train.append(X2)
            Y_train.append(y1)
            X_test.append(Xt)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)

    results = async(X_train, Y_train, X_test)

    ii = 0
    for i in range(h_patch_n):
        for j in range(w_patch_n):
            h_blur_patch = h_blur_patches[i][j]
            mu = np.reshape(results[ii], (sh - 2, sh - 2))
            h_blur_patch[1:-1, 1:-1] = mu
            h_blur_patches[i][j] = h_blur_patch
            ii = ii + 1
            print(ii)

    h_r = np.zeros((hh, hw))
    h_r = util.construct_patch(h_r, h_blur_patches, h_blur_upper_left, sh, h_patch_n, w_patch_n)
    # np.testing.assert_array_equal(h_r_blur, h_r)

    return h_r
