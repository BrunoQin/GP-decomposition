import numpy as np
from sklearn.feature_extraction import image


def get_patches(h, s, h_patch_n, w_patch_n):
    return image.extract_patches_2d(h, (s, s), h_patch_n * w_patch_n)


def get_center():
    pass


def get_set(patch):
    set = image.extract_patches_2d(patch, (3, 3))
    set = np.reshape(set, (-1, 1, 3 * 3))
    set = np.squeeze(set, 1)
    X = np.hstack((set[:, 0:4], set[:, 5:9]))
    y = set[:, 4]
    return X, y


def construct_patch(patches, h, w):
    # todo build the re-construct
    return image.reconstruct_from_patches_2d(patches, (h, w))
