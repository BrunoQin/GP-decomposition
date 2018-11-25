import numpy as np
from sklearn.feature_extraction import image


def get_patches(image, upper_left, s, h_patch_n, w_patch_n):
    patches = [[0] * w_patch_n for i in range(h_patch_n)]
    for i in range(h_patch_n):
        for j in range(w_patch_n):
            patches[i][j] = image[upper_left[i][j][0]:upper_left[i][j][0]+s, upper_left[i][j][1]:upper_left[i][j][1]+s]
    patches = np.array(patches)
    print(patches.shape)
    return patches


def get_upper_left_coordinate(h, w, s, h_patch_n, w_patch_n):
    x = np.round(np.linspace(0, h-s, num=h_patch_n, endpoint=True))
    y = np.round(np.linspace(0, w-s, num=w_patch_n, endpoint=True))
    upper_left = [[0] * w_patch_n for i in range(h_patch_n)]
    for i in range(len(x)):
        for j in range(len(y)):
            upper_left[i][j] = (int(x[i]), int(y[j]))
    return upper_left


def get_set(patch):
    _set = image.extract_patches_2d(patch, (3, 3))
    _set = np.reshape(_set, (-1, 1, 3 * 3))
    _set = np.squeeze(_set, 1)
    X = np.hstack((_set[:, 0:4], _set[:, 5:9]))
    y = _set[:, 4]
    y = np.reshape(y, (-1, 1))
    return X, y


def construct_patch(reconstruct, patches, upper_left, s, h_patch_n, w_patch_n):
    for i in range(h_patch_n):
        temp = np.zeros((s, reconstruct.shape[1]))
        for j in range(w_patch_n):
            if j != 0:
                col = np.array(range(s)) + upper_left[i][j][1]
                inter = np.intersect1d(col, last_col)
                a = np.linspace(0, 1, len(inter))
                W = None
                for k in range(s):
                    if W is None:
                        W = a
                    else:
                        W = np.vstack((W, a))
                patches[i][j][:, 0:len(inter)] = patches[i][j][:, 0:len(inter)] * W
                temp[:, inter] = temp[:, inter] * (1 - W)
                temp[:, upper_left[i][j][1]:upper_left[i][j][1]+s] = temp[:, upper_left[i][j][1]:upper_left[i][j][1]+s] + patches[i][j]
            else:
                temp[:, upper_left[i][j][1]:upper_left[i][j][1]+s] = patches[i][j]
            last_col = np.array(range(upper_left[i][j][1]+s))
        if i != 0:
            row = np.array(range(s)) + upper_left[i][0][0]
            inter = np.intersect1d(row, last_rov)
            a = np.linspace(0, 1, len(inter))
            a = np.reshape(a, (1, -1))
            W = None
            for k in range(reconstruct.shape[1]):
                if W is None:
                    W = a.T
                else:
                    W = np.hstack((W, a.T))
            reconstruct[inter, :] = reconstruct[inter, :] * (1 - W)
            temp[0:len(inter), :] = temp[0:len(inter), :] * W
            reconstruct[upper_left[i][j][0]:upper_left[i][j][0]+s, :] = reconstruct[upper_left[i][j][0]:upper_left[i][j][0]+s, :] + temp
        else:
            reconstruct[upper_left[i][j][0]:upper_left[i][j][0]+s, :] = temp
        last_rov = np.array(range(upper_left[i][j][0]+s))
    return reconstruct
