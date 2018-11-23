from time import time
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

img = np.ones((128, 256))

height, width = img.shape


# Extract all reference patches from the left half of the image
print('Extracting reference patches...')
t0 = time()
patch_size = (128, 128)
data = extract_patches_2d(img, patch_size)
#data = data.reshape(data.shape[0], -1)
#intercept = np.mean(data, axis=0)
#data -= intercept
print('done in %.2fs.' % (time() - t0))

b = reconstruct_from_patches_2d(data, img.shape)


