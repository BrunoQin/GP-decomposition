import gpflow
import numpy as np
import netCDF4

import prepare
import stage1
import stage2

if __name__ == '__main__':
    data = prepare.origin_data('/Users/macbookpro/Desktop/13_3_origin.nc')
    data = np.array(data)

