import netCDF4


def origin_data(file_name):
    file = netCDF4.Dataset(file_name)
    l_r = file.variables['temp'][0, 0, :, :]
    file.close()
    return l_r
