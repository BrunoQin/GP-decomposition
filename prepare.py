import netCDF4


def origin_data(file_name):
    file = netCDF4.Dataset(file_name)
    data = file.variables['temp'][0, 0, :, :]
    file.close()
    return data
