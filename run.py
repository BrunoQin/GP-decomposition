import prepare
import process

if __name__ == '__main__':
    l_r = prepare.origin_data('/Users/macbookpro/Desktop/13_3_origin.nc')
    h_r_blur = process.stage1(l_r)
    h_r = process.stage2(h_r_blur, l_r)
