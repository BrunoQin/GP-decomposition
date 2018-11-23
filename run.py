import prepare
import stage1
import stage2

if __name__ == '__main__':
    l_r = prepare.origin_data('/Users/macbookpro/Desktop/13_3_origin.nc')
    h_r_blur = stage1.stage1(l_r)
    h_r = stage2.stage2(h_r_blur, l_r)

