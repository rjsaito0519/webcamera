import tifffile
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

isTest = 1
y_offset = 2
x_offset = 18
y_slice = slice( y_offset, -1*(y_offset+1) )
x_slice = slice( x_offset, -1*(x_offset+1) )


dir = "./data/241Am"
save_dir = "./data/monazu"
img_name = "*"
filepath = glob.glob("{}/{}.tiff".format(dir, img_name))

for path in filepath:
    image = tifffile.imread(path)[y_slice, x_slice]
    # image = tifffile.imread(path)
    if isTest:
        print(path)
        plt.imshow(image)
        plt.show()
        break
    else:
        save_name = os.path.splitext(os.path.basename(path))[0]
        # n = len( glob.glob("{}/*.tiff".format(save_dir)) )
        # save_name = "Am_{:0=4}".format( n+1 )
        print(save_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        tifffile.imwrite("{}/{}.tiff".format( save_dir, save_name ), image)
