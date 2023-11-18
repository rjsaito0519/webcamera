import tifffile
import glob

filepath = glob.glob("./*.tiff")

image = tifffile.imread(filepath[0])
tifffile.imwrite('temp.tiff', image)
