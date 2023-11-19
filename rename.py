import os
import glob

dir = "./data/241Am"
save_dir = "./data/241Am"

filepath = glob.glob("{}/*.tiff".format(dir))

for path in filepath:
    save_name = os.path.splitext(os.path.basename(path))[0]
    # n = len( glob.glob("{}/*.tiff".format(save_dir)) )
    # save_path = "{}/90Sr_{:0=3}.tiff".format( save_dir, n+1 )
    save_path = "{}/241Am_{}.tiff".format( save_dir, save_name[3:] )
    
    print(save_path)
    os.rename(path, save_path) 
