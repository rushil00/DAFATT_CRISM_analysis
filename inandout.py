import numpy as np
from scipy.io import loadmat, savemat

def getTargetLibrary(tlib_path :str):
    if tlib_path.split(".")[-1]== "npy":
        target_lib= np.load(tlib_path)
    # elif tlib_path.split(".")[-1]== "mat":
    #     target_lib = loadmat(tlib_path)
    
    targetLibraryRef= target_lib[1:,1:]
    targetLibraryName= target_lib[0,:]
    return targetLibraryRef, targetLibraryName


from spectral import open_image
def getImage(img_path):
    return open_image(img_path)