import os
import numpy as np
import os
from scipy.io import loadmat, savemat
from freadenvi import *
from fwriteenvi import *
from algorithms import *
from fatt import *
from atgp import *
import numpy as np
import multiprocessing
# directory_path = '../nili_fosae/mtrdr/2006/2006_345/if_img_files/'
# file_list = [file for file in os.listdir(directory_path) if file.endswith('.img')]
from spectral import open_image
def process_window(window, a, b, data, TargetLibraryRef, TargetLibraryName, wave, Fline, Fsample):
    print(f"Processing the {window}th window")

    nline, nsample, nband = data.shape
    n =  TargetLibraryRef.shape[1]
    detect = np.zeros((nline, nsample, n))
    detect_square = np.zeros((Fline, Fsample, n))

    for i in range(1, nline - a[window] + 1):
        for j in range(1, nsample - b[window] + 1):
            data1 = np.reshape(data[i:i + a[window] - 1, j:j + b[window] - 1, :], (a[window] * b[window], nband)).T
            kf, NorRMSE, model = FATT(data1, TargetLibraryRef, TargetLibraryName, wave, EigNumDM='Hysime')

            for num in range(n):
                if NorRMSE[num] <= 1.5e-4:
                    detect[i:i + a[window] - 1, j:j + b[window] - 1, num] = 1

    if Fsample == 640:
        detect_square[:, 31:630, :] = detect
    else:
        detect_square[:, : , :] = detect

    return detect_square

def main():
    # tr_dir = input("Enter the folder path containing CRISM I/F data: ")
    FileNum = 1  # Assuming a single file for simplicity

    # Load CRISM data (replace this with actual loading code)
    data_initial = open_image('D:\RUSHIL-2021-24\Programming_2022-23\isro_project_dec\\nili_fosae\mtrdr\\2006\\2006_345\output\output_mat_1.hdr')
    data= data_initial[10:30,10:35,178:314]
    # Load Target Library
    TargetLibrary_data = loadmat('D:\RUSHIL-2021-24\Programming_2022-23\isro_project_dec\\factor_analysis\TargetLibrary_paper.mat')
    TargetLibraryRef = TargetLibrary_data['TargetLibrary'][104:, 1:]
    TargetLibraryName = TargetLibrary_data['TargetLibraryName'][1:]
    TargetLibraryFileName = TargetLibrary_data['TargetLibraryFileName'][1:]
    n = TargetLibraryRef.shape[1]
    wave = TargetLibrary_data['TargetLibrary'][104:, 0]

    # a = [6, 8, 5, 7, 10]
    # b = [8, 6, 10, 7, 5]
    a = [6]
    b = [8]
    DETECT = {}

    pool = multiprocessing.Pool()

    for window in range(len(a)):
        result = pool.apply_async(process_window, (window, a, b, data, TargetLibraryRef, TargetLibraryName, wave, 20, 25))
        DETECT[window] = result.get()

    pool.close()
    pool.join()

    w1 = DETECT[0]
    w2 = DETECT[1]
    w3 = DETECT[2]
    w4 = DETECT[3]
    w5 = DETECT[4]

    inter = np.zeros((20, 25 , n))

    for i in range(n):
        inter[:, :, i] = w1[:, :, i] & w2[:, :, i] & w3[:, :, i] & w4[:, :, i] & w5[:, :, i]

    # Assuming a filename for output
    # OutputFileName = f"{tr_dir}/DAFATTResults/output.img"
    OutputFileName = f"./DAFATTResults/output.img"

    np.save(OutputFileName, inter)

if __name__ == "__main__":
    main()