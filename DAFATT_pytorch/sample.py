import torch
from algorithms import *
from scipy.io import loadmat
from spectral import open_image

# Check if CUDA is available
# if torch.cuda.is_available():
#     # Set the default tensor type to CUDA tensors
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Define data and endmembers
M = torch.randn(240000, 500).cuda()  # Random data matrix (p x N) on GPUs
U = torch.randn(240000, 12).cuda()     # Random endmembers (p x q) on GPU
W = hyperUcls(M, U)
# Load CRISM data
data_initial = open_image('../nili_fosae/mtrdr/2006/2006_345/HRL/hrl0000b8c2_07_if183j_mtr3.hdr')
# Set data range
(xx_1, xx_2), (yy_1, yy_2) = (351, 480), (200, 350)
# data = torch.tensor(data_initial[yy_1:yy_2, xx_1:xx_2, :])
data = torch.tensor(data_initial[150:157, 150:157, 178:313])
nline, nsample, nband = data.shape
data1 = torch.reshape(data[:, :,:], (nsample*nline, nband))

TargetLibrary_data = loadmat('D:\RUSHIL-2021-24\Programming_2022-23\isro_project_dec\\factor_analysis\TargetLibrary_paper.mat')
TargetLibraryRef = torch.tensor(TargetLibrary_data['TargetLibrary'][104:-1, 1:55])
TargetLibraryName = TargetLibrary_data['TargetLibraryName'][0:]
n = TargetLibraryRef.shape[0]
wave = torch.tensor(TargetLibrary_data['TargetLibrary'][104:-1, 0])

noise_type = 'additive'
verbose = 'off'
w, Rn = estNoise(data1.T, noise_type)
kf, Ek = hysime(data[:, :, :], w, Rn)
print("Noise estimate for every pixel (w.shape): ",w.shape)
print("Noise estimate for noise correlation (Rn.shape): ",Rn.shape)
print("Ek.shape: ",Ek.shape)
print("kf: ",kf)
# print("Ek: ",Ek)
print("data1.shape: ",data1.shape)

# =======================================================================

def dafatt(data, targetlibrary):
    nline, nsample, nband = data.shape
    data1 = torch.reshape(data, (nsample*nline, nband))
    Normaldata = data1 - torch.mean(data1, dim=1, keepdim=True)
    C = torch.matmul(Normaldata, Normaldata.t()) / data1.shape[1]
    eigenvectors, eigenvalues, _ = torch.svd(C)
    eigva = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)
    noise_type = 'additive'
    verbose = 'off'
    w, Rn = estNoise(data1.T, noise_type)
    kf, Ek = hysime(data[:, :, :], w, Rn)
    # targetlibrary= TargetLibraryRef
    L, P = targetlibrary.shape
    targetlibraryNor = targetlibrary / torch.tile(torch.sum(targetlibrary, dim=0, keepdim=True), (data1.shape[1], 1))
    model = torch.zeros_like(targetlibraryNor, dtype=torch.double)
    model1 = torch.zeros_like(targetlibraryNor,dtype=torch.double)
    NorRMSE = torch.zeros(targetlibraryNor.shape[1],dtype=torch.double)
    RMSE = torch.zeros(targetlibraryNor.shape[1],dtype=torch.double)
    for i in range(targetlibrary.shape[1]):
        X_hat_tv_i= UCLS(targetlibrary[:,i],Ek)
        model[:,i] = torch.matmul(Ek.double(),X_hat_tv_i)
        model1[:, i] = model[:, i] / torch.sum(model[:, i], dim=0)
        NorRMSE[i] = torch.sqrt(torch.sum((model1[:, i] - targetlibraryNor[:, i]) ** 2) / L)
    return kf, Ek, NorRMSE, model, model1, targetlibraryNor


kf, Ek, NorRMSE, model, model1,targetlibraryNor = dafatt(data, TargetLibraryRef)
# print(f'NorRMSE: {NorRMSE}')
# print(f'NorRMSE.shape: {NorRMSE.shape}')


# =========================================================================
# a= [7]
# b= [7]
# print(n)
# detect_square = torch.zeros((nline, nsample, nband))
# DETECT = {}
# # for window in range(len(a)):
# detect = torch.zeros((nline, nsample, n))
#     # for i in range(1, nline - a[window] + 1):
#     #     for j in range(1, nsample - b[window] + 1):
# i,j = 1,1
# window= 0
# data1 = data[i:i + a[window], j:j + b[window], :]
# kf, Ek, NorRMSE, model, model1,targetlibraryNor = dafatt(data1, TargetLibraryRef)
# for num in range(n):
#     if NorRMSE[num] <= 1.16e-4:
#         detect[i:i + a[window] - 1, j:j + b[window] - 1, num] = 1
# detect_square[:,:,:]= detect
# # DETECT[window]= detect_square

# # w1 = DETECT[0].astype(bool)
# # inter = torch.zeros((nline,nsample , nband))
# # for i in range(n):
# #     inter[:, :, i] = w1[:, :, i]
# print(f'NorRMSE: {NorRMSE}')
# print(f'model.shape: {model.shape}')
# # print(inter)
# # print(f'targetLibraryNor: {targetlibraryNor}')
