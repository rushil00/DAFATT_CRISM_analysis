import numpy as np
# from factor_analysis.algorithms import *
from algorithms import *
from hysime import *
def dimensionReduction2d(x, num=2, shape=[], type='pca'): #for the factor analysis portion? LINK: https://github.com/ZhaohuiXue/A-SPN-release/blob/main/data.py
    def pca(x, n_components, shape):
        shp = shape
        data = np.transpose(x, [1, 0])
        data_norm = data - np.mean(data, 1, keepdims=True)

        sigma = np.cov(data_norm)
        [U, S, V] = np.linalg.svd(sigma)

        u = U[:, 0:n_components]
        s = S[0:n_components]
        v = V[0:n_components, :]

        # project to a new column vector space
        data_pca = np.dot(np.transpose(u), data_norm)

        # rescale each variable to unit variance.
        epison = 0.0
        data_pca = np.dot(np.diag((1 / (np.sqrt(s + epison)))), data_pca)
        data_pca = np.transpose(data_pca, [1, 0])
        return data_pca.astype(dtype=np.float32)

    return pca(x, num, shape=shape)


import numpy as np
from scipy.linalg import svd

def FATT(data, targetlibrary, targetlibraryName, wavelength, **kwargs):
    """
    Using Factor Analysis and Target Transformation (FATT) to find endmembers.

    Parameters:
    - data: Mixing matrix [L (channels) x N (pixel number)]
    - targetlibrary: Target spectra matrix [L (channels) x P (target spectra number)]
    - targetlibraryName: Mineral name of each spectrum in the target library
    - wavelength: Wavelength array [L x 1], unit: micrometer
    - kwargs:
        - 'EigNumDM': Eigenvector number determination method
                      Options: 'FATTPaper', 'SpectralInfo', 'Hysime'
                      Default: 'FATTPaper'

    Returns:
    - kf: The eigenvector number
    - NorRMSE: Root Mean Squared Error (RMSE) between the normalized library spectra and modeled spectra
    - model: Modeled spectra matrix [L x P]
    """
    L,P= targetlibrary.shape
    LM,N= data.shape
    # Factor analysis
    Normaldata = data - np.mean(data, axis=1)[:, np.newaxis]
    C = np.dot(Normaldata, Normaldata.T) / N
    eigenvectors, eigenvalues, _ = svd(C)
    eigva = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    if 'EIGNUMDM' in kwargs:
        eignum = kwargs['EIGNUMDM']
    else:
        eignum = 'FATTPaper'  # Default

    if eignum == 'Hysime':
        noise_type = 'poisson'
        verbose = 'off'
        w, Rn = estNoise(data.T, noise_type)
        # print(f"w.shape: {w.shape} \n Rn.shape: {Rn.shape}")
        kf, Ek = hysime(data.T, w, Rn)
    else:
        if eignum == 'SpectralInfo':
            Info = np.sum(eigva)
            cum = np.abs(np.cumsum(eigva) / Info - 0.9995)
            kf = np.argmin(cum)
            Ek = eigenvectors[:, :kf + 1]
        else:
            if eignum == 'FATTPaper':
                kf = 10  # Default eigenvector number
                Ek = eigenvectors[:, :kf + 1]

    # Target Transform
    targetlibraryNor = targetlibrary / np.tile(np.sum(targetlibrary, axis=0), (data.shape[0], 1))
    model = np.zeros_like(targetlibraryNor)
    model1 = np.zeros_like(targetlibraryNor)
    NorRMSE = np.zeros(targetlibraryNor.shape[1])
    RMSE = np.zeros(targetlibraryNor.shape[1])

    # for i in range(targetlibrary.shape[1]):
    #         X_hat_tv_i = hyperUcls(targetlibrary[:,i], Ek)
    #         model[:, i] = np.dot(Ek, X_hat_tv_i.flatten())
    # model1 = model / np.tile(np.sum(model), (data.shape[0], 1))
    # for i in range(targetlibrary.shape[1]):
    #     NorRMSE[i] = np.sqrt(np.sum((model1[:, i] - targetlibraryNor[:, i])**2) / L)
    #  =====================================
    # X_hat_tv= hyperUcls(targetlibrary, Ek)
    # NorRMSE = np.sqrt(np.sum((X_hat_tv - targetlibraryNor)**2) / L)
    # ----------------------------------------------------
    # print(f"Ek.shape: {Ek.shape} || TargetLirbraryNor.shape: {targetlibraryNor.shape}")
    X_hat_tv_i = UCLS(targetlibrary.T, Ek.T)
    # X_hat_tv_i = hyperUcls(targetlibrary, Ek)
    # print(f"X_hat_tv_i.shape= {X_hat_tv_i}")
    model = np.dot(Ek, X_hat_tv_i.T)
    # model = np.dot(Ek, X_hat_tv_i)
    model1 = model / np.tile(np.sum(model,axis=0), (data.shape[0], 1))
    for i in range(targetlibrary.shape[1]):
        NorRMSE[i] = np.sqrt(np.sum((model1[:, i] - targetlibraryNor[:, i])**2) / data.shape[0])
        RMSE[i] = np.sqrt(np.sum((model[:, i] - targetlibrary[:, i])**2) / L)
        
    #  =====================================
    return kf, Ek, NorRMSE, model, model1, targetlibraryNor




# original from logic
# def FATT_(data, targetlibrary, targetlibraryName, wavelength, **kwargs):
#     """
#     Using Factor Analysis and Target Transformation (FATT) to find endmembers.

#     Parameters:
#     - data: Mixing matrix [L (channels) x N (pixel number)]
#     - targetlibrary: Target spectra matrix [L (channels) x P (target spectra number)]
#     - targetlibraryName: Mineral name of each spectrum in the target library
#     - wavelength: Wavelength array [L x 1], unit: micrometer
#     - kwargs:
#         - 'EigNumDM': Eigenvector number determination method
#                       Options: 'FATTPaper', 'SpectralInfo', 'Hysime'
#                       Default: 'FATTPaper'

#     Returns:
#     - kf: The eigenvector    number
#     - NorRMSE: Root Mean Squared Error (RMSE) between the normalized library spectra and modeled spectra
#     - model: Modeled spectra matrix [L x P]
#     """

#     # Factor analysis
#     Normaldata = data - np.mean(data, axis=1)[:, np.newaxis]
#     C = np.dot(Normaldata, Normaldata.T) / data.shape[1]
#     eigenvectors, eigenvalues, _ = svd(C)
#     eigva = eigenvalues[::-1]
#     eigenvectors = eigenvectors[:, ::-1]

#     if 'EIGNUMDM' in kwargs:
#         eignum = kwargs['EIGNUMDM']
#     else:
#         eignum = 'FATTPaper'  # Default

#     if eignum == 'Hysime':
#         noise_type = 'additive'
#         verbose = 'off'
#         w, Rn = estNoise(data, noise_type)
#         kf, Ek = hysime(data, w, Rn)
#     else:
#         if eignum == 'SpectralInfo':
#             Info = np.sum(eigva)
#             cum = np.abs(np.cumsum(eigva) / Info - 0.9995)
#             kf = np.argmin(cum)
#             Ek = eigenvectors[:, :kf + 1]
#         else:
#             if eignum == 'FATTPaper':
#                 kf = 10  # Default eigenvector number
#                 Ek = eigenvectors[:, :kf + 1]

#     # Target Transform
#     targetlibraryNor = targetlibrary / np.tile(np.sum(targetlibrary, axis=0), (data.shape[0], 1))
#     # model = np.zeros_like(targetlibrary)
#     # model1 = np.zeros_like(targetlibraryNor)
#     # NorRMSE = np.zeros(targetlibrary.shape[1])

#     for i in range(targetlibrary.shape[1]):
#         X_hat_tv_i = hyperUcls(targetlibrary[:,i], Ek)
#         model[:, i] = np.dot(Ek, X_hat_tv_i)
#         model1[:, i] = model[:, i] / np.tile(np.sum(model[:, i]), (data.shape[0], 1))
#         NorRMSE[i] = np.sqrt(np.sum((model1[:, i] - targetlibraryNor[:, i])**2) / data.shape[0])

#     return kf, NorRMSE, model