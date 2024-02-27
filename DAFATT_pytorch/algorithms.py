import torch
# Check if GPU is available

# Check if CUDA is available
# if torch.cuda.is_available():
#     # Set the default tensor type to CUDA tensors
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)


def hyperUcls(M, U):
    """
    Unconstrained least squares abundance estimation.

    Parameters:
    - M: 2D data matrix (p x N) or a 1D column vector (p,) if N == 1
    - U: 2D matrix of endmembers (p x q)

    Returns:
    - W: Abundance maps (q x N)
    """
    if M.ndim == 1:
        # If M is a 1D column vector, reshape it to a 2D matrix
        M = M.reshape(-1, 1)
    elif M.ndim != 2:
        raise ValueError('M must be a p x N matrix or a 1D column vector if N == 1.')

    p1, N = M.shape
    p2, q = U.shape

    if p1 != p2:
        raise ValueError('M and U must have the same number of spectral bands.')

    Minv = torch.pinverse(torch.tensor(U))
    W = torch.zeros((q, N))

    for n1 in range(N):
        W[:, n1] = Minv @ torch.tensor(M[:, n1])

    return W
# =======================================================================================================
# Unconstrained Least Squares Abundance estimation Algo
# =======================================================================================================
def UCLS(M, U):
    """
    Performs unconstrained least squares abundance estimation.

    Parameters:
        M: `torch.Tensor`
            2D data matrix (N x p).

        U: `torch.Tensor`
            2D matrix of endmembers (q x p).

    Returns: `torch.Tensor`
        An abundance maps (N x q).
    """
    Uinv = torch.linalg.pinv(U.T)
    return torch.matmul(Uinv.to(torch.double), M.to(torch.double).T).T

# ===============================================================================================================
# Noise Estimation Algorithm
# ===============================================================================================================
import torch
def estNoise(r, noise_type='additive'):
    """
    This function infers the noise in a
    hyperspectral data set, by assuming that the
    reflectance at a given band is well modelled
    by a linear regression on the remaining bands.

    Parameters:
        y: `numpy array`
            a HSI cube ((m*n) x p) <=> L x N

       noise_type: `string [optional 'additive'|'poisson']`

    Returns: `tuple numpy array, numpy array`
        * the noise estimates for every pixel (L x p)
        * the noise correlation matrix estimates (p x p)

    Copyright:
        Jose Nascimento (zen@isel.pt) and Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """
    # def est_additive_noise(r):
    #     small = 1e-6
    #     L, N = r.shape
    #     w=np.zeros((L,N), dtype=np.float)
    #     RR=np.dot(r,r.T)
    #     RRi = np.linalg.pinv(RR+small*np.eye(L))
    #     RRi = np.matrix(RRi)
    #     for i in range(L):
    #         XX = RRi - (RRi[:,i]*RRi[i,:]) / RRi[i,i]
    #         RRa = RR[:,i]
    #         RRa[i] = 0
    #         beta = np.dot(XX, RRa)
    #         beta[0,i]=0;
    #         w[i,:] = r[i,:] - np.dot(beta,r)
    #     Rw = np.diag(np.diag(np.dot(w,w.T) / N))
    #     return w, Rw
    r = r.t()
    small = 1e-6
    L, N = r.shape
    w=torch.zeros((L,N), dtype=torch.float,device=r.device)
    RR=r@r.T
    # print((small*torch.eye(L,device=r.device)).device)
    temp=RR+small*torch.eye(L,device=r.device)
    # print(temp.device)
    RRi = torch.pinverse(temp)

    # RRi = np.matrix(RRi)
    for i in range(L):
        XX = RRi - (RRi[:,i].unsqueeze(1)*RRi[i,:].unsqueeze(0)) / RRi[i,i] #potential check-point
        RRa = RR[:,i]
        RRa[i] = 0
        beta =XX@RRa
        beta[i]=0
        w[i,:] = r[i,:] - beta@r
    Rw = torch.diag(torch.diag((w.T@w) / N))
    return w.T, Rw.T
# ==================================================================
def estNoise_1(y, noise_type='additive'):
    """
    This function infers the noise in a
    hyperspectral data set, by assuming that the
    reflectance at a given band is well modelled
    by a linear regression on the remaining bands.

    Parameters:
        y: `torch tensor`
            a HSI cube ((m*n) x p)

       noise_type: `string [optional 'additive'|'poisson']`

    Returns: `tuple torch tensor, torch tensor`
        * the noise estimates for every pixel (N x p)
        * the noise correlation matrix estimates (p x p)

    Reference from:
        Jose Nascimento and Jose Bioucas-Dias 
        For any comments contact the authors
    """
    def est_additive_noise(r):

        small = 1e-6
        L, N = r.shape
        w = torch.zeros((L, N), dtype=torch.float64)
        RR = torch.matmul(r, r.t())
        RRi = torch.pinverse(RR + small * torch.eye(L))
        for i in range(L):
            XX = RRi - (RRi[:, i].view(-1, 1) * RRi[i, :]) / RRi[i, i]
            RRa = RR[:, i]
            RRa[i] = 0
            beta = torch.matmul(XX, RRa)
            beta[0, i] = 0
            w[i, :] = r[i, :] - torch.matmul(beta, r)
        Rw = torch.diag(torch.diag(torch.matmul(w, w.t()) / N))
        return w, Rw

    y = y.T
    L, N = y.shape
    if noise_type == 'poisson':
        sqy = torch.sqrt(y * (y > 0))
        u, Ru = est_additive_noise(sqy)
        x = (sqy - u)**2
        w = torch.sqrt(x) * u * 2
        Rw = torch.matmul(w.t(), w) / N
    else:
        w, Rw = est_additive_noise(y)
    return w, Rw.T


# ==================================================================================================
# Hysime Algorithm
# ==================================================================================================
# taken from https://github.com/bearshng/mac-net
def hysime(y, n, Rn):
    """
    Hyperspectral signal subspace estimation

    Parameters:
        y: `numpy array`
            hyperspectral data set (each row is a pixel)
            with ((m*n) x p), where p is the number of bands
            and (m*n) the number of pixels.

        n: `numpy array`
            ((m*n) x p) matrix with the noise in each pixel.

        Rn: `numpy array`
            noise correlation matrix (p x p)

    Returns: `tuple integer, numpy array`
        * kf signal subspace dimension
        * Ek matrix which columns are the eigenvectors that span
          the signal subspace.

    Copyright:
        Jose Nascimento (zen@isel.pt) & Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """
    h, w, numBands = y.shape
    y = torch.reshape(y, (w * h, numBands))
    y=y.T
    n=n
    Rn=Rn.T
    L, N = y.shape
    Ln, Nn = n.shape
    d1, d2 = Rn.shape

    x = y - n.T
    # print(f'x.shape: {x.shape}')

    Ry = (y@y.T) / N
    Rx = (x@x.T)/ N
    # print(f'Rx.shape: {Rx.shape}')
    # print(f'Ry.shape: {Ry.shape}')
    E, dx, V =torch.svd(Rx) #torch.svd(Rx.cpu())?
    E=E.to(device=y.device)
    # print(V)
    Rn = Rn+torch.sum(torch.diag(Rx))/L/10**5 * torch.eye(L,device=y.device)
    Py = torch.diag(E.T@(Ry@E))
    Pn = torch.diag(E.T@(Rn@E))
    cost_F = -Py + 2 * Pn
    kf = 1+torch.sum(cost_F < 0)
    ind_asc = torch.argsort(cost_F)
    Ek = E[:, ind_asc[0:kf]]
    return kf, Ek # Ek.T ?
# =====================================================================
#original hysime self-refactored:
def hysime_1(y, n, Rn):
    """
    Hyperspectral signal subspace estimation

    Parameters:
        y: `torch tensor`
            hyperspectral data set (each row is a pixel)
            with ((m*n) x p), where p is the number of bands
            and (m*n) the number of pixels.

        n: `torch tensor`
            ((m*n) x p) matrix with the noise in each pixel.

        Rn: `torch tensor`
            noise correlation matrix (p x p)

    Returns: `tuple integer, torch tensor`
        * kf signal subspace dimension
        * Ek matrix which columns are the eigenvectors that span
          the signal subspace.

    Reference from:
        Jose Nascimento & Jose Bioucas-Dias
    """

    y = y.t()
    n = n.t()
    Rn = Rn.t()
    L, N = y.shape
    Ln, Nn = n.shape
    d1, d2 = Rn.shape

    x = y - n

    Ry = torch.matmul(y, y.t()) / N
    Rx = torch.matmul(x, x.t()) / N
    E, dx, V = torch.svd(Rx)

    Rn = Rn + torch.sum(torch.diag(Rx)) / L / (10 ** 5) * torch.eye(L)
    Py = torch.diag(torch.matmul(E.t(), torch.matmul(Ry, E)))
    Pn = torch.diag(torch.matmul(E.t(), torch.matmul(Rn, E)))
    cost_F = -Py + 2 * Pn
    kf = torch.sum(cost_F < 0)
    _, ind_asc = torch.sort(cost_F)
    Ek = E[:, ind_asc[0:kf]]
    return kf, Ek


# ===============================================================================================================
# Factor Analysis and Target Transformation
# ===============================================================================================================
import torch

def FATT(data, targetlibrary, targetlibraryName, wavelength, **kwargs):
    """
    Using Factor Analysis and Target Transformation (FATT) to find endmembers.

    Parameters:
    - data: Mixing matrix [nb (channels) x px (pixel number)]
    - targetlibrary: Target spectra matrix [nb (channels) x targSpec (target spectra number)]
    - targetlibraryName: Mineral name of each spectrum in the target library
    - wavelength: Wavelength array [nb x 1], unit: micrometer
    - kwargs:
        - 'EigNumDM': Eigenvector number determination method
                      Options: 'FATTPaper', 'SpectralInfo', 'Hysime'
                      Default: 'FATTPaper'

    Returns:
    - kf: The eigenvector number
    - NorRMSE: Root Mean Squared Error (RMSE) between the normalized library spectra and modeled spectra
    - model: Modeled spectra matrix [nb x targSpec]
    """
    nline, nsample, nband = data.shape
    data1 = torch.reshape(data, (nsample*nline, nband))
    L, P = targetlibrary.shape
    LM, N = data1.shape
    # Factor analysis
    Normaldata = data1 - torch.mean(data1, dim=1, keepdim=True)
    C = torch.matmul(Normaldata, Normaldata.t()) / N
    eigenvectors, eigenvalues, _ = torch.svd(C)
    eigva = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    if 'EIGNUMDM' in kwargs:
        eignum = kwargs['EIGNUMDM']
    else:
        eignum = 'FATTPaper'  # Default

    if eignum == 'Hysime':
        noise_type = 'additive'
        verbose = 'off'
        w, Rn = estNoise(data1.T, noise_type)
        kf, Ek = hysime(data[:,:,:], w, Rn)
    else:
        if eignum == 'SpectralInfo':
            Info = torch.sum(eigva)
            cum = torch.abs(torch.cumsum(eigva, dim=0) / Info - 0.9995)
            kf = torch.argmin(cum)
            Ek = eigenvectors[:, :kf + 1]
        else:
            if eignum == 'FATTPaper':
                kf = 10  # Default eigenvector number
                Ek = eigenvectors[:, :kf + 1]

    targetlibraryNor = targetlibrary / torch.tile(torch.sum(targetlibrary, dim=0, keepdim=True), (N, 1))
    model = torch.zeros_like(targetlibraryNor, dtype=torch.double)
    model1 = torch.zeros_like(targetlibraryNor,dtype=torch.double)
    NorRMSE = torch.zeros(targetlibraryNor.shape[1],dtype=torch.double)
    RMSE = torch.zeros(targetlibraryNor.shape[1],dtype=torch.double)
    for i in range(targetlibrary.shape[1]):
        X_hat_tv_i= UCLS(targetlibrary[:,i],Ek)
        model[:,i] = torch.matmul(Ek.double(),X_hat_tv_i)
        model1[:, i] = model[:, i] / torch.sum(model[:, i], dim=0)
        NorRMSE[i] = torch.sqrt(torch.sum((model1[:, i] - targetlibraryNor[:, i]) ** 2) / L)

    # X_hat_tv_i = UCLS(targetlibrary.T, Ek.T)
    # model = torch.matmul(Ek, X_hat_tv_i.t())
    # model1 = model / torch.tile(torch.sum(model, dim=0, keepdim=True), (data.shape[0], 1))
    # print("TARGETLIBRARY SHAPE:",targetlibrary.shape[1])
    # for i in range(targetlibrary.shape[1]):
    #     NorRMSE[i] = torch.sqrt(torch.sum((model1[:, i] - targetlibraryNor[:, i]) ** 2) / data.shape[0])
    #     RMSE[i] = torch.sqrt(torch.sum((model[:, i] - targetlibrary[:, i]) ** 2) / L)

    return kf, Ek, NorRMSE, model, model1, targetlibraryNor

def normalize_columns(matrix):
    return matrix / torch.sum(matrix, dim=0, keepdim=True)
# ===================================================================
from tqdm import tqdm
def dafatt(data, targetlibrary):
    nline, nsample, nband = data.shape
    data1 = torch.reshape(data, (nsample*nline, nband))
    # Normaldata = data1 - torch.mean(data1, dim=1, keepdim=True)
    # C = torch.matmul(Normaldata, Normaldata.t()) / data1.shape[1]
    # eigenvectors, eigenvalues, _ = torch.svd(C)
    # eigva = eigenvalues.flip(0)
    # eigenvectors = eigenvectors.flip(1)
    noise_type = 'additive'
    verbose = 'off'
    w, Rn = estNoise(data1.T, noise_type)
    kf, Ek = hysime(data[:, :, :], w.T, Rn)
    # targetlibrary= TargetLibraryRef
    L, P = targetlibrary.shape
    targetlibraryNor = targetlibrary / torch.tile(torch.sum(targetlibrary, dim=0, keepdim=True), (data1.shape[1], 1))
    model = torch.zeros_like(targetlibraryNor, dtype=torch.double)
    model1 = torch.zeros_like(targetlibraryNor,dtype=torch.double)
    NorRMSE = torch.zeros(targetlibraryNor.shape[1],dtype=torch.double)
    RMSE = torch.zeros(targetlibraryNor.shape[1],dtype=torch.double)
    for i in tqdm(range(targetlibrary.shape[1]), desc= "TargetTransform", leave= False):
        X_hat_tv_i= UCLS(targetlibrary[:,i].double().unsqueeze(0),Ek.double().t())
        # print(f'x_hat_tv_i.shape: {X_hat_tv_i.shape}')
        model[:,i] = torch.matmul(Ek.double(),X_hat_tv_i.t()).t().squeeze(0)
        model1[:, i] = model[:, i] / torch.sum(model[:, i], dim=0)
        NorRMSE[i] = torch.sqrt(torch.sum((model1[:, i] - targetlibraryNor[:, i]) ** 2) / L)
    return kf, Ek, NorRMSE, model, model1, targetlibraryNor