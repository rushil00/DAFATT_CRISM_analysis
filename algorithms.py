#
# Code taken from PySptools by Christian Therien
# https://pysptools.sourceforge.io/index.html
#
import numpy as np


def hysime_(y, n, Rn):
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

    y = y
    n = n
    # print(f"y.shape: {y.T.shape}   n.shape: {n.T.shape} (inputs)")
    Rn = Rn.T
    L, N = y.shape
    # print(f"y.shape: {y.shape}")
    # print(y.shape,": Shape of hyperspectral data matrix (input)")
    # print(n.shape,": Shape of noise matrix (input)")
    Ln, Nn = n.shape
    d1, d2 = Rn.shape

    x = y - n

    Ry = np.dot(y, y.T) / N
    Rx = np.dot(x, x.T) / N
    E, dx, V = np.linalg.svd(Rx)

    Rn = Rn + np.sum(np.diag(Rx))/L/10**5 * np.eye(L)
    Py = np.diag(np.dot(E.T, np.dot(Ry, E)))
    Pn = np.diag(np.dot(E.T, np.dot(Rn, E)))
    cost_F = -Py + 2 * Pn
    kf = np.sum(cost_F < 0)
    ind_asc = np.argsort(cost_F)
    Ek = E[:, ind_asc[:kf]]
    if Ek.shape[1]==0: # kf= 0 IS HACKED!!
        Ek= E[:,:4]
    return kf, Ek


def estNoise_(y, noise_type='additive'):
    """
    This function infers the noise in a
    hyperspectral data set, by assuming that the
    reflectance at a given band is well modelled
    by a linear regression on the remaining bands.

    Parameters:
        y: `numpy array`
            a HSI cube ((m*n) x p)

       noise_type: `string [optional 'additive'|'poisson']`

    Returns: `tuple numpy array, numpy array`
        * the noise estimates for every pixel (N x p)
        * the noise correlation matrix estimates (p x p)

    Copyright:
        Jose Nascimento (zen@isel.pt) and Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """
    def est_additive_noise(r):

        small = 1e-6
        L, N = r.shape
        w = np.zeros((L, N), dtype=np.float64)
        RR = np.dot(r, r.T)
        RRi = np.linalg.pinv(RR+small*np.eye(L))
        RRi = np.matrix(RRi)
        for i in range(L):
            XX = RRi - (RRi[:, i]*RRi[i, :]) / RRi[i, i]
            RRa = RR[:, i]
            RRa[i] = 0
            beta = np.dot(XX, RRa)
            beta[0, i] = 0
            w[i, :] = r[i, :] - np.dot(beta, r)
        Rw = np.diag(np.diag(np.dot(w, w.T) / N))
        return w, Rw

    y = y.T
    L, N = y.shape
    # verb = 'poisson'
    if noise_type == 'poisson':
        sqy = np.sqrt(y * (y > 0))
        u, Ru = est_additive_noise(sqy)
        x = (sqy - u)**2
        w = np.sqrt(x)*u*2
        Rw = np.dot(w, w.T) / N
    # additive
    else:
        w, Rw = est_additive_noise(y)
    return w.T, Rw.T



# def hyperUcls(M, U):
#     """
#     Unconstrained Least Squares (UCLS) for abundance estimation.

#     Parameters:
#     - M: 2D data matrix (p x N) p: bands , n: pixels
#     - U: 2D matrix of endmembers (p x q) p: bands , q: endmembers

#     Returns:
#     - W: Abundance maps (q x N)
#     """
#     # Input Validation
#     # if M.ndim != 2:
#     #     raise ValueError("M must be a p x N matrix.")
#     print(f"M.shape= {M.shape} , U.shape= {U.shape}")
#     if M.ndim == 1:
#         # If M is a 1D column vector, reshape it to a 2D matrix
#         M = M.reshape(-1, 1)
#     elif M.ndim != 2:
#         raise ValueError('M must be a p x N matrix or a 1D column vector if N == 1.')


#     if U.ndim != 2:
#         raise ValueError("U must be a p x q matrix.")

#     # Dimension Checking
#     p1, N = M.shape
#     p2, q = U.shape
#     if p1 != p2:
#         raise ValueError("M and U must have the same number of spectral bands.")

#     # Least Squares Abundance Estimation
#     Minv = np.linalg.pinv(U)
#     W = np.zeros((q, N))
#     for n1 in range(N):
#         W[:, n1] = np.dot(Minv, M[:, n1])
#     # W =  np.linalg.lstsq(M,U,rcond=None)[0]
#     print(f"W.shape= {W.shape}")
#     W= W.T
#     return W[:,0]

# from scipy.linalg import pinv
# def hyperUcls(M, U):
#     Minv = pinv(U)
#     W = np.dot(Minv, M)
#     return W

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

    Minv = np.linalg.pinv(U)
    W = np.zeros((q, N))

    for n1 in range(N):
        W[:, n1] = Minv @ M[:, n1]

    return W