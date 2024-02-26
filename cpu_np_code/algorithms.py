import numpy as np

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