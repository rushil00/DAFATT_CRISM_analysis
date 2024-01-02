import numpy as np
import scipy as sp
def ATGP(data, q):
    """
    Automatic Target Generation Process endmembers induction algorithm

    Parameters:
        data: `numpy array`
            2d matrix of HSI data ((m x n) x p)

        q: `int`
            Number of endmembers to be induced (positive integer > 0)

    Returns: `tuple: numpy array, numpy array`
        * Set of induced endmembers (N x p).
        * Induced endmembers indexes vector.

    References:
      A. Plaza, C.-I. Chang, "Impact of Initialization on Design of Endmember
      Extraction Algorithms", Geoscience and Remote Sensing, IEEE Transactions on,
      vol. 44, no. 11, pgs. 3397-3407, 2006.
    """
    nsamples, nvariables = data.shape

    # Algorithm initialization
    # the sample with max energy is selected as the initial endmember
    max_energy = -1
    idx = 0
    for i in range(nsamples):
        r = data[i]
        val = np.dot(r, r)
        if val > max_energy:
          max_energy = val
          idx = i

    # Initialization of the set of endmembers and the endmembers index vector
    E = np.zeros((q, nvariables), dtype=np.float32)
    E[0] = data[idx] # the first endmember selected
    # Generate the identity matrix.
    I = np.eye(nvariables)
    IDX = np.zeros(q, dtype=np.int)

    IDX[0] = idx

    for i in range(q-1):
        UC = E[0:i+1]
        # Calculate the orthogonal projection with respect to the pixels at present chosen.
        # This part can be replaced with any other distance
        PU = I - np.dot(UC.T,np.dot(sp.linalg.pinv(np.dot(UC,UC.T)),UC))
        max_energy = -1
        idx = 0
        # Calculate the most different pixel from the already selected ones according to
        # the orthogonal projection (or any other distance selected)
        for j in range(nsamples):
            r = data[j]
            result = np.dot(PU, r)
            val = np.dot(result.T, result)
            if val > max_energy:
                max_energy = val
                idx = j
    # The next chosen pixel is the most different from the already chosen ones
        E[i+1] = data[idx]
        IDX[i+1] = idx

    return E, IDX