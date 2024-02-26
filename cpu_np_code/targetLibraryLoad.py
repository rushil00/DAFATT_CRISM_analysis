import scipy.io

def loadTargetLibrary(fname):
    # Load Target Library
    mat_contents = scipy.io.loadmat(fname)
    TargetLibrary = mat_contents['TargetLibrary']
    TargetLibraryName = mat_contents['TargetLibraryName']
    TargetLibraryFileName = mat_contents['TargetLibraryFileName']

    # Extract Data from Loaded Target Library
    TargetLibraryRef = TargetLibrary[104:, 1:]
    TargetLibraryName = TargetLibraryName[1:].flatten()
    TargetLibraryFileName = TargetLibraryFileName[1:].flatten()

    # Set Variable 'n'
    n = TargetLibraryRef.shape[1]

    # Extract Wavelength Information
    wave = TargetLibrary[104:, 0].flatten()
    return TargetLibrary, TargetLibraryName, TargetLibraryFileName, TargetLibraryRef, wave