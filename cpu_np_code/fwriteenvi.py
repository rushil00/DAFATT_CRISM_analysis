import numpy as np
import os

def fwriteenvi(filename, data, wavelength, bandname, wavelengthunit=2):
    # Check user input
    if not isinstance(filename, str):
        raise ValueError('filename should be a char string')
    
    m, n, d = data.shape
    if m == 0 or n == 0 or data.size == 0:
        raise ValueError('data is empty')

    # check wavelength unit
    if wavelengthunit is not None and (wavelengthunit > 4 or wavelengthunit < 1):
        raise ValueError('wavelength unit is not correct')

    pathstr, name, ext = os.path.splitext(filename)

    if len(pathstr) == 0:
        enviheadfile = name + '.hdr'
    else:
        enviheadfile = os.path.join(pathstr, name + '.hdr')

    with open(enviheadfile, 'w+') as fidhead:
        if fidhead < 0:
            raise IOError('open head file error')

        datatype = ['bit8', 'uint8', 'int16', 'int32', 'float32', 'float64', 'uint16', 'uint32', 'int64', 'uint64', 'double']
        datatypeid = ['1', '1', '2', '3', '4', '5', '12', '13', '14', '15', '5']
        interleave = ['bsq', 'bil', 'bip']
        unit = ['Nanometers', 'Micrometers', 'Wavenumber', 'Unknown']

        pos = datatype.index(data.dtype.name)
        if pos is None:
            raise ValueError('unsupported data type')

        fidhead.write('%s\n%s\n' % ('ENVI', 'description = {'))
        fidhead.write('%s\n' % ['    Create New File Result [' + str(np.fix(np.clock())) + ']}'])
        fidhead.write('samples = %d\nlines = %d\nbands   = %d\n' % (n, m, d))
        fidhead.write('%s\n%s\n' % ('header offset = 0', 'file type = ENVI Standard'))
        fidhead.write('data type = %s\ninterleave = %s\n' % (datatypeid[pos], interleave[0]))
        fidhead.write('sensor type = Unknown\nbyte order = 0\nwavelength units = %s\n' % unit[wavelengthunit])

        if wavelength is not None:
            if wavelength.shape[0] == 1:
                wavelength = wavelength.reshape(-1, 1)

            fidhead.write('wavelength = {\n')
            for i in range(wavelength.shape[0] - 1):
                fidhead.write('%11.6f, ' % wavelength[i, 0])
                if i % 6 == 0:
                    fidhead.write('\n')

            fidhead.write('%11.6f}\n' % wavelength[-1, 0])

            if wavelength.shape[1] == 2:
                fidhead.write('fwhm = {\n')
                for i in range(wavelength.shape[0] - 1):
                    fidhead.write('%11.6f, ' % wavelength[i, 1])
                    if i % 6 == 0:
                        fidhead.write('\n')
                fidhead.write('%11.6f}\n' % wavelength[-1, 1])

        if bandname is not None:
            fidhead.write('\nband names= {\n')
            for i in range(len(bandname) - 1):
                fidhead.write('%s,\n' % bandname[i])
            fidhead.write('%s}\n' % bandname[-1])

    # write to file
    with open(filename, 'wb') as fid:
        if fid < 0:
            raise IOError('can not open file to write data')

        for i in range(d):
            np.ndarray.tofile(fid, data[:, :, i].T, sep='', format=data.dtype)

    # Status
    status = 1

    return status
