import numpy as np

def freadenvi(fname):
    # Parameters initialization
    elements = ['samples', 'lines', 'bands', 'data type', 'interleave']
    d = ['uchar', 'int16', 'int32', 'float32', 'float64', 'uint16', 'uint32', 'int64', 'uint64']
    interleave = ['bsq', 'bil', 'bip']

    # Check user input
    if not isinstance(fname, str):
        raise ValueError('fname should be a char string')

    # Generate header file name from data file name
    suffixidx = fname.rfind('.')
    if suffixidx > 0:
        headerfile = fname[:suffixidx] + '.hdr'
    else:
        headerfile = fname + '.hdr'

    # Open ENVI header file to retrieve s, l, b & d variables
    with open(headerfile, 'r') as rfid:
        # Read ENVI image header file and get p(1) : nb lines,
        # p(2) : nb samples, p(3) : nb bands, t : data type and b: interleave
        p = [0, 0, 0]
        t = ''
        b = ''
        for tline in rfid:
            first, second = tline.split('=')
            first = first.strip()

            if first == elements[0]:
                p[0] = int(second.split()[1])
            elif first == elements[1]:
                p[1] = int(second.split()[1])
            elif first == elements[2]:
                p[2] = int(second.split()[1])
            elif first == elements[3]:
                t = d[int(second)]
            elif first == elements[4]:
                b = second.strip()

    # Open the ENVI image and store it in the 'image' NumPy array
    print('Opening {} lines x {} cols x {} bands'.format(p[1], p[0], p[2]))
    print('of type {} image...'.format(t))
    with open(fname, 'rb') as fid:
        image = np.fromfile(fid, dtype=t)

    # Reshape the image based on interleave
    if b == interleave[0]:
        # bsq
        image = image.reshape((p[1], p[2], p[0]))
        image = np.transpose(image, (0, 2, 1))
    elif b == interleave[1]:
        # bil
        image = image.reshape((p[2], p[0], p[1]))
        image = np.transpose(image, (2, 0, 1))
    elif b == interleave[2]:
        # bip
        image = image.reshape((p[2], p[0], p[1]))
    else:
        raise ValueError('Unknown image data interleave')

    return image, p, t, b

# Example usage:
# image, p, t, b = freadenvi('your_data_file.dat')
