



def subset_img(data, subset=None):
    """"For subset the CRISM image into \n
    the regions of interest.\n
    Inputs: \n\
    ---------------------------------------------
    ``data``: The image data (m x n x p) where \n
    m: Samples, n: lines, p: bands \n
    ``subset``: {'xx': (xx_1 , xx_2), 'yy': (yy_1 , yy_2), 'zz': (zz_1 , zz_2)}
            'xx': (start_pixel_x , end_pixel_x) 
            'yy': (start_pixel_y , end_pixel_y)
            'zz': (start_band, end_band)
    Returns:
    ----------------------------------------------
    ``data``: the final subset image
    ----------------------------------------------"""
    if subset is not None:    
        if "xx" in subset.keys():
            (xx_1, xx_2)= subset["xx"]
            if "yy" in subset.keys():
                (yy_1,yy_2)= subset["yy"]   
                if "zz" in subset.keys():
                    (zz_1, zz_2) =subset["zz"]
                    data= data[yy_1:yy_2,xx_1:xx_2,zz_1:zz_2]
                else:
                    data= data[yy_1:yy_2,xx_1:xx_2,:]
            elif "zz" in subset.keys():
                (zz_1,zz_2) = subset["zz"]
                data= data[:,xx_1:xx_2,zz_1:zz_2]
            else:
                data= data[:,xx_1:xx_2,:]
        elif "yy" in subset.keys():
            (yy_1,yy_2)= subset["yy"]   
            if "zz" in subset.keys():
                (zz_1,zz_2)= subset["zz"]
                data= data[yy_1:yy_2,:,zz_1:zz_2]
            else:
                data= data[yy_1:yy_2,:,:]
        elif "zz" in subset.keys():
            (zz_1,zz_2)= data["zz"]
            data= data[:,:,zz_1:zz_2]
    else:
        data= data[:,:,:]
    return data

def contrast_stretch(data):
    """"For processing the CRISM image into \n
    a visible form, using contrast stretching."""
    # contrast stretching for image visibility with minimum noise:
    from skimage import exposure
    import numpy as np
    for ind, ch in enumerate (data[:,:,:].T):
    #masking no-data values
        ch_mask = ch != 65535.  
        #2nd and 98th percentile ignoring no-data values (if present)
        p2, p98 = np.nanpercentile(ch[ch_mask], (2, 98)) 
        data[:,:,ind] = exposure.rescale_intensity(ch, in_range=(p2, p98)).T
    return data


