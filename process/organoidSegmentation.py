import numpy as np
import pandas as pd
import skimage.filters as filters
from scipy import ndimage
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_objects
from skimage import measure, morphology, segmentation
from skimage.filters._median import median
from scipy.ndimage import gaussian_filter
from tifffile import imwrite
from skimage.transform import resize

def structuring_element(r1, r2, r3):
    """
    Constructs a 3D structuring element with three different radii.

    Parameters:
    ----------
    r1 : int
        The radius of the sphere along the x-axis.
    r2 : int
        The radius of the sphere along the y-axis.
    r3 : int
        The radius of the sphere along the z-axis.

    Returns:
    -------
    selem : ndarray of bools
        A 3D boolean array with the desired structuring element.
    """
    # Create a 3D meshgrid with the desired dimensions
    x, y, z = np.indices((r1*2+1, r2*2+1, r3*2+1))

    # Compute the distance from the center of the sphere
    center = np.array([r1, r2, r3])
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    # Create a boolean array with the sphere
    selem = distances <= max(r1, r2, r3)                                                                                                                                                                                                                                                                                                                                                             
    return selem    
                  
def DetectMainOrganoid(im3D, method, organoidParameter, keepLargestOrganoid, volumeMinOrganoid = None, resolution = None, bitsperpixel = None):
    if len(im3D.shape)==4:
        if method == "Otsu Threshold":
            return(DetectMainOrganoidRGB_CLAHE(im3D, organoidParameter, keepLargestOrganoid, volumeMinOrganoid, resolution, bitsperpixel)).astype('uint16')
        elif method == "Dynamic Range Threshold":
            return(DetectMainOrganoidRGB_removeBG(im3D, organoidParameter, keepLargestOrganoid, volumeMinOrganoid, resolution)).astype('uint16')
    else:
        if method == "Otsu Threshold":
            return(DetectMainOrganoidI_CLAHE(im3D, organoidParameter, keepLargestOrganoid, volumeMinOrganoid, resolution, bitsperpixel)).astype('uint16')
        elif method == "Dynamic Range Threshold":
            return(DetectMainOrganoidI_removeBG(im3D, organoidParameter, keepLargestOrganoid, volumeMinOrganoid, resolution)).astype('uint16')

def DetectMainOrganoidRGB_CLAHE(imRGB3D, organoidParameter, keepLargestOrganoid, volumeMinOrganoid, resolution, bitsperpixel):
    #parameters
    scale_otsu = organoidParameter #0.8
    sigma_pxl = round(6/resolution[1])
    min_sz_pxl = round(300/(resolution[1]*resolution[2]))
    max_val = pow(2, bitsperpixel)
    
    imgI = imRGB3D.mean(axis=-1)
    # imwrite("D:/Elise/images_figure/mean.tif", imgI.astype("uint16"))
    total_slices = imRGB3D.shape[0]
    
    organoidMask = imgI.copy()
    organoidMask = normalize_image(organoidMask, max_val)
    for i in range(total_slices):
        organoidMask[i]= fast_clahe(organoidMask[i], imgI[i])
    organoidMask = organoidMask.astype('uint16')

    organoidMask = gaussian_filter(organoidMask, (1, 1, sigma_pxl), truncate=True)
    organoidMask = gaussian_filter(organoidMask, (1, sigma_pxl, 1), truncate=True)
    
    thresh = filters.threshold_otsu(organoidMask, nbins=256) * scale_otsu
    mask = (organoidMask > thresh).astype(np.uint8)

    for z in range(total_slices):
        mask[z] = remove_small_particles(mask[z], min_sz_pxl)
    labeled = measure.label(mask, connectivity=1)
    counts = np.bincount(labeled.ravel())[1:]
    if keepLargestOrganoid:
        largest = np.zeros_like(mask)
        if labeled.max() > 0:
            largest_label = np.argmax(counts) + 1
            # largest = np.zeros_like(imgI)
            largest[labeled == largest_label] = 1
    else:
        volumeMinOrganoid = volumeMinOrganoid / (resolution[0] * resolution[1] * resolution[2])
        indices_labels = np.where(counts > volumeMinOrganoid)[0]
        labels = np.unique(labeled)[1:]
        mask_labels_large = np.isin(labeled, labels[indices_labels])
        largest = mask_labels_large * labeled
    # selem_dil = morphology.disk(sigma_pxl, decomposition='sequence')
    # erode_pxl = sigma_pxl+5
    # selem_er = morphology.disk(erode_pxl, decomposition='sequence')
    # for z in range(total_slices):
    #     # change by victor 20231201
    #     #mask[z] = morphology.dilation(largest[z], selem_dil)
    #     #mask[z] = morphology.remove_small_holes(mask[z])
    #     #mask[z] = morphology.erosion(mask[z], selem_er)
    #     largest_bool=largest[z].astype('bool')
    #     largest_bool=morphology.dilation(largest_bool, selem_dil)
    #     largest_bool=morphology.remove_small_holes(largest_bool)
    #     mask[z] = morphology.erosion(largest_bool, selem_er)
        
    # return mask

    return fast_organoid_post_process(largest, resolution[1])
    
    
def DetectMainOrganoidI_CLAHE(imI3D, organoidParameter, keepLargestOrganoid, volumeMinOrganoid, resolution, bitsperpixel):
    scale_otsu = organoidParameter #0.8
    sigma_pxl = round(6/resolution[1])
    min_sz_pxl = round(300/(resolution[1]*resolution[2]))
    max_val = pow(2, bitsperpixel)
    
    total_slices = imI3D.shape[0]
    
    organoidMask = imI3D.copy()
    organoidMask = normalize_image(organoidMask, max_val)
    for i in range(total_slices):
        organoidMask[i]= fast_clahe(organoidMask[i], imI3D[i])
    organoidMask = organoidMask.astype('uint16')

    # change by Victor 2023 10 05
    organoidMask = gaussian_filter(organoidMask, (1, 1, sigma_pxl), truncate=True)
    organoidMask = gaussian_filter(organoidMask, (1, sigma_pxl, 1), truncate=True)
    
    thresh = filters.threshold_otsu(organoidMask, nbins=256) * scale_otsu
    mask = (organoidMask > thresh).astype(np.uint8)

    for z in range(total_slices):
        mask[z] = remove_small_particles(mask[z], min_sz_pxl)
    labeled = measure.label(mask, connectivity=1)
    counts = np.bincount(labeled.ravel())[1:]
    if keepLargestOrganoid:
        largest_label = np.argmax(counts) + 1
        # largest = np.zeros_like(imI3D)
        largest = np.zeros_like(mask)
        largest[labeled == largest_label] = 1
    else:
        volumeMinOrganoid = volumeMinOrganoid / (resolution[0] * resolution[1] * resolution[2])
        indices_labels = np.where(counts > volumeMinOrganoid)[0]
        labels = np.unique(labeled)[1:]
        mask_labels_large = np.isin(labeled, labels[indices_labels])
        largest = mask_labels_large * labeled
    # selem_dil = morphology.disk(sigma_pxl, decomposition='sequence')
    # erode_pxl = sigma_pxl+5
    # selem_er = morphology.disk(erode_pxl, decomposition='sequence')
    # for z in range(total_slices):
    #     mask[z] = morphology.dilation(largest[z], selem_dil)
    #     mask[z] = morphology.remove_small_holes(mask[z])
    #     mask[z] = morphology.erosion(mask[z], selem_er)
    # return mask
    return fast_organoid_post_process(largest, resolution[1])
        
def DetectMainOrganoidRGB_removeBG(imRGB3D, organoidParameter, keepLargestOrganoid, volumeMinOrganoid, resolution):
    """
    Detect the main organoid in an image using otsu multi-thresholding

    Args:
        imRGB3D (np.ndarray): RGB stack of organoid Z X Y C

    Returns:
        np.ndarray: binary mask of organoid stack Z X Y
    """
    imgI = imRGB3D.mean(axis=-1)
    
    organoidMask = np.zeros(imgI.shape)
    imI3D_Blured = gaussian_filter(imgI,(0,4,4))
    #remove background
    hist, bins = np.histogram(imI3D_Blured.flatten(), bins=255)
    delta = (np.percentile(imI3D_Blured.flatten(), 99.9) - bins[np.argmax(hist)]) * organoidParameter #/10
    pic_intensity = bins[np.argmax(hist)] + delta
    
    for slide in range(imI3D_Blured.shape[0]):
        hist_z, bins_z = np.histogram(imI3D_Blured[slide].flatten(), bins=255)
        delta_z = (np.percentile(imI3D_Blured[slide].flatten(), 99.9) - bins_z[np.argmax(hist_z)])* organoidParameter #/10
        pic = np.maximum(bins_z[np.argmax(hist_z)] + delta_z, pic_intensity)
        organoidMask[slide, imI3D_Blured[slide] > pic] = 1

    organoidMask = ndimage.binary_fill_holes(organoidMask).astype(int)
    organoidLabel = label(organoidMask,connectivity=1)
    organoidLabel = MedianFilterPostProcessing(organoidLabel)
    # objAreas = pd.DataFrame(regionprops_table(organoidLabel,properties=["area","label"]))
    # mainObjectLabel = objAreas.sort_values("area",ascending=False).iloc[0]["label"]
    # organoidMask = organoidLabel == mainObjectLabel
    
    counts = np.bincount(organoidLabel.ravel())[1:]
    if keepLargestOrganoid:
        largest_label = np.argmax(counts) + 1
        organoidMask = np.zeros_like(imgI)
        organoidMask[organoidLabel == largest_label] = 1
    else:
        volumeMinOrganoid = volumeMinOrganoid / (resolution[0] * resolution[1] * resolution[2])
        indices_labels = np.where(counts > volumeMinOrganoid)[0]
        labels = np.unique(organoidLabel)[1:]
        mask_labels_large = np.isin(organoidLabel, labels[indices_labels])
        largest = mask_labels_large * organoidLabel
        
    for i in range(organoidMask.shape[0]):
        organoidMask[i] = remove_small_objects(ndimage.binary_fill_holes(organoidMask[i]),64)
    return organoidMask

def DetectMainOrganoidI_removeBG(imI3D, organoidParameter, keepLargestOrganoid, volumeMinOrganoid, resolution):
    """
    Detect the main organoid in an image using otsu multi-thresholding

    Args:
        imRGB3D (np.ndarray): nuclei intensity stack of organoid Z X Y C

    Returns:
        np.ndarray: binary mask of organoid stack Z X Y
    """
    organoidMask = np.zeros(imI3D.shape)
    imI3D_Blured = gaussian_filter(imI3D,(0,4,4))
    #remove background
    hist, bins = np.histogram(imI3D_Blured.flatten(), bins=255)
    delta = (np.percentile(imI3D_Blured.flatten(), 99.9) - bins[np.argmax(hist)])* organoidParameter #/10
    pic_intensity = bins[np.argmax(hist)] + delta
    
    for slide in range(imI3D_Blured.shape[0]):
        hist_z, bins_z = np.histogram(imI3D_Blured[slide].flatten(), bins=255)
        delta_z = (np.percentile(imI3D_Blured[slide].flatten(), 99.9) - bins_z[np.argmax(hist_z)])* organoidParameter #/10
        pic = np.maximum(bins_z[np.argmax(hist_z)] + delta_z, pic_intensity)
        organoidMask[slide, imI3D_Blured[slide] > pic] = 1
    
    organoidMask = ndimage.binary_fill_holes(organoidMask).astype(int)
    organoidLabel = label(organoidMask,connectivity=1)
    organoidLabel = MedianFilterPostProcessing(organoidLabel)
    # objAreas = pd.DataFrame(regionprops_table(organoidLabel,properties=["area","label"]))
    # mainObjectLabel = objAreas.sort_values("area",ascending=False).iloc[0]["label"]
    # organoidMask = organoidLabel == mainObjectLabel
    
    counts = np.bincount(organoidLabel.ravel())[1:]
    if keepLargestOrganoid:
        largest_label = np.argmax(counts) + 1
    
        organoidMask = np.zeros_like(imI3D)
        organoidMask[organoidLabel == largest_label] = 1
    else:
        volumeMinOrganoid = volumeMinOrganoid / (resolution[0] * resolution[1] * resolution[2])
        indices_labels = np.where(counts > volumeMinOrganoid)[0]
        labels = np.unique(organoidLabel)[1:]
        mask_labels_large = np.isin(organoidLabel, labels[indices_labels])
        largest = mask_labels_large * organoidLabel
        
    for i in range(organoidMask.shape[0]):
        organoidMask[i] = remove_small_objects(ndimage.binary_fill_holes(organoidMask[i]),64)
    return organoidMask

def fast_clahe(image, image16_bits, block_radius = 127, bins = 256, slope = 10, box_x_min = 0, box_y_min = 0, box_x_max = None, box_y_max = None): #256 pour bins
    
    def createHistogram(blockRadius: int, bins: int, blockXCenter: int, blockYCenter: int, src: np.ndarray):
        hist = np.zeros(bins + 1)
        
        xMin = int(max(0, blockXCenter - blockRadius))
        yMin = int(max(0, blockYCenter - blockRadius))
        
        xMax = int(min(src.shape[1], blockXCenter + blockRadius + 1))
        yMax = int(min(src.shape[0], blockYCenter + blockRadius + 1))
        
        cropped_img = src[yMin:yMax, xMin:xMax]
        hist, bins = np.histogram(cropped_img.flatten(), bins=256, range=(0,256))
        
        return hist
    
    block_radius = int((int(block_radius) - 1 ) / 2)
    bins = int(bins) - 1
    slope = float(slope)
    
    src = image.copy()
    ceiled_src = np.ceil(src/255.0 * bins).astype(int)
    image_out = image16_bits.copy()
    # Set default boxXMax and boxYMax if not provided
    if box_x_max is None:
        box_x_max = image.shape[1]
    if box_y_max is None:
        box_y_max = image.shape[0]
    h, w = image.shape[:2]
    src = (src[box_y_min:box_y_max, box_x_min:box_x_max])
    dst = np.zeros_like(src.flatten(), np.float32)
    
    block_size = int(2 * block_radius + 1)
    limit = int(slope * block_size**2 / bins + 0.5)
    nc = w // block_size
    nr = h // block_size
    cm = w - nc * block_size
    
    if cm == 0:
        cs = np.zeros(nc)
        for i in range(nc):
            cs[i] = int(i * block_size + block_radius + 1)
    elif cm == 1:
        cs = np.zeros(nc + 1)
        for i in range(nc):
            cs[i] = int(i * block_size + block_radius + 1)
        cs[nc] = int(w - block_radius - 1)
    else:
        cs = np.zeros(nc + 2)
        cs[0] = int(block_radius + 1)
        for i in range(nc):
            cs[i+1] = int(i * block_size + block_radius + 1 + cm/2)
        cs[nc+1] = int(w - block_radius - 1)
        
    rm = h - nr * block_size
    if rm == 0:
        rs = np.zeros(nr)
        for i in range(nr):
            rs[i] = int(i * block_size + block_radius + 1)
    elif rm == 1:
        rs = np.zeros(nr+1)
        for i in range(nr):
            rs[i] = int(i * block_size + block_radius + 1)
        rs[nr] = int(h - block_radius - 1)
    else:
        rs = np.zeros(nr+2)
        rs[0] = int(block_radius + 1)
        for i in range(nr):
            rs[i+1] = int(i * block_size + block_radius + 1 + rm/2)
        rs[nr+1] = int(h - block_radius - 1)
    
    for r in range(len(rs) + 1):
        r0 = max(0, r-1)
        r1 = min(len(rs)-1, r)
        dr = int(rs[r1] - rs[r0])
        hist = createHistogram(block_radius, bins, cs[0], rs[r0], src)
        tr = create_transfer(hist, limit)
        if r0 == r1:
            br = tr
        else:
            hist = createHistogram(block_radius, bins, cs[0], rs[r1], src)
            br = create_transfer(hist, limit)
        
        y_min = int(0 if r==0 else rs[r0])
        y_max = int(rs[r1] if r<len(rs) else h-1)
        for c in range(len(cs) + 1):
            c0 = max(0, c-1)
            c1 = min(len(cs)-1, c)
            dc = int(cs[ c1 ] - cs[ c0 ])
				
            tl = tr
            bl = br
            
            if c0 != c1 : 
                hist = createHistogram(block_radius, bins, cs[ c1 ], rs[ r0 ], src )
                tr = create_transfer(hist, limit)
                if r0 == r1 :
                    br = tr
                else : 
                    hist = createHistogram(block_radius, bins, cs[ c1 ], rs[ r1 ], src )
                    br = create_transfer(hist, limit) 
            x_min = int(0 if c == 0 else cs[c0])
            x_max = int(cs[c1] if c < len(cs) else w-1)

            Y = np.arange(y_min, y_max, 1)
            X = np.arange(x_min, x_max, 1)
            XX, YY = np.meshgrid(X, Y)
            ar_shape = XX.shape
            WY = ((- Y + rs[r1] ) / (dr if dr!=0 else 0.000000000000001)).astype(float)
            WY =  np.clip(WY ,-10000000000,10000000000)
            if dr==0: WY[WY==0]=np.nan
            
            WX = ((- XX + cs[c1]) / (dc if dc!=0 else 0.000000000000001)).astype(float)
            WX =  np.clip(WX ,-10000000000,10000000000)
            if dc==0: WX[WX==0]=np.nan
            O = (YY*w+XX).flatten()
            V = ceiled_src.flatten()[O]
            T00 = tl[V]
            T01 = tr[V]
            T10 = bl[V]
            T11 = br[V]
            if c0 == c1 :
                T0 = T00.astype(float)
                T1 = T10.astype(float)
            else :
                T0 = WX.flatten() * T00 + (1.0 - WX.flatten()) * T01
                T1 =  WX.flatten() * T10 + (1.0 -  WX.flatten()) * T11
            if r0 == r1 :	
                T = T0.astype(float)
            else:
                T = WY[:,np.newaxis]* T0.reshape(WX.shape) + ( 1.0 - WY[:,np.newaxis] )*T1.reshape(WX.shape)
            dst[O] = np.clip(np.ceil(T.flatten()*255),0,255)
            image_out = ShortApply(image_out, image16_bits, src, dst, box_x_min, 
                                   box_y_min, box_x_max, box_y_max,
                                    x_min, y_min, x_max, y_max)       
    return (image_out).astype('uint16')


def fast_organoid_post_process(organoid_label_mask:np.ndarray, xreolution:int):
    """Remove holes on organoids and smooth its surface using a downsampling algorithme for higher efficiency, performing similar erosion as done before,
     but adapting the size of the image to work with a 6 micron/3pixel radius erosion/dilation kernel

    Args:
        organoid_label_mask (np.ndarray): organoid mask to be filled
        xreolution (int): slice x resolution in um

    Returns:
        np.ndarray: organoid label mask filled
    """

    labels = np.unique(organoid_label_mask)[1:]
    mask_2_micron_per_XYpix = resize(organoid_label_mask,np.array(organoid_label_mask.shape)//[1,2/xreolution,2/xreolution],order=0,anti_aliasing=False)
    selem_low = morphology.disk(3) #6 micron radius disk on reduced size image
    selem_high = morphology.disk(5) #5 pixel radius disk on initial size image
    organoid_label_filled = np.zeros_like(organoid_label_mask)
    for z in range(len(organoid_label_mask)):
        small_dil = morphology.dilation(footprint=selem_low,image=mask_2_micron_per_XYpix[z])
        small_filled = np.zeros_like(small_dil)
        for l in labels:
            small_filled[morphology.remove_small_holes(small_dil==l)]=l
        small_er = morphology.erosion(footprint=selem_low, image=small_filled)
        resized_small = resize(output_shape=organoid_label_mask[z].shape, order=0, image=small_er,anti_aliasing=False)
        nohole = np.max(axis=0, a = [organoid_label_mask[z], resized_small])
        organoid_label_filled[z] = morphology.erosion(footprint = selem_high,image=nohole)
    return measure.label(organoid_label_filled>0, connectivity=1)


def ShortApply(image_out, image, src, dst, box_x_min, box_y_min, box_x_max, \
                box_y_max, cellXMin, cellYMin, cellXMax, cellYMax):
    xMin = np.maximum( box_x_min, cellXMin )
    yMin = np.maximum( box_y_min, cellYMin )
    xMax = np.minimum( box_x_max, cellXMax )
    yMax = np.minimum( box_y_max, cellYMax )
    
    cropped_image = image[yMin:yMax, xMin:xMax]
    cropped_src = src[yMin:yMax, xMin:xMax]
    cropped_dst = dst.reshape(image_out.shape)[yMin:yMax, xMin:xMax]
    
    min_val = 0
    a = np.where(cropped_src == 0, 1, cropped_dst / cropped_src)
    b = a * (cropped_image - min_val) + min_val
    result = np.clip(np.ceil(b), 0, 65535).astype('uint16')
    
    image_out[yMin:yMax, xMin:xMax] = result
    return image_out

def clip_histogram(hist, limit):
    clipped_hist = hist.copy()
    clipped_entries = 0
    clipped_entries_before = None
    
    while clipped_entries != clipped_entries_before:
        clipped_entries_before = clipped_entries
        clipped_entries = 0
        D = clipped_hist - limit
        Dsup = D>0
        clipped_entries = np.sum(D[Dsup])
        clipped_hist[Dsup] = limit

        d = clipped_entries // len(hist)
        m = clipped_entries % len(hist)

        clipped_hist+=d
        
        if m != 0:
            s = int((len(hist) - 1) / m)
            s_range = np.arange(int(s / 2), len(hist), s)
            clipped_hist[s_range] +=1
                
    return clipped_hist
    
def create_transfer(hist, limit):
    cdfs = clip_histogram(hist, limit)
    h_min = len(hist) - 1
    
    for i in range(h_min):
        if cdfs[i] != 0:
            h_min = i
            break
        
    cdfs = np.cumsum(cdfs)
    cdf = cdfs[-1]
    cdf_min = float(cdfs[h_min])
    cdf_max = float(cdfs[len(hist)-1])
    transfer = (cdfs.astype(float)-cdf_min) / (cdf_max - cdf_min)
    return transfer

def remove_small_particles(image, pixel_size_threshold):
    cleared = segmentation.clear_border(image > 0)
    labeled = measure.label(cleared)
    props = measure.regionprops(labeled, intensity_image=image)
    mask = np.zeros_like(image, dtype=bool)
    for prop in props:
        if prop.area < pixel_size_threshold:
            mask[prop.coords[:, 0], prop.coords[:, 1]] = True
    filtered = np.copy(image)
    filtered[mask | (labeled == 0)] = 0
    return filtered

def normalize_image(image, max_val):
    min_val = 0 
    scale = 256.0/(max_val-min_val+1)
    image_normalized = ((image * scale)+0.5).astype(int)
    image_normalized = np.clip(image_normalized, 0, 255)
    return image_normalized.astype('float32')

def MedianFilterPostProcessing(maskOrLabel,size=3,repeat=1):
    filteredMask = maskOrLabel.copy()
    for i in range(repeat):
        filteredMask =  median(filteredMask, SphereFootprint(size))
    return filteredMask

def SphereFootprint(radius=3, dtype=np.uint8):
    L = np.arange(-radius, radius + 1)
    X, Y, Z = np.meshgrid(L, L, L)
    return np.array((X ** 2 + Y ** 2 + Z**2) <= radius ** 2, dtype=dtype)

if __name__ == "__main__":
    from tifffile import imread, imwrite
    import matplotlib.pyplot as plt

    imRGB3D = imread("statics/data/RGB_test.tif") # RGB stack of organoid Z X Y C
    mainOrganoid = DetectMainOrganoidRGB_removeBG(imRGB3D)
    imwrite("statics/data/test_mainOrganoid_RGB.tif",mainOrganoid)
    imI3D = imread("statics/data/I_test.tif") # Intensity stack of organoid Z X Y C
    mainOrganoid = DetectMainOrganoidI_removeBG(imRGB3D[...,2])
    imwrite("statics/data/test_mainOrganoid_I.tif",mainOrganoid)
    mainOrganoid = DetectMainOrganoidI_removeBG(imI3D)
    imwrite("statics/data/test_mainOrganoid_I2.tif",mainOrganoid)
    print("Done")
