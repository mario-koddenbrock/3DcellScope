import numpy as np
from skimage.morphology import erosion, dilation
from skimage.measure import regionprops
from PIL import ImageColor
import random
import cv2
from csbdeep.utils import normalize
from matplotlib.cm import gist_rainbow
from tifffile import imwrite

def BuildCellMontage(nucleiLabelMap, cellLabelMap, imRGB, imCell3D):
    #mask_color = createColorMask(cellLabelMap)
    # rgb_cell = np.repeat(cellLabelMap[..., np.newaxis], 3, axis=-1).astype('uint8')
    #montage = np.zeros(mask_color.shape).astype('uint8')
    # for z in range(montage.shape[0]):
    #     montage[z] = cv2.addWeighted(cv2.cvtColor(rgb_cell[z], cv2.COLOR_RGB2BGR),1.,cv2.cvtColor(mask_color[z], cv2.COLOR_RGB2BGR),0.2,0)

    # if np.max(imRGB)>255:
    #     imRGB = imRGB.copy()
    #     for i in range(imRGB.shape[-1]):
    #         imRGB[...,i]=normalize(imRGB[...,i], 0, 100, axis=(0,1,2))*255
    #     imRGB = imRGB.astype('uint8')
    # montage_cell = montage.astype('uint8')
    contours_cell = findContours(cellLabelMap)
    contours_nuclei = findContours(nucleiLabelMap)
    montage_cell = createMontage(imCell3D, nucleiLabelMap, contours_cell)
    montage_temp = createMontage_col(normalize(imRGB) if np.max(imRGB)>255 else imRGB, nucleiLabelMap, contours_nuclei, (209, 0, 164))
    montage_nuclei_cell = createMontage_col(montage_temp, cellLabelMap, contours_cell, (255, 234, 1))
    return montage_cell, montage_nuclei_cell

def get_random_colors(n, RGB_out=True):
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n)]
    if RGB_out:
        colors = [ImageColor.getcolor(c, 'RGB') for c in colors]
    return colors

def get_random_colors2(n, RGB_out=True,col_seed:int = None,):
    rng = np.random.default_rng(col_seed) if col_seed is not None else np.random.default_rng()
    return [(np.array(list(gist_rainbow(c[0])[:-1]))*255*(0.5+c[1:]*0.5)).astype(int) for c in rng.random((n,4))]
#create an image with the same shape than mask which only contains the contours

def findContours(mask):
    contours = np.zeros(mask.shape)
    for z in range(len(mask)):
        contours[z] = ((mask[z] != erosion(mask[z])) | (mask[z] != dilation(mask[z])))*mask[z]
    return contours

#montage which overlap the image and the mask's contours
def createMontage(im, mask, contours): 
    montage4D = im.copy()
    montage4D = normalize_data(montage4D)
    montage4D = np.repeat(montage4D[..., np.newaxis], 3, axis=-1)

    regions = regionprops(mask.astype(int))
    colors = get_random_colors(len(regions)+1)
    for col, el in zip(colors, regions):
        lab = el["label"]
        slice_ = el["slice"]
        rep = montage4D[slice_]
        rep2 = contours[slice_]
        rep[rep2==lab] = col
        montage4D[slice_]= rep
    return montage4D.astype('uint8')

def createContourOverlay(mask,col_seed:int=None, contourColor = None):
    contourOverlay = np.zeros(mask.shape + (4,))
    contours = findContours(mask)
    regions = regionprops(mask.astype(int))
    colors = get_random_colors2(len(regions),col_seed=col_seed)
    for col, el in zip(colors,regions):
        lab = el["label"]
        slice_ = el["slice"]
        contourOverlay[slice_][mask[slice_]==lab] = tuple(col)+(26,)
        contourOverlay[slice_][contours[slice_]==lab] = tuple(col if contourColor is None else contourColor ) +(255,)
    return contourOverlay

def createOverlay(mask,col_seed:int=None):
    contourOverlay = np.zeros(mask.shape+(4,))
    regions = regionprops(mask.astype(int))
    colors = get_random_colors2(len(regions),col_seed=col_seed)
    for col, el in zip(colors,regions):
        lab = el["label"]
        slice_ = el["slice"]
        contourOverlay[slice_][mask[slice_]==lab] = tuple(col)+(50,)
    return contourOverlay

def createMontageTransparency(img, overlay):
    # normalize A channel from 0-255 to 0-1
    a_overlay = overlay[:,:,:,3] / 255.0
    if len(img.shape)<=3:
        img = np.repeat(img[..., np.newaxis], 3, axis=-1)
    # set adjusted colors
    for color in range(0, 3):
        img[:,:,:,color] = (a_overlay * overlay[:,:,:,color] + \
            img[:,:,:,color] * (1 - a_overlay))
    return img.astype('uint8')

def normalize_data(im_input):
    im_out = np.maximum(im_input, 0, im_input - np.percentile(im_input, 25))
    maxi = np.percentile(im_out, 99.9)
    im_out = np.minimum(np.maximum(im_out / maxi * 255, 0), 255)
    return im_out 

def createColorMask(mask): 
    res = np.zeros(mask.shape)
    res = np.repeat(res[..., np.newaxis], 3, axis=-1)

    regions = regionprops(mask.astype(int))
    colors = get_random_colors(len(regions)+1)
    for col, el in zip(colors, regions):
        lab = el["label"]
        slice_ = el["slice"]
        rep = res[slice_]
        rep2 = mask[slice_] 
        rep[rep2==lab] = col
        res[slice_]= rep
    return res.astype('uint8')

def createMontage_col(im, mask, contours, col): 
    montage4D = im.copy()

    regions = regionprops(mask.astype(int))
    colors = [col for i in range(len(regions)+1)]
    for col, el in zip(colors, regions):
        lab = el["label"]
        slice_ = el["slice"]
        rep = montage4D[slice_]
        rep2 = contours[slice_]
        rep[rep2==lab] = col
        montage4D[slice_]= rep
    return montage4D.astype('uint8')

def create_composite(cell_channel, nuclei_mask, red_channel = None):
    if red_channel is None:
        red_channel = np.zeros(cell_channel.shape)
    composite = np.stack((red_channel, cell_channel, nuclei_mask))
    composite = np.moveaxis(composite, 0, 3) 

    return composite.astype('uint16')

def write_composite(dict_composite, path):
    metadata = {'axes': 'ZCYX', 'mode': 'composite'}
    ranges_list = []
    nb_channel = len(dict_composite)
    comp = np.zeros((nb_channel,) + dict_composite['C1'].shape)
    dict_composite['nuclei_mask'] = findContours(dict_composite['nuclei_mask'])
    if 'cell_mask' in dict_composite:
        dict_composite['cell_mask'] = findContours(dict_composite['cell_mask'])
    if 'organoid_mask' in dict_composite:
        dict_composite['organoid_mask'] = findContours(dict_composite['organoid_mask'])
    for i, ch in enumerate(dict_composite.keys()):
        comp[i] = dict_composite[ch]
        ranges_list.append(float(dict_composite[ch].min()))
        ranges_list.append(float(dict_composite[ch].max()))
    metadata['Ranges'] = tuple(ranges_list)
    comp = np.moveaxis(comp, 0, 1) 
    imwrite(path, comp.astype(np.uint16), imagej=True,
        photometric='MINISBLACK', metadata=metadata) 
    return

def imQuantilNorm(im3DI):
    im3DI = im3DI - np.quantile(im3DI,0.01)
    im3DI = im3DI/np.quantile(im3DI,0.99)*255
    im3DI[im3DI>255] = 255
    return im3DI.astype(np.uint8)