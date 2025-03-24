from scipy.ndimage import gaussian_filter
from skimage.filters._unsharp_mask import unsharp_mask
from skimage.exposure import match_histograms
import numpy as np
from .fileManager import *

def has_numbers(inputString): #determines if a string contains a number and is not empty
    num=all(char.isdigit() or '.' for char in inputString)
    not_empty=(inputString!='')
    return num&not_empty

def normalize_image(image, max_val):
    min_val = 0 
    scale = 256.0/(max_val-min_val+1)
    image_normalized = ((image * scale)+0.5).astype(int)
    image_normalized = np.clip(image_normalized, 0, 255)
    return image_normalized.astype('float32')

def updatePreprocessParameters(window1):
    method = window1["preproc_method"].get()
    if method == "gaussian difference":
        window1["-P1Text-"].update(value = "sigma")
        window1["-P1Text-"].update(visible = True)
        window1["-P2Text-"].update(visible = False)
        window1["-P1-"].update(visible = True)
        window1["-P2-"].update(visible = False)
    elif method == "gaussian division":
        window1["-P1Text-"].update(value = "sigma")
        window1["-P1Text-"].update(visible = True)
        window1["-P2Text-"].update(visible = False)
        window1["-P1-"].update(visible = True)
        window1["-P2-"].update(visible = False)
    elif method == "unsharp mask":
        window1["-P1Text-"].update(value = "radius")
        window1["-P1Text-"].update(visible = True)
        window1["-P1-"].update(visible = True)
        window1["-P2Text-"].update(value = "amount")
        window1["-P2Text-"].update(visible = True)
        window1["-P2-"].update(visible = True)   
    elif method == "correct bleaching":
        window1["-P1Text-"].update(visible = False)
        window1["-P2Text-"].update(visible = False)
        window1["-P1-"].update(visible = False)
        window1["-P2-"].update(visible = False)       
    return 

def addPreproc(window1, dict_preproc, list_preproc):
    channel = window1["preproc_channel"].get()
    method = window1["preproc_method"].get()
    if method == "gaussian difference":
        if has_numbers(window1["-P1-"].get())==True:
            sigma = int(window1["-P1-"].get()) 
            if channel in dict_preproc.keys():
                dict_preproc[channel].append(gaussianDifference(sigma))
            else:
                dict_preproc[channel] = [gaussianDifference(sigma)]
            list_preproc.append("C"+ channel + " " + method + " sigma " + str(sigma))
    elif method == "gaussian division":
        if has_numbers(window1["-P1-"].get())==True:
            sigma = int(window1["-P1-"].get()) 
            if channel in dict_preproc.keys():
                dict_preproc[channel].append(gaussianDivision(sigma))
            else:
                dict_preproc[channel] = [gaussianDifference(sigma)]
            list_preproc.append("C"+ channel + " " + method + " sigma " + str(sigma))
    elif method == "unsharp mask":
        if (has_numbers(window1["-P1-"].get())==True) and (has_numbers(window1["-P2-"].get())==True):
            radius = int(window1["-P1-"].get()) 
            amount = float(window1["-P2-"].get())
            if channel in dict_preproc.keys():
                dict_preproc[channel].append(unsharpMask(radius, amount))
            else:
                dict_preproc[channel] = [unsharpMask(radius, amount)]
            list_preproc.append("C"+ channel + " " + method + " radius " + str(radius)+ " amount " + str(amount))
    elif method == "correct bleaching":
        if channel in dict_preproc.keys():
            dict_preproc[channel].append(correctBleaching())
        else:
            dict_preproc[channel] = [correctBleaching()]
        list_preproc.append("C"+ channel + " " + method)
    window1["-PREPROC LIST"].update(values = list_preproc)
    return dict_preproc, list_preproc

def removePreproc(window1, dict_preproc, list_preproc):
    try:
        prepToRemove = window1["-PREPROC LIST"].get()[0]
        indexToRemove = window1["-PREPROC LIST"].get_indexes()[0] #list_preproc.index(prepToRemove)
        
        count = len([element for index, element in enumerate(list_preproc[:indexToRemove]) if not element.startswith(prepToRemove[0:2])])
        indexInDict = indexToRemove - count
        dict_preproc[prepToRemove[1]].pop(indexInDict)
        
        list_preproc.pop(indexToRemove)
        
        window1["-PREPROC LIST"].update(values = list_preproc)
    except:
        dict_preproc, list_preproc = dict_preproc, list_preproc
    return dict_preproc, list_preproc

def get_preproc_name(dict_preproc):
    name = ""
    channels = list(dict_preproc.keys())
    for c in channels:
        name = name +("__" if len(name)>0 else "")+ "C"+ c + "_"
        preprocesses = dict_preproc[c]
        for process in preprocesses:
            name = name + str(process) + "_"
        name = name[:-1]
    return name

def apply_preproc(dict_preproc, img, imInfo):
    if len(dict_preproc)>0:
        if imInfo.channel<=1:
            im = img.copy()
            preprocesses = dict_preproc["1"]
            for process in preprocesses:
                im = process(im)
        elif imInfo.channel>1:
            im = img.copy()
            channels = list(dict_preproc.keys())
            for c in channels:
                preprocesses = dict_preproc[c]
                for process in preprocesses:
                    im[:,:,:,int(c)-1] = process(im[:,:,:,int(c)-1])
    else:
        return img
    return im

class PreProcess:
    def __init__(self,func,name,**kwargs):
        self.func = func
        self.name = name
        self.parameters = kwargs
    def __call__(self,im):
        return(self.func(im))
    def __str__(self) -> str:
        return  "-".join([self.name]+["%s%s"%(str(k),str(v)) for k,v in self.parameters.items()])
    
    
def removeBackground(thresh):
    def preprocessFunc(im):
        im = im-thresh
        im[im<0] = 0
        return im
    return PreProcess(preprocessFunc,"removeBackground",thresh=thresh)

def addBackground(thresh):
    def preprocessFunc(im):
        im = im+thresh
        return im
    return  PreProcess(preprocessFunc,"addBackground",thresh=thresh)

def correctBleaching():
    def preprocessFunc(im):
        z = im.shape[0]
        percentiles_99 = np.zeros(z)
        for slide in range(z):
            percentiles_99[slide] = np.percentile(im[slide].flatten(), 99)
        pic_99 = np.argmax(percentiles_99)
        img_ref = im[pic_99]
        img_corrected = im.copy()
        for slice in range(pic_99, z):
            img_corrected[slice] = match_histograms(im[slice], img_ref)
        return img_corrected
    return  PreProcess(preprocessFunc,"correctBleaching")

def gaussianDifference(sigma):
    def preprocessFunc(im):
        im_blur = np.float32(gaussian_filter(im, (0, sigma,sigma))) #(z,y,x) 
        im_prep = np.float32(im) - im_blur
        return imToDisplay(im_prep)
    return PreProcess(preprocessFunc,"gaussianDifference",sigma=sigma)

def gaussianDivision(sigma):
    def preprocessFunc(im):
        im_blur = np.float32(gaussian_filter(im, (0, sigma,sigma))) #(z,y,x) 
        im_prep = np.divide(np.float32(im), im_blur, out=np.zeros_like(im).astype('float32'), where=im_blur != 0)
        return imToDisplay(im_prep)
    return PreProcess(preprocessFunc,"gaussianDivision",sigma=sigma)

def unsharpMask(radius, amount):
    def preprocessFunc(im):
        im_prep = unsharp_mask(im, radius=radius, amount=amount) #np.float32(im)
        return imToDisplay(im_prep)
    return PreProcess(preprocessFunc,"unsharpMask",radius=radius, amount=amount)

# def Clahe(bitsperpixel):
#     def preprocessFunc(im):
#         max_val = pow(2, bitsperpixel)
        
#         total_slices = im.shape[0]
        
#         im_prep = im.copy()
#         im_prep = normalize_image(im_prep, max_val)
#         for i in range(total_slices):
#             im_prep[i]= fast_clahe(im_prep[i], im[i])

