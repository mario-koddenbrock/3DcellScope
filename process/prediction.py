import os
import json
import importlib
import sys

# gpu_var = sg.popup_yes_no("Use GPU ? ", icon='logo_gui.ico')
# if gpu_var == 'No':
#     print("Not using GPU")
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# elif gpu_var == 'Yes':
#     print("Using GPU")
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# import torch
import tensorflow as tf
print('Tensorflow gpu_device_name: ' + tf.test.gpu_device_name())
tf.autograph.experimental.do_not_convert
import matplotlib.pyplot as plt
plt.set_cmap('gray')
import matplotlib
import numpy as np
matplotlib.rcParams["image.interpolation"] = "none"
from glob import glob
from tifffile import imread
from csbdeep.utils import normalize
from .stardist_elise import random_label_cmap
from skimage.morphology import erosion, dilation
from .stardist_elise.models import Config3D, StarDist3D, StarDistData3D, StarDist2D
from tifffile import imread, imwrite
from scipy.ndimage import gaussian_filter
from skimage.filters._unsharp_mask import unsharp_mask
import matplotlib.pyplot as plt
import random as rd
from skimage.measure import regionprops, label
from skimage import util
import pandas as pd
import base64
from PIL import ImageColor, Image
import os.path
import random
import imageio as io
from tqdm import tqdm
from skimage.transform import resize
from .StatisticsForGui import infoFusion, getFeatures, addColumns
from pathlib import Path
from .cellSegmentation import maskAndReLabel, SegWatershed
from .organoidSegmentation import DetectMainOrganoid
from .fileManager import *
from .montage import createContourOverlay, createMontageTransparency, write_composite
import tempfile
from tensorflow.python.keras.saving import hdf5_format
from .preprocessing import *
import h5py
import shutil
import io as IO


#create a list with n colors in RGB format if RGB_out is true
def get_random_colors(n, RGB_out=True):
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n)]
    if RGB_out:
        colors = [ImageColor.getcolor(c, 'RGB') for c in colors]
    return colors

def normalize_data(im_input):
    im_out = np.maximum(im_input, 0, im_input - np.percentile(im_input, 25))
    maxi = np.percentile(im_out, 99.9)
    im_out = np.minimum(np.maximum(im_out / maxi * 255, 0), 255)
    return im_out 

#loadModel
def loadModelIO(defaultModel):
    
    if Path(defaultModel).exists() and Path(defaultModel).is_dir():
        model_path = Path(defaultModel)
    else:
        modelFolderPositionBasedOnCurrentFilePosition = str(Path(__file__).absolute().parent.parent)
        model_path = Path(modelFolderPositionBasedOnCurrentFilePosition)/"models"/defaultModel 

    configPath = model_path / "config.json"
    model_obj = StarDist2D if '2D' in defaultModel else StarDist3D
    try:
        model = model_obj(None, name=configPath.parent.name, basedir=os.path.abspath(configPath.parent.parent))
    except:
        return None
    if (model_path / "weights_best.h5").exists() or (model_path / "weights_last.h5").exists():
        return model
    else:
        return None

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


def remove_small_areas(mask, area_threshold):
    labeled = label(mask)
    props = regionprops(labeled)
    mask_out = labeled.copy()
    labels_to_remove = [prop.label for prop in props if prop.area < area_threshold]
    mask_out[np.isin(mask_out, labels_to_remove)] = 0
    return label(mask_out)    


def remove_large_areas(mask, area_threshold):
    labeled = label(mask)
    props = regionprops(labeled)
    mask_out = labeled.copy()
    labels_to_remove = [prop.label for prop in props if prop.area > area_threshold]
    mask_out[np.isin(mask_out, labels_to_remove)] = 0
    return label(mask_out)    


#uses the Stardist predict_instances function to predict the label of a mask, also returns the montage and volumes before nms
def predict_label(model,list_im, im_i,im_shape, masks = None, prob = None, nms = None, volume_min = None, volume_max = None, file_out=None, show_dist=True, results=None):
    n_channel = 1 if list_im[0].ndim == 3 else list_im[0].shape[-1]
    axis_norm = (0,1,2)   # normalize channels independently
    img = normalize(list_im[im_i], 1,99.8, axis=axis_norm)
    maskMainOrganoid = masks[im_i] if masks is not None else None #np.ones(im_shape).astype('uint8')
    if file_out is None or not os.path.exists(file_out):
        labels, details, volume, results = model.predict_instances(img, prob_thresh= prob, nms_thresh = nms, volume_min = volume_min, volume_max = volume_max, results=results)  
    else:
        labels = imread(file_out)
    mask_shape = im_shape[1:-1] if len(im_shape) > 3 else im_shape 
    labels = resize(labels.astype('int16'), mask_shape, order=0, anti_aliasing=False)
    if model.config.n_dim == 2:
        if volume_min is not None:
            labels = remove_small_areas(labels, volume_min)
        if volume_max is not None:
            labels = remove_large_areas(labels, volume_max)
        volume = np.array([p.area for p in regionprops(labels)])
    img = resize(img, im_shape, order=1)
    if len(labels.squeeze().shape) == 2:
        labels, maskMainOrganoid = maskAndReLabel(labels.squeeze()[np.newaxis], maskMainOrganoid)
    else:
        labels, maskMainOrganoid = maskAndReLabel(labels, maskMainOrganoid)
    return (img, labels, maskMainOrganoid, volume, results)

#uses the previous function to predict labels, montage, volumes for 1 image using the path
#also writes the results in the original folder
def predict_images(link, factor_resize_X, factor_resize_Y, factor_resize_Z, modelName, modelPath, result_folder_name, prob_thresh = None, nms_thresh = None, channel = None, organoChannel = None, organoMethod = None, organoidParameter = None, volume_min = None, volume_max = None, results=None, prep = None): #channel = 0 (red), 1 (green), 2 (blue)

    imInfo = ImInfo(link)
    assert imInfo.IsHandeled(), "unHandeled Image"

    im = imInfo.LoadImage()
    result_folder = imInfo.MainPath().parent / result_folder_name
    result_folder.mkdir(exist_ok=True)

    name_orig = imInfo.name


    if prep is not None:
        im_prep = prep if imInfo.channel<1 else prep[:,:,:,channel]
        im_prep_resize= resize(im_prep, (int(round(im_prep.shape[0] * factor_resize_Z)), int(round(im_prep.shape[1] * factor_resize_Y)),int(round(im_prep.shape[2] * factor_resize_X))),anti_aliasing=True, order = 1)
        X_prep = [im_prep_resize]
    #cut pictures and keep slected channel if RGB picture
    if imInfo.channel > 0:
        im_channel=im[:,:,:,channel]
        
    else:
        im_channel=im
    if organoChannel is None:
        organoidMask = None
        masks = None
    elif organoChannel == "all":
        organoidMask = DetectMainOrganoid(im, organoMethod, organoidParameter, imInfo.resolution, imInfo.bitsPerPixel) if prep is None else DetectMainOrganoid(prep, organoMethod, organoidParameter, imInfo.resolution, imInfo.bitsPerPixel)
        masks = [organoidMask]
    else:
        organoidMask = DetectMainOrganoid(im[:,:,:,organoChannel], organoMethod, organoidParameter, imInfo.resolution, imInfo.bitsPerPixel) if prep is None else DetectMainOrganoid(prep[:,:,:,organoChannel], organoMethod, organoidParameter, imInfo.resolution, imInfo.bitsPerPixel)
        masks = [organoidMask]
    im_shape = im_channel.shape
    #resize pictures 
    img_resize= resize(im_channel, (int(round(im_channel.shape[0] * factor_resize_Z)), int(round(im_channel.shape[1] * factor_resize_Y)),int(round(im_channel.shape[2] * factor_resize_X))),anti_aliasing=True, order = 1)
    X = [img_resize]
    img_resize=[]
    
    if modelName is not None and modelPath is not None:
        model = StarDist3D(None, name = modelName, basedir= modelPath.replace(os.sep, '/'))
    else:
        model = loadModelIO()
        print("using internal model")
    predict_label_args = {
        "masks":masks,
        "im_shape" : im_shape, "prob" : prob_thresh, "nms" : nms_thresh, "volume_min" : volume_min,
        "volume_max" : volume_max, "results":results}
    if prep is not None:
        print("image size : ", im_channel.shape)
        img, labels, masks, volume, results = predict_label(model, X_prep, 0,**predict_label_args)
        preprocessed_image = Path(result_folder) / ( name_orig+'_preprocessed-image.tif')
        imwrite(preprocessed_image, im_prep, photometric='minisblack')
        img = im_channel
    else:
        print("image size : ", im_channel.shape)
        img, labels, masks, volume, results = predict_label(model, X, 0,**predict_label_args)
    
    image_file = Path(result_folder) / ( name_orig+'_nuclei-image.tif')
    imwrite(image_file, im_channel, photometric='minisblack')
    mask_file = Path(result_folder) / ( name_orig+'_nuclei-mask.tif')
    labels = labels.astype("uint16")
    imwrite(mask_file, labels, photometric='minisblack')
    if masks is not None:
        organoMaskFile = Path(result_folder) / ( name_orig+'_organoid-mask.tif')
        imwrite(organoMaskFile, masks.astype("uint16"),photometric='minisblack')
    return labels, img, volume, results, masks
