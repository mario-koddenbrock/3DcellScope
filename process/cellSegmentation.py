# -*- coding: utf-8 -*-
# author Titouan Poquillon & Elise Drimaracci
# 
# DEDICATED MODUL FOR CELL SEGMENTATION
import numpy as np
import pandas as pd
from skimage.filters import threshold_multiotsu
from scipy import ndimage
from skimage.measure import label, regionprops_table, regionprops
from skimage.morphology import remove_small_objects, dilation
from skimage.segmentation import watershed
from skimage.filters._median import median
from tifffile import imread
from .montage import BuildCellMontage
from .organoidSegmentation import DetectMainOrganoid, structuring_element
from .StatisticsForGui import getFeatures, infoFusion
from .fileManager import *

def updateCellsParameters(window1):
    method = window1["cell_segmentation"].get()
    if method == "intensity channel":
        window1["-P1TextCellSeg-"].update(value = "dist_max(pixel)", visible = True)
        window1["-P1CellSeg-"].update(value = "14", visible = True, disabled = False)
        window1["-P2TextCellSeg-"].update(visible = False)
        window1["-P2CellSeg-"].update(visible = False)
    elif method == "distance map":
        window1["-P1TextCellSeg-"].update(value = "dist_max(pixel)", visible = True)
        window1["-P1CellSeg-"].update(value = "14", visible = True, disabled = False)
        window1["-P2TextCellSeg-"].update(value = "radius(pixel)", visible = True)
        window1["-P2CellSeg-"].update(value = "5", visible = True, disabled = False)       
    return 

def SegWatershed(filePath2Im3D, nucleusLabelMap, cell_seg_method, mainOrganoid = None, channelCell = 1, cytoStains = False, dx = 1, dy = 1, dz = 1, radius = 5, dist_max = 14, preprocessed_im = None):

    fileInfo = ImInfo(filePath2Im3D)
    if preprocessed_im is not None:
        im = preprocessed_im
    else:
        im = fileInfo.LoadImage()
    # if mainOrganoid is None:
    #     mainOrganoid = DetectMainOrganoid(im3D)
    if fileInfo.channel>0:
        imCellI = imToDisplay(im[...,min(channelCell,fileInfo.channel-1)],imType=ImType.GrayScale)
    else:
        #WARNING not ideal
        imCellI = imToDisplay(im,imType=ImType.GrayScale)
    imCellI = np.squeeze(imCellI)
    if cell_seg_method == 'Watershed intensity channel' :
        # cellLabelMap = WaterchedFromNucleiSeed(imCellI,nucleusLabelMap, cytoStains)#, organoidMask=mainOrganoid)
        cellLabelMap = WaterchedFromNucleiSeed(imCellI,nucleusLabelMap, cytoStains, organoidMask=mainOrganoid)
    elif cell_seg_method == 'Watershed distance map' :
        # cellLabelMap = WaterchedFromDistanceMap(nucleusLabelMap,dx, dy, dz, radius)#, organoidMask = mainOrganoid)
        cellLabelMap = WaterchedFromDistanceMap(nucleusLabelMap,dx, dy, dz, radius, organoidMask = mainOrganoid)
    
    cellLabelMap[mainOrganoid == 0] = 0
    props = regionprops(cellLabelMap.astype('uint16'))
    # distance_map = np.zeros_like(cellLabelMap)
    # bounding_boxes = []
    cell_too_far_map:np.ndarray = np.zeros_like(cellLabelMap, dtype=bool)
    for prop in props:
        # bbox = prop.bbox
        # bounding_boxes.append(bbox)
        # nucleus = nucleusLabelMap[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]].copy()
        # nucleus[nucleus != prop.label] = 0
        # nucleus[nucleus == prop.label] = 1
        # cell_bbox = cellLabelMap[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]].copy()
        # one_bbox = np.ones(cell_bbox.shape)
        bslice = prop.slice
        cell_bbox = 1*prop.image
        nucleus = 1*(cell_bbox*nucleusLabelMap[bslice]>0)
        one_bbox = np.ones(cell_bbox.shape, cell_bbox.dtype)
        
        cell_segmentation = one_bbox - nucleus
        cell_distance_map = ndimage.distance_transform_edt(cell_segmentation, (dz,dx,dy))
        # distance_map[cellLabelMap == prop.label] = cell_distance_map[cell_bbox == prop.label]
        a=(cell_distance_map[cell_bbox>0])>=dist_max
        cell_too_far_map[bslice][cell_bbox>0] = (cell_distance_map[cell_bbox>0])>=dist_max
    
    # cellLabelMap[distance_map> dist_max] = 0 #dist_max en um
    cellLabelMap[cell_too_far_map] = 0 #dist_max en um
    cellLabelMap[nucleusLabelMap==0] = MedianFilterPostProcessing(cellLabelMap)[nucleusLabelMap==0]
    return cellLabelMap.astype('uint16')


def WaterchedFromNucleiSeed(ImIntensity,nucleusLabelMap, cytoStains, organoidMask):
    """_summary_

    Args:
        ImIntensity (np.ndarray): intensity image of organoid stack Z X Y 
        nucleusLabelMap (np.ndarray): label mask of nuclei stack Z X Y
        organoidMask (np.ndarray): binary mask of organoid stack Z X Y
        

    Returns:
        np.ndarray: label mask of cell stack Z X Y
    """
    # cellWatershed = watershed(-ImIntensity,markers=nucleusLabelMap) if cytoStains else watershed(ImIntensity,markers=nucleusLabelMap)
    cellWatershed = watershed(-ImIntensity,markers=nucleusLabelMap, mask=organoidMask) if cytoStains else watershed(ImIntensity,markers=nucleusLabelMap, mask=organoidMask)

    return cellWatershed

def WaterchedFromDistanceMap(nucleusLabelMap,dx = 1, dy = 1, dz = 1, radius = 5, organoidMask=None):
    binary_mask = np.zeros(nucleusLabelMap.shape)
    binary_mask[nucleusLabelMap==0]=1
    edt = abs(-ndimage.distance_transform_edt(binary_mask, (dz,dx,dy)) )
    rx, ry, rz = (radius/dx, radius/dy, radius/dz) 
    selem = structuring_element(int(rz), int(rx), int(ry)) 
    donut_mask = ndimage.binary_dilation(nucleusLabelMap, structure=selem)
    if organoidMask is not None:
        donut_mask[organoidMask==0]=0
    cellWatershed = watershed(edt,markers=nucleusLabelMap,mask=donut_mask)
    cellWatershed = cellWatershed.astype("uint16")
    return cellWatershed

def MedianFilterPostProcessing(maskOrLabel,size=3,repeat=1):
    filteredMask = maskOrLabel.copy()
    for i in range(repeat):
        filteredMask =  median(filteredMask, SphereFootprint(size))
    return filteredMask

def maskAndReLabel(labelMap, mask=None, min_size = 64):
    if mask is None:
        mask_organo = np.ones(labelMap.shape, dtype=np.uint16)
        return labelMap, None
    else : 
        mask_organo = mask.copy()
        old_mask = mask.copy()
    labelMapMasked = np.zeros(labelMap.shape, dtype=np.uint16)
    i=1
    for k, region in enumerate(regionprops(labelMap.astype(np.uint16))):
        centroid = (int(region["centroid"][0]), int(region["centroid"][1]), int(region["centroid"][2]))
        if (region["area"]>min_size) & (mask_organo[centroid] != 0):
            labelMapMasked[region["slice"]][region["image"]] = i
            i+=1
            mask_organo[region["slice"]][region["image"]]=mask_organo[centroid]
    if mask is not None:
        selem = structuring_element(int(1), int(1), int(2))
        dilatation = dilation(mask_organo-old_mask, footprint=selem)
        mask_organo = np.maximum(mask_organo, dilatation) if mask is not None else None
    return labelMapMasked.astype("float32"), mask_organo.astype("uint16") if mask is not None else None

def SphereFootprint(radius=3, dtype=np.uint8):
    L = np.arange(-radius, radius + 1)
    X, Y, Z = np.meshgrid(L, L, L)
    return np.array((X ** 2 + Y ** 2 + Z**2) <= radius ** 2, dtype=dtype)

if __name__ == "__main__":
    from tifffile import imread, imwrite
    import matplotlib.pyplot as plt

    imRGB3D = imread("statics/data/RGB_test.tif") # RGB stack of organoid Z X Y C
    nucleusLabelMap = imread("statics/data/test_mask.tif") # nuclei mask of organoid Z X Y
    mainOrganoid = DetectMainOrganoid(imRGB3D)
    nucleusLabelMap = maskAndReLabel(nucleusLabelMap,mainOrganoid)
    imwrite("statics/data/test_mainOrganoid.tif",mainOrganoid)
    cellLabelMap = WaterchedFromNucleiSeed(imRGB3D[...,1],nucleusLabelMap,organoidMask=mainOrganoid)
    cellLabelMap = MedianFilterPostProcessing(cellLabelMap)
    imwrite("statics/data/test_cellLabelMap2.tif",cellLabelMap)
    dx=dy=dz = 0.5
    organoInfoDF, nucleiInfoDF, cellInfoDF, cytoInfoDF, mitoInfoDF, apoInfoDF, threshInfoDF = getFeatures(nucleusLabelMap,imRGB3D[...,2],mainOrganoid,np.sum(imRGB3D, axis=-1),cellLabelMap,imRGB3D[...,1],dx=dx, dy=dy, dz=dz)
    fusionInfoDf1 = infoFusion(organoInfoDF, nucleiInfoDF, cellInfoDF)
    fusionInfoDf2 = infoFusion(organoInfoDF, nucleiInfoDF)
    for i, el in enumerate(SegWatershed("statics/data/RGB_test.tif", nucleusLabelMap)):
        imwrite("statics/data/test_%d.tif"%i,el)
    print("done")