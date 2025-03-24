from .preprocessing import apply_preproc, get_preproc_name
from .organoidSegmentation import DetectMainOrganoid
from .prediction import predict_label, loadModelIO
from tifffile import imwrite
from pathlib import Path
from skimage.transform import resize
import os
from .fileManager import imToDisplay, ImType, ImInfo
from .montage import createContourOverlay, createMontageTransparency, write_composite
from .cellSegmentation import SegWatershed
from .StatisticsForGui import infoFusion, getFeatures, addColumns, getMitoApoFeatures
from .stardist_elise.models import StarDist3D, StarDist2D
import numpy as np
import pandas as pd
import time

MODEL_CACHE = {}
LIMITS = [128] + [512] * 2

def convert_mu(VolumeList, dx=None, dy=None, dz=None, is_3D=True): 
    """ convert a volume of list with the input voxel size """
    if (dx is not None and dy is not None and dz is not None) and (dx!=1 or dy!=1 or dz!=1):
        vol_convert=[]
        for volume in VolumeList:
            vol_convert.append(volume * dx * dy * dz)
        return vol_convert
    else:
        return VolumeList
    
    
def predict_file_global(filePath,channelNuclei, channelCell, channelOrganoid, organoMethod, organoidParameter,
    keepLargestOrganoid, volumeMinOrganoid,  prep, dict_preproc, 
    factor_resize_Z, factor_resize_Y, factor_resize_X, 
    prob_thresh, nms_thresh, volume_min, volume_max, modelName,modelPath, defaultModel,
    cell, cell_seg_method, cellStains, cytoStains, dx, dy, dz, radius_cell, dist_max, 
    exportChannels, exportOverlays,
    exportParameters, exportComposite):
    montage_file, mask_cell, montage_cell, montage_nuclei_cell, mask_cyto = None,None,None, None, None
    if volume_min is not None:
        volume_min = int(volume_min) * (factor_resize_X*factor_resize_Y*factor_resize_Z*(1/dx)*(1/dy)*(1/dz))
    if volume_max is not None:
        volume_max = int(volume_max) * (factor_resize_X*factor_resize_Y*factor_resize_Z*(1/dx)*(1/dy)*(1/dz))
    
    fileInfo = ImInfo(filePath)
    if not fileInfo.IsHandeled():
        return
    global MODEL_CACHE
    if not (modelName,modelPath,defaultModel) in MODEL_CACHE:
        MODEL_CACHE.clear()
        model = loadModelIO(str(Path(modelPath)/modelName) if modelName is not None and modelPath is not None else defaultModel)
        if model is None:
            return 0
        MODEL_CACHE[(modelName,modelPath,defaultModel)]  = model
        
        
    model = MODEL_CACHE[(modelName,modelPath,defaultModel)]

    
    is_3D_model = model.config.n_dim == 3
    if (fileInfo.type is ImType.Im2D) and (is_3D_model):
        return []
    elif (fileInfo.type is not ImType.Im2D) and not (is_3D_model):
        return []
                        
    #added by victor in nov 2023 for mol dev
    if prob_thresh==0:
         prob_thresh=0.01    
  
    FolderName = fileInfo.name+'_nms'+str(nms_thresh)+'_prob' +str(prob_thresh) #+'_resize'+ 'X' + str(factor_resize_X)+ 'Y' + str(factor_resize_Y)+ 'Z' + str(factor_resize_Z)
    FolderName += (get_preproc_name(dict_preproc)) if len(dict_preproc)>0 else ""
    # FolderName += ( '_volMin' + str(volume_min)) if volume_min is not None else ""
    # FolderName += ( '_volMax' + str(volume_max)) if volume_max is not None else ""
    result_folder = Path(filePath).parent / FolderName
    result_folder.mkdir(exist_ok=True)
    name_split = fileInfo.name.split("_")
    
    dict_composite = {}
    img_i = fileInfo.LoadImage()
    if fileInfo.channel >0:
        channelNucleiTMP = min(channelNuclei,fileInfo.channel-1) if channelNuclei is not None else None
        channelCellTMP = min(channelCell,fileInfo.channel-1) if channelCell is not None else None
        list_chan = ['C' + str(channel_range) for channel_range in range(1, fileInfo.channel+1)]
        for i_compo, c in enumerate(list_chan):
            dict_composite[c] = img_i[:,:,:,i_compo]
            if exportChannels:
                image_file = Path(result_folder) / ( fileInfo.name + '_' + c + '.tif')
                imwrite(image_file, img_i[:,:,:,i_compo], photometric='minisblack')
    else:
        channelNucleiTMP = channelCellTMP = None
        dict_composite['C1'] = img_i
        if exportChannels:
            image_file = Path(result_folder) / ( fileInfo.name + '_C1' + '.tif')
            imwrite(image_file, img_i, photometric='minisblack')
    #cut pictures and keep slected channel if RGB picture
    if len(img_i.shape) == 4:
        if channelNucleiTMP is None:
            im_channel=img_i[:,:,:,0]
        else:
            im_channel=img_i[:,:,:,channelNucleiTMP]
    else:
        im_channel=img_i
    
    im_montage =  im_channel.copy()
    
    if prep:
        pp_time = time.time()
        im_prep = apply_preproc(dict_preproc, img_i, fileInfo)
        print(f'Preprocessing done ! ({time.time() - pp_time} seconds)')
        im_channel = im_prep[:,:,:,channelNucleiTMP] if fileInfo.channel>0 else im_prep
        if fileInfo.channel >0:
            list_chan = ['C' + str(channel_range) for channel_range in range(1, fileInfo.channel+1)]
            for i_compo, c in enumerate(list_chan):
                dict_composite[c + '_prep'] = im_prep[:,:,:,i_compo]
        else:
            dict_composite['C1_prep'] = im_prep
    else:
        im_prep=None   
        


    resolution=[dz, dy, dx] #changed by Victor 2023 10 05 for Molecular device
    org_time = time.time()
    if organoMethod is None:
        organoidMask = None
        masks = None
    elif channelOrganoid == "all":
        organoidMask = DetectMainOrganoid(fileInfo.Display(), organoMethod, organoidParameter, keepLargestOrganoid, volumeMinOrganoid,resolution, fileInfo.bitsPerPixel) if not prep else DetectMainOrganoid(im_prep, organoMethod, organoidParameter,  keepLargestOrganoid, volumeMinOrganoid, fileInfo.resolution, fileInfo.bitsPerPixel)
        masks = [organoidMask]
    else:
        channelOrganoidTMP = min(int(channelOrganoid),fileInfo.channel-1) if channelOrganoid is not None else None
        if fileInfo.channel >0:
            organoidMask = DetectMainOrganoid( imToDisplay(img_i[:,:,:,channelOrganoidTMP], imType = ImType.GrayScale), organoMethod, organoidParameter,  keepLargestOrganoid, volumeMinOrganoid, resolution, fileInfo.bitsPerPixel) if not prep\
                else DetectMainOrganoid( imToDisplay(im_prep[:,:,:,channelOrganoidTMP], imType = ImType.GrayScale), organoMethod, organoidParameter,  keepLargestOrganoid, volumeMinOrganoid, resolution, fileInfo.bitsPerPixel)
        else:
            organoidMask = DetectMainOrganoid(imToDisplay(img_i, imType = ImType.GrayScale), organoMethod, organoidParameter,  keepLargestOrganoid, volumeMinOrganoid, resolution, fileInfo.bitsPerPixel) if not prep else \
                DetectMainOrganoid(imToDisplay(im_prep, imType = ImType.GrayScale), organoMethod, organoidParameter,  keepLargestOrganoid, volumeMinOrganoid, resolution, fileInfo.bitsPerPixel)
        masks = [organoidMask]
    print(f'Organoid detection done ! ({time.time() - org_time} seconds)')
    
    im_shape = im_channel.shape
    #resize pictures
    if im_shape[0] > 1: # 3D image
        img_resize= resize(im_channel, (int(round(im_channel.shape[0] * factor_resize_Z)), int(round(im_channel.shape[1] * factor_resize_Y)),int(round(im_channel.shape[2] * factor_resize_X))), anti_aliasing=True, order = 1)
    else:
        img_resize= resize(im_channel, (1, int(round(im_channel.shape[1] * factor_resize_Y)), int(round(im_channel.shape[2] * factor_resize_X))),anti_aliasing=True, order = 1)
        img_resize = img_resize[..., np.newaxis]
        im_shape = im_shape + (1,)
    
    if any([(lambda x, y: x > y)(img_resize.shape[i], lim) for i, lim in enumerate(LIMITS)]):
        return img_resize.shape
    
    img_i_size = [img_resize]

    predict_label_args = {
    "masks":masks,
    "im_shape" : im_shape, "prob" : prob_thresh, "nms" : nms_thresh, "volume_min" : volume_min,
    "volume_max" : volume_max}
    
    
    start_predict_label = time.time()
    img, labels, masks, volume, results = predict_label(model, img_i_size, 0,  **predict_label_args)
    end_predict_label = time.time()
    print('Time for predict_label ' + str(end_predict_label - start_predict_label))
    dict_composite['nuclei_mask'] = labels
    
    if exportOverlays :
        nucleiContourOverlay = createContourOverlay(labels,1)
        montage_nuclei = createMontageTransparency(imToDisplay(im_montage), nucleiContourOverlay)
        montage_file = Path(result_folder) / ( fileInfo.name +'_nuclei-montage.tif')
        imwrite(montage_file, montage_nuclei, photometric='RGB')
    
    mask_file = Path(result_folder) / ( fileInfo.name +'_nuclei-mask.tif')
    labels = labels.astype("uint16")
    imwrite(mask_file, labels, photometric='minisblack')
    if masks is not None:
        organoMaskFile = Path(result_folder) / ( fileInfo.name +'_organoid-mask.tif')
        imwrite(organoMaskFile, masks.astype("uint16"),photometric='minisblack')
        # imwrite("D:/Elise/images_figure/organanoidSegFinal.tif", masks.astype("uint16"))
    else :
        organoMaskFile = None
    
    cell_time = time.time()
    if cell:
        mask_cell = SegWatershed(fileInfo.MainPath(),labels,cell_seg_method,masks,channelCellTMP, cytoStains, dx = dx, dy = dy, dz = dz, radius = radius_cell, dist_max = dist_max, preprocessed_im = im_prep)
        print(f'Cell detection done ! ({time.time() - cell_time} seconds)')
        # imwrite("D:/Elise/images_figure/CellSegCropped.tif", mask_cell.astype("uint16"))
        mask_cyto = mask_cell.copy()
        mask_cyto[labels>0]=0
        dict_composite['cell_mask'] = mask_cell
        imwrite(Path(result_folder) / ( fileInfo.name + '_cell-mask.tif'), mask_cell, photometric='minisblack')
        imwrite(Path(result_folder) / ( fileInfo.name + '_cyto-mask.tif'), mask_cyto, photometric='minisblack')
        if exportOverlays:
            cellContourOverlay = createContourOverlay(mask_cell,1, [255,255,0])
            nucleiCellContourOverlay = createContourOverlay(labels,1,[255,0,255])
            nucleiCellContourOverlay[labels==0] = cellContourOverlay[labels==0]
            montage_cell = createMontageTransparency(imToDisplay(img_i[...,channelCellTMP]), cellContourOverlay)
            montage_nuclei_cell = createMontageTransparency(imToDisplay(img_i), nucleiCellContourOverlay)
            imwrite(Path(result_folder) / ( fileInfo.name + '_cell-montage.tif'), montage_cell, photometric='RGB')
            imwrite(Path(result_folder) / ( fileInfo.name + '_cell-nuclei-montage.tif'), montage_nuclei_cell, photometric='RGB')

    if organoidMask is not None : 
        dict_composite['organoid_mask'] = masks
    
    
    organoInfoDF, nucleiInfoDF, cellInfoDF, cytoInfoDF = getFeatures(
        nucleiLabelMap = labels,imIntensities=img_i,organoMask=masks,
        nChannel = fileInfo.channel, im_prep = im_prep if prep else None,cellLabelMap=mask_cell if cell else None, 
        cytoLabelMap = mask_cyto if cell else None, dx=dx, dy=dy, dz=dz)
    

    mitoInfoDF, apoInfoDF = getMitoApoFeatures(imIntensities=img_i,organoMask=masks, nChannel = fileInfo.channel, im_prep = im_prep if prep else None, dx=dx,dy=dy,dz=dz)
    
    if exportComposite:
        write_composite(dict_composite, Path(result_folder) / ( fileInfo.name + '_composite.tif'))
        
    fusionInfoDf = infoFusion( organoInfoDF, nucleiInfoDF, cellInfoDF, cytoInfoDF)
    fusionInfoDf = addColumns(fusionInfoDf, col_to_add = None, name_split = name_split, filename = fileInfo.name)
    csv_path = Path(result_folder) / ( fileInfo.name +'_fusion-statistics.csv')
    fusionInfoDf.to_csv(csv_path, index = False)
    
    if organoInfoDF is not None :
        organoInfoDF = addColumns(organoInfoDF, col_to_add = None, name_split = name_split, filename = fileInfo.name)
        csv_path = Path(result_folder) / ( fileInfo.name +'_organoid-statistics.csv')
        organoInfoDF.to_csv(csv_path, index = False)
    
    nucleiInfoDF = addColumns(nucleiInfoDF, col_to_add = None, name_split = name_split, filename = fileInfo.name)
    csv_path = Path(result_folder) / ( fileInfo.name+'_nuclei-statistics.csv')
    nucleiInfoDF.to_csv(csv_path, index = False)

    if cell:
        cellInfoDF = addColumns(cellInfoDF, col_to_add = None, name_split = name_split, filename = fileInfo.name)
        csv_path = Path(result_folder) / ( fileInfo.name + '_cell-statistics.csv')
        cellInfoDF.to_csv(csv_path, index = False)
        cytoInfoDF = addColumns(cytoInfoDF, col_to_add = None, name_split = name_split, filename = fileInfo.name)
        csv_path = Path(result_folder) / ( fileInfo.name+'_cyto-statistics.csv')
        cytoInfoDF.to_csv(csv_path, index = False)
    
            
    #before NMS
    vol_list = volume/(factor_resize_X*factor_resize_Y*factor_resize_Z) #creates the list of volume before nms and divide by the scaling factors
    vol_list = np.delete(vol_list, np.where(vol_list == 0)) #deletes 0 in the list
    vol_convert = convert_mu(vol_list, dx, dy, dz) #uses the voxel size to change the volume list
    fl = fileInfo.name
    file_list = [fl for i in range(len(vol_list))]
    vol_rows = [[file_list[i], vol_convert[i]] for i in range(len(vol_list))]     

    return {
        "file" : True,
        
        "folder" : False,

        "dataframes":{

            "dt_nuclei": nucleiInfoDF,

            "dt_organo": organoInfoDF,

            "dt_cell": cellInfoDF,

            "dt_cyto": cytoInfoDF,
            
            "dt_mito": mitoInfoDF,
            
            "dt_apo": apoInfoDF,
                        
            "volume_list_before_nms" : vol_convert,
            
            "volume_list_before_nms_rows" : vol_rows

            },

        "result_folder" : str(result_folder),
        
        "images_paths":{

            "montage_nuclei":montage_file,

            "mask_nuclei":mask_file,
            
            "mask_organo" : organoMaskFile,

            "montage_cell": (Path(result_folder) / ( fileInfo.name + '_cell-montage.tif')) if montage_cell is not None else None,

            "mask_cell": (Path(result_folder) / ( fileInfo.name + '_cell-mask.tif')) if mask_cell is not None else None,

            "montage_cell_nuclei" : Path(result_folder) / ( fileInfo.name + '_cell-nuclei-montage.tif') if montage_nuclei_cell is not None else None,

            "mask_cyto": Path(result_folder) / ( fileInfo.name + '_cyto-mask.tif') if mask_cyto is not None else None,
            
                        
            }

        }
