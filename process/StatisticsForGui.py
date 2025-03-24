from re import M
import numpy as np
import imageio as io
import SimpleITK as sitk
import pandas as pd
from tifffile import imread
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.measure import regionprops

def getFeatures(nucleiLabelMap,imIntensities,organoMask, nChannel,im_prep = None, cellLabelMap=None, cytoLabelMap=None, 
                              dx=1,dy=1,dz=1,thresholdMasks = None, dict_thresh=None, **kwargs ):

    cell = cellLabelMap is not None

    nucleiInfoDF = getIndividualFeatures(nucleiLabelMap,imIntensities, nChannel, im_prep = im_prep, 
                                         dx=dx, dy=dy, dz=dz, thresholdMasks=thresholdMasks, dictThreshold=dict_thresh,**kwargs)
    if organoMask is not None:
        # nucleiInfoDF = addDistanceFromMaskBorder(nucleiInfoDF,organoMask,dx=dx, dy=dy, dz=dz)
        organoedts = {orglab["label"]: ndimage.distance_transform_edt(orglab["image"], (dz, dx, dy)) if (dx and dy and dz) else ndimage.distance_transform_edt(orglab["image"]) for orglab in regionprops(organoMask)}
        nucleiInfoDF = addDistanceFromMaskBorder(nucleiInfoDF, organoMask, organoedts)
        organoInfoDF = getIndividualFeatures(organoMask,imIntensities, nChannel, im_prep = im_prep, 
                                             dx=dx, dy=dy, dz=dz, thresholdMasks=thresholdMasks, dictThreshold=dict_thresh,**kwargs)
        for organo in organoInfoDF["cell_Id"]:
            organoInfoDF.loc[organoInfoDF["cell_Id"] == organo, "object_number"] = len(nucleiInfoDF[nucleiInfoDF["corresponding_organoid"] == organo])
    else : 
        organoInfoDF = None
    if cell:
        cellInfoDF = getIndividualFeatures(cellLabelMap,imIntensities,nChannel, im_prep = im_prep,
                                           dx=dx, dy=dy, dz=dz, thresholdMasks=thresholdMasks, dictThreshold=dict_thresh,**kwargs)
        # cellInfoDF =  addDistanceFromMaskBorder(cellInfoDF,organoMask,dx=dx, dy=dy, dz=dz) if organoMask is not None else cellInfoDF
        cellInfoDF = addDistanceFromMaskBorder(cellInfoDF, organoMask, organoedts) if organoMask is not None else cellInfoDF
        cytoInfoDF = getIndividualFeatures(cytoLabelMap,imIntensities,nChannel, im_prep = im_prep, 
                                           dx=dx, dy=dy, dz=dz, thresholdMasks=thresholdMasks, dictThreshold=dict_thresh,**kwargs)
        # cytoInfoDF =  addDistanceFromMaskBorder(cytoInfoDF,organoMask,dx=dx, dy=dy, dz=dz) if organoMask is not None else cytoInfoDF
        cytoInfoDF = addDistanceFromMaskBorder(cytoInfoDF, organoMask, organoedts) if organoMask is not None else cytoInfoDF
    else:
        cellInfoDF = None
        cytoInfoDF = None
        
    return organoInfoDF, nucleiInfoDF, cellInfoDF, cytoInfoDF

def getMitoApoFeatures(imIntensities,organoMask, nChannel,im_prep = None, 
                        mitoLabelMap=None, apoLabelMap=None,dx=1,dy=1,dz=1,thresholdMasks = None, dict_thresh=None,**kwargs ):

    if mitoLabelMap is None and apoLabelMap is None:
        return None, None
    if organoMask is not None:
        organoedts = {orglab: ndimage.distance_transform_edt(organoMask==orglab, (dz,dx,dy)) if (dx and dy and dz) else ndimage.distance_transform_edt(organoMask==orglab) for orglab in np.unique(organoMask)[1:]}
    if mitoLabelMap is not None:
        mitoInfoDF = getIndividualFeatures(mitoLabelMap,imIntensities,nChannel, im_prep = im_prep, 
                                           thresholdMasks=thresholdMasks, dictThreshold=dict_thresh, dx=dx, dy=dy, dz=dz,**kwargs)
        # mitoInfoDF =  addDistanceFromMaskBorder(mitoInfoDF,organoMask,dx=dx, dy=dy, dz=dz) if organoMask is not None else mitoInfoDF
        mitoInfoDF = addDistanceFromMaskBorder(mitoInfoDF, organoMask, organoedts) if organoMask is not None else mitoInfoDF
    else:
        mitoInfoDF = None
    
    if apoLabelMap is not None:
        apoInfoDF = getIndividualFeatures(apoLabelMap, imIntensities,nChannel, im_prep = im_prep, 
                                          dx=dx, dy=dy, dz=dz, thresholdMasks=thresholdMasks, dictThreshold=dict_thresh,**kwargs)
        # apoInfoDF =  addDistanceFromMaskBorder(apoInfoDF,organoMask,dx=dx, dy=dy, dz=dz) if organoMask is not None else apoInfoDF
        apoInfoDF = addDistanceFromMaskBorder(apoInfoDF, organoMask, organoedts) if organoMask is not None else apoInfoDF
    else:
        apoInfoDF = None
        
    return mitoInfoDF, apoInfoDF

def getThreshFeatures(imIntensities, nChannel,im_prep = None, 
                        threshLabelMap=None,dx=1,dy=1,dz=1,**kwargs ):

    if threshLabelMap is not None:
        threshInfoDF = getIndividualFeatures(threshLabelMap, imIntensities,nChannel, im_prep = im_prep, dx=dx, dy=dy, dz=dz,**kwargs)
    else:
       threshInfoDF = None 
    return threshInfoDF

def infoFusion(organoInfoDF, nucleiInfoDF, cellInfoDF=None, cytoInfoDF=None):
    cell = cellInfoDF is not None
    organo = organoInfoDF is not None

    if organo:
        fusionDF = nucleiInfoDF.add_prefix('nuclei_').merge(
            organoInfoDF.add_prefix('organo_'),
            left_on="nuclei_corresponding_organoid", right_on="organo_cell_Id", how="inner"
        ).drop(["organo_cell_Id"],axis=1).rename(
            columns={"nuclei_cell_Id":"cell_Id"}
        ).set_index("cell_Id")
        fusionDF["corresponding_organoid"] = fusionDF["nuclei_corresponding_organoid"]
    else:
        fusionDF = nucleiInfoDF.set_index("cell_Id").add_prefix('nuclei_')

    if cell:
        fusionDF = fusionDF.join(
            cellInfoDF.set_index("cell_Id").add_prefix('cell_'),how="left").join(
            cytoInfoDF.set_index("cell_Id").add_prefix('cyto_'),how="left")
    
    return fusionDF.reset_index()
    


def addDistanceFromMaskBorder(objectInfoDf,mask, edts):

    centroids = objectInfoDf[["centroid_Z", "centroid_Y", "centroid_X"]].values.astype(np.uint16)
    objectInfoDf["corresponding_organoid"] = [mask[tuple(centroid)] for centroid in centroids]
    
    #change victor 20231201 change 0 to np.float64(0)    
    objectInfoDf["dist_to_border_um"] = len(objectInfoDf["corresponding_organoid"]) * [np.float64(0)]
    objectInfoDf["dist_to_border_ratio"] = len(objectInfoDf["corresponding_organoid"]) * [np.float64(0)]
    for organo in regionprops(mask):
        lab_organo = organo["label"]
        ofset_organo = organo["bbox"][:len(organo["bbox"])//2]
        centroids_organo = centroids[objectInfoDf["corresponding_organoid"] == lab_organo] - ofset_organo
        edt = edts[organo["label"]]
        distances = [edt[tuple(centroid)] for centroid in centroids_organo]
        ratios = distances/np.max(edt)
        objectInfoDf.loc[objectInfoDf["corresponding_organoid"] == lab_organo, "dist_to_border_um"] = distances[:len(objectInfoDf[objectInfoDf["corresponding_organoid"] == lab_organo])]
        objectInfoDf.loc[objectInfoDf["corresponding_organoid"] == lab_organo, "dist_to_border_ratio"] = ratios[:len(objectInfoDf[objectInfoDf["corresponding_organoid"] == lab_organo])]
       
    return objectInfoDf

def getIndividualFeatures(label, image, nChannel, filename=None, name_split=None, col_to_add = None, 
                          im_prep = None, dx=None, dy=None, dz=None, thresholdMasks = None, dictThreshold = None):
    label=label.astype('uint16')
    inputMask = sitk.GetImageFromArray(label)
    df_features = getShapeFeaturesFromArray(inputMask, filename, name_split, col_to_add , dx, dy, dz)
    if nChannel>1:
        for i in range(nChannel):
            df_features = df_features.join(getIntensityFeaturesFromArray(inputMask, image, i),)
            if im_prep is not None : 
                df_features = df_features.join(getIntensityFeaturesFromArray(inputMask, image, i, im_prep = im_prep),)
    else:
        df_features = df_features.join(getIntensityFeaturesFromArray(inputMask, image),)
        if im_prep is not None :    
            df_features = df_features.join(getIntensityFeaturesFromArray(inputMask, image, im_prep = im_prep),)
            
    if thresholdMasks is not None:
        i = 0
        for c in dictThreshold.keys():
            for parameters in dictThreshold[c]:
                thresholdMask = thresholdMasks[:,:,:,i]
                df_features = df_features.join(getIntensityFeaturesFromThresh(inputMask, thresholdMask, meth = parameters[0], param = parameters[1], sup = parameters[2], channel = c))
                i+=1
    
    
    
    return df_features
    
def getIntensityFeaturesFromArray(inputMask, image, channel = None, im_prep = None):
    inputChannel_np = image[:,:,:,channel] if channel is not None else image
    if len(inputChannel_np.shape) == 4:
        inputChannel_np = inputChannel_np[..., 0]
    inputChannel = sitk.GetImageFromArray(inputChannel_np)
    inputChannel_square = sitk.GetImageFromArray(np.square(inputChannel_np))
    input_mask_inverted = np.uint8(sitk.GetArrayFromImage(inputMask) == 0)
    filter = sitk.LabelStatisticsImageFilter()
    filter.Execute(inputChannel, inputMask)
    filter2 = sitk.LabelStatisticsImageFilter()
    filter2.Execute(inputChannel_square, inputMask)
    nLabels=filter.GetNumberOfLabels()
    if channel is None:
        list_names = ["max_intensity_value", "min_intensity_value",
                      "average_intensity_value",
                      "standard_deviation_intensity_value", "sum_intensity_value", "snr_value"]
    else:
        list_names = ['max_intensity_value_C'+ str(channel+1), 'min_intensity_value_C'+ str(channel+1), 'average_intensity_value_C'+ str(channel+1),\
            'standard_deviation_intensity_value_C'+ str(channel+1), 'sum_intensity_value_C'+ str(channel+1),
            'snr_value_C' + str(channel + 1)]
    list_names = [el + "_prep" for el in list_names]  if im_prep is not None else list_names
    intensityFeatures_df = pd.DataFrame(columns=list_names, index=np.arange(1, nLabels+1))
    
    for i in range(1, nLabels+1):
        if not filter.HasLabel(i):
            continue
        intensityFeatures_df.loc[i, list_names[0]] =  filter.GetMaximum(i)
        intensityFeatures_df.loc[i, list_names[1]] =  filter.GetMinimum(i)
        intensityFeatures_df.loc[i, list_names[2]] = filter.GetMean(i)
        intensityFeatures_df.loc[i, list_names[3]] =  filter.GetSigma(i)
        intensityFeatures_df.loc[i, list_names[4]] = filter.GetSum(i)
        signal_power = filter2.GetMean(i)
        bb = filter.GetBoundingBox(i)
        if len(bb) == 6:
            x_min, x_max, y_min, y_max, z_min, z_max = bb
            input_crop = inputChannel_np[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
            mask_crop = input_mask_inverted[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        else:
            x_min, x_max, y_min, y_max = bb
            input_crop = inputChannel_np[y_min:y_max+1, x_min:x_max+1]
            mask_crop = input_mask_inverted[y_min:y_max+1, x_min:x_max+1]
        noise_power = np.var(input_crop[mask_crop > 0]) + 1e-8
        intensityFeatures_df.loc[i, list_names[5]] = 10 * np.log10(signal_power / noise_power)
    return intensityFeatures_df

def getIntensityFeaturesFromThresh(inputMask, image, meth = None, param = None, sup = None, channel = None):

    inputChannel = sitk.GetImageFromArray(image)
    filter = sitk.LabelStatisticsImageFilter()
    filter.Execute(inputChannel, inputMask)
    nLabels=filter.GetNumberOfLabels()
    supInf = "sup" if sup else "inf"
    name = "C"+ channel + "_"+meth.split()[0]+"_"+str(param) + "_" + supInf
    list_names = ['max_intensity_value_threshold_'+ name, 'min_intensity_value_threshold_'+ name, 'average_intensity_value_threshold_'+ name,\
            'standard_deviation_intensity_value_threshold_'+ name, 'sum_intensity_value_threshold_'+ name]
    intensityFeatures_df = pd.DataFrame(columns=list_names, index=np.arange(1, nLabels+1))
    for i in range(1, nLabels+1):
        if not filter.HasLabel(i):
            continue
        intensityFeatures_df.loc[i, list_names[0]] =  filter.GetMaximum(i)
        intensityFeatures_df.loc[i, list_names[1]] =  filter.GetMinimum(i)
        intensityFeatures_df.loc[i, list_names[2]] = filter.GetMean(i)
        intensityFeatures_df.loc[i, list_names[3]] =  filter.GetSigma(i)
        intensityFeatures_df.loc[i, list_names[4]] = filter.GetSum(i)
    return intensityFeatures_df

#this function uses the label and the image (channel) to get the features
#can use the voxel size if specified
def getShapeFeaturesFromArray(inputMask, filename=None, name_split=None, col_to_add = None, dx=None, dy=None, dz=None):
    

    filter = sitk.LabelShapeStatisticsImageFilter()
    filter.Execute(inputMask)

    nLabels = filter.GetNumberOfLabels()
    dim = inputMask.GetDimension()
    cellId = []
    bbX = []
    bbY = []
    bbZ = []

    bbW = []
    bbH = []
    bbWZ = []

    cX= []
    cY = []
    cZ = []
    volume = []
    elongation = []
    equivalantDiameterX = []
    equivalantDiameterY = []
    equivalantDiameterZ = []
    roundness = []
    is_3d = dim>2
    for i in range(nLabels):
        if not filter.HasLabel(i+1):
            continue
        bb = filter.GetBoundingBox(i+1)
        cellId.append(i+1)
        bbX.append(bb[0])
        bbW.append(bb[2+1*is_3d])
        bbY.append(bb[1])
        bbH.append(bb[3+1*is_3d])

        if is_3d:
            bbZ.append(bb[2])
            bbWZ.append(bb[5])
        cc = filter.GetCentroid(i + 1)
        cX.append(cc[0])
        cY.append(cc[1])
        if is_3d:
            cZ.append(cc[2])
        elongation.append(filter.GetElongation(i + 1))
        eqdi = filter.GetEquivalentEllipsoidDiameter(i + 1)
        equivalantDiameterZ.append(eqdi[0])
        equivalantDiameterY.append(eqdi[1])
        equivalantDiameterX.append(eqdi[2])
        roundness.append(filter.GetRoundness(i + 1))
        volume.append(filter.GetNumberOfPixels(i + 1))
        
    if dx is not None and dy is not None and dz is not None:
        bbX_mu, bbY_mu, bbZ_mu, bbW_mu, bbH_mu, bbWZ_mu = bbX[:], bbY[:], bbZ[:], bbW[:], bbH[:], bbWZ[:]
        centroidX_mu, centroidY_mu, centroidZ_mu = cX[:], cY[:], cZ[:]
        volume_mu = volume[:]
        equivalantDiameterX_mu, equivalantDiameterY_mu, equivalantDiameterZ_mu = equivalantDiameterX[:], equivalantDiameterY[:], equivalantDiameterZ[:]
        
        for i in range(len(cellId)):
            bbX_mu[i] = bbX_mu[i]*dx
            bbY_mu[i] = bbY_mu[i]*dy
            bbW_mu[i] = bbW_mu[i]*dx
            bbH_mu[i] = bbH_mu[i]*dy
            if is_3d:
                bbZ_mu[i] = bbZ_mu[i]*dz
                bbWZ_mu[i] = bbWZ_mu[i]*dz
            centroidX_mu[i] = centroidX_mu[i]*dx
            centroidY_mu[i] = centroidY_mu[i]*dy
            centroidZ_mu[i] = centroidZ_mu[i]*dz
            volume_mu[i] = volume_mu[i]*dx*dy*dz
            equivalantDiameterX_mu[i] = equivalantDiameterX_mu[i]*dx
            equivalantDiameterY_mu[i] = equivalantDiameterY_mu[i]*dy
            equivalantDiameterZ_mu[i] = equivalantDiameterZ_mu[i]*dz
        
    if is_3d:
        features = ["cell_Id", "bb_X", "bb_Y", "bb_Z", "bb_W", "bb_H", "bb_WZ", "centroid_X", "centroid_Y", "centroid_Z", "volume", 
                    "elongation", "roundness", "equivalent_ellipsoid_diameter_X", "equivalent_ellipsoid_diameter_Y", "equivalent_ellipsoid_diameter_Z", 
                    "bb_X_um", "bb_Y_um", "bb_Z_um", "bb_W_um", "bb_H_um", "bb_WZ_um",
                    "centroid_X_um", "centroid_Y_um", "centroid_Z_um", "volume_um3", "equivalent_ellipsoid_diameter_X_um",
                    "equivalent_ellipsoid_diameter_Y_um", "equivalent_ellipsoid_diameter_Z_um"]
        df = pd.DataFrame(np.array([cellId, bbX, bbY, bbZ, bbW, bbH, bbWZ, cX, cY, cZ, volume, elongation, roundness, equivalantDiameterX, 
                                    equivalantDiameterY, equivalantDiameterZ, 
                                    bbX_mu, bbY_mu, bbZ_mu, bbW_mu, bbH_mu, bbWZ_mu, centroidX_mu, centroidY_mu, centroidZ_mu, volume_mu,
                                    equivalantDiameterX_mu, equivalantDiameterY_mu, equivalantDiameterZ_mu]).T ,columns=features,index=cellId)
    else:
        features = ["cell_Id","bb_X", "bb_Y", "bb_W", "bb_H", "centroid_X", "centroid_Y", "centroid_Z", "volume", "elongation", "roundness",
                    "equivalent_ellipsoid_diameter_X", "equivalent_ellipsoid_diameter_Y", "equivalent_ellipsoid_diameter_Z", "max_intensity_value",
                    "bb_X_um", "bb_Y_um", "bb_W_um", "bb_H_um", "centroid_X_um", "centroid_Yum", "centroid_Zum", "volume_um3",
                    "equivalent_ellipsoid_diameter_X_um", "equivalent_ellipsoid_diameter_Y_um", "equivalent_ellipsoid_diameter_Z_um"]
        df = pd.DataFrame(np.array([cellId, bbX, bbY, bbW, bbH, cX, cY, cZ, volume, elongation, roundness, equivalantDiameterX, 
                                    equivalantDiameterY, equivalantDiameterZ, 
                                    bbX_mu, bbY_mu, bbW_mu, bbH_mu, centroidX_mu, centroidY_mu, centroidZ_mu, volume_mu,
                                    equivalantDiameterX_mu, equivalantDiameterY_mu, equivalantDiameterZ_mu]).T,columns=features,index=cellId)
    if filename is not None and name_split is not None:
        df = addColumns(df, col_to_add,name_split,filename)        
    return df

def addColumns(df, col_to_add,name_split,filename):
    df.insert(0, 'file', filename)
    if col_to_add:
        nb_col = col_to_add
    else : 
        nb_col = len(name_split)
    for i in range(nb_col):
        col_name = 'name_split_' + str(i+1)
        if i+1 > len(name_split):
            el = ''
        else :
            el = name_split[i]
        df.insert(i+1, col_name, el) 
    return df.copy()

#uses the previous function to get a dictionnary of features for a list of mask and images





    
