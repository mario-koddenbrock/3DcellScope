from tifffile import imread, imwrite, TiffFile
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from csbdeep.utils import normalize
import math
import re

COMPOSIT_RGB_COL = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
CHANELTAGS = ("C1","C2","C3","C4","C5","C6",)

class ImType(Enum):
    FileNotFound = 1
    UnHandeled = 2
    RGB = 3
    GrayScale = 4
    ChannelFiles = 5
    Composite = 6
    Im2D = 7


class ImInfo:
    type: ImType
    paths: tuple 
    name: str = ""
    channel: int = 0
    channelTags: tuple = ()
    loadOrder:str = "ZXY"
    dtype: np.uint8
    resolution: tuple = (1., 1., 1.)
    bitsPerPixel: int

    def __init__(self, imPath, channelTags = CHANELTAGS) -> None:
        self.FindInfo(imPath, channelTags)

    def IsHandeled(self):
        return self.type not in [ImType.FileNotFound, ImType.UnHandeled] #, ImType.Im2D]

    def FindInfo(self, imPath, channelTags = CHANELTAGS):
        imPath = Path(imPath)
        name = "".join(imPath.name.split(".")[:-1])            
        self.name = name
        self.paths = [imPath]
        
        if not imPath.exists() or not imPath.is_file():
            self.type = ImType.FileNotFound
            return
        
        try:
            # tags = TiffTags(imPath)
            tf = TiffFile(imPath)
        except:
            self.type = ImType.UnHandeled
            return
        
        print('name =', name)
        # self.resolution = readTiffVoxelSize(imPath)
        # self.bitsPerPixel = bitsPerPixel(imPath)
        try:
            tags = TiffTags(tf)
        except:
            tf.close()
            self.type = ImType.UnHandeled
            return
        self.resolution = readTiffVoxelSize(tf)
        self.bitsPerPixel = bitsPerPixel(tf)
        tf.close()
        self.dtype = tags["dtype"]
        serieShape = tags["serieShape"]
        pageShape = tags["pageShape"]
        extraDim = serieShape[:-len(pageShape)] # legth can be either 0(2d) 1 (3d) or 2 (3d composite)
        
        demo_names = ['Human-Colon-Organoids-C1', 'HCT116-Cells-Monolayer-C1',
                      'PDAC-C1', 'PDAC-C2', 'ZeroG-Breast-Cancer-Spheroid-C1']
        if name in demo_names:
            paths, localTags = getFileTagSiblings(imPath, channelTags)
            self.name = nameOfChannelFile(name, localTags[0])
            self.type = ImType.ChannelFiles
            self.paths = paths
            self.channel = 0
            self.loadOrder = 'ZXY'
            if name == 'PDAC-C1':
                self.type = ImType.ChannelFiles
                self.channel = 2
            # print('self.type =', self.type)
            return
        else:
            is_pattern_in_name = any([name.endswith('_C' + str(i)) for i in range(10)])
            if not is_pattern_in_name:
                self.channel = 0
                self.type = ImType.UnHandeled
                return
        
        # print('extradim =', extraDim, ' pageShape =', pageShape)
        if len(extraDim)==1 and len(pageShape)==2:
            paths, localTags = getFileTagSiblings(imPath, channelTags)
            # print('paths =', paths)
            if len(paths)==1:
                self.channel = 0
                self.type = ImType.UnHandeled
            else:
                # print('is a channel file')
                self.channel = len(paths)
                self.type = ImType.ChannelFiles
                self.name = nameOfChannelFile(name, localTags[0])
                self.paths = paths
                self.channelTags = localTags
                # return
        else:
            self.type = ImType.UnHandeled
        # print('self.type =', self.type)

    def MainPath(self):
        return Path(self.paths[0])

    def LoadImage(self):
        if self.type in [ImType.RGB, ImType.GrayScale]:
            return imread(self.MainPath())
        elif self.type is ImType.ChannelFiles:
            return np.stack([imread(path) for path in self.paths], axis = -1)
        elif self.type is ImType.Composite:
            im = imread(self.MainPath())
            if self.loadOrder == "ZCXY": #case Z,C,X,Y
                return im.transpose(0,2,3,1)
            else: return im #case Z,X,Y,C
        elif self.type is ImType.Im2D:
            im = imread(self.MainPath())
            if len(im.shape) == 2:
                im = im[np.newaxis, ..., np.newaxis]
            elif len(im.shape) == 3:
                im = im[np.newaxis]
            im = np.transpose(im, axes=(0, 2, 3, 1))
            return im # (1, Y, X, C)
        else:
            return None

    def Display(self,imArray=None):
        if imArray is None:
            imArray = self.LoadImage()
        return imToDisplay(imArray, self.type)

    def __str__(self):
        return self.name

# def TiffTags(imPath):
#     tif_tags = {}
#     with TiffFile(imPath) as tif:
#         for tag in tif.pages[0].tags.values():
#             name, value = tag.name, tag.value
#             tif_tags[name] = value
#         tif_tags["pageShape"] = tif.pages[0].shape
#         tif_tags["serieShape"] = tif.series[0].shape
#         tif_tags["dtype"] = tif.series[0].dtype
#     return tif_tags

def TiffTags(tif:TiffFile):
    tif_tags = {}
    for tag in tif.pages[0].tags.values():
        tif_tags[tag.name] = tag.value
    tif_tags["pageShape"] = tif.pages[0].shape
    series = tif.series
    serie0 = series[0]
    tif_tags["serieShape"] = serie0.shape
    tif_tags["dtype"] = tif.series[0].dtype
    return tif_tags

# def readTiffVoxelSize(file_path):

#     def _xy_voxel_size(tags, key):
#         assert key in ['XResolution', 'YResolution']
#         if key in tags:
#             num_pixels, units = tags[key].value
#             return units / num_pixels
#         # return default
#         return 1.

#     with TiffFile(file_path) as tiff:
#         image_metadata = tiff.imagej_metadata
#         if image_metadata is not None:
#             z = image_metadata.get('spacing', 1.)
#         else:
#             # default voxel size
#             z = 1.

#         tags = tiff.pages[0].tags
#         # parse X, Y resolution
#         y = _xy_voxel_size(tags, 'YResolution')
#         x = _xy_voxel_size(tags, 'XResolution')
#         # return voxel size
#         return (z, y, x)
    
def readTiffVoxelSize(tif:TiffFile):

    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return 1.

    image_metadata = tif.imagej_metadata
    if image_metadata is not None:
        z = image_metadata.get('spacing', 1.)
    else:
        # default voxel size
        z = 1.
        
    tags = tif.pages[0].tags
    # parse X, Y resolution
    y = _xy_voxel_size(tags, 'YResolution')
    x = _xy_voxel_size(tags, 'XResolution')
    # return voxel size
    return (z, y, x)

    
# def bitsPerPixel(path):
#     try:
#         with TiffFile(path) as tif:
#             image_metadata = tif.imagej_metadata
#             if image_metadata is not None:
#                 info = image_metadata.get('Info')
#                 infoFound=False
#                 if info is not None:
#                     try:
#                         match = re.search(r"BitsPerPixel = (\d+)", info)
#                         p = int(match.group(1))
#                         infoFound=True
#                     except:
#                         infoFound=False
                
#                 maxVal=image_metadata.get('max')
#                 if maxVal is not None: 
#                     p=int(math.ceil(math.log2(maxVal)))       
#         return p
#     except:
#         image = imread(path)
#         max_im = image.max()
#         image = []
#         p = int(math.ceil(math.log2(max_im))) if max_im > 0 else 8
#     return p

def bitsPerPixel(tif:TiffFile):
    try:
        image_metadata = tif.imagej_metadata
        if image_metadata is not None:
            info = image_metadata.get('Info')
            match = re.search(r"BitsPerPixel = (\d+)", info)
            p = int(match.group(1))
        else:
            raise Exception("Can't read BitsPerPixel metadata")
    except: # assume value based the first tif page (to avoid loading heavy 3d stack)
        image = tif.asarray(0)
        max_im = image.max()
        p = int(math.ceil(math.log2(max_im))) if max_im > 0 else 8
    return p


def imToDisplay(imArray, imType = None):
    if imType is None:
        imType = arraytToImtype(imArray)
    if imType in [ImType.UnHandeled,ImType.FileNotFound]: # ,ImType.Im2D]:
        return None
    elif imType is ImType.GrayScale:
        return (normalize(imArray, 0, 100, axis=(0,1,2))*255).astype('uint8')
    elif imType in [imType.ChannelFiles, imType.Composite]:
        if imArray.shape[-1]==1:
            return (normalize(imArray[...,0], 0, 100, axis=(0,1,2))*255).astype('uint8')
        elif imArray.shape[-1] <= 3:
            rgbComposit = np.zeros(imArray.shape[:-1]+(3,))
            for i, col in zip(range(imArray.shape[-1]), COMPOSIT_RGB_COL):
                imChan = normalize(imArray[...,i], 0, 100, axis=(0,1,2))
                for j in range(3):
                    rgbComposit[...,j] += imChan*col[j]
            rgbComposit[rgbComposit>255]=255
        else : 
            rgbComposit = imArray.copy()
        return rgbComposit.astype(np.uint8)



def arraytToImtype(imArray):
    if imArray is None:
        return ImType.FileNotFound
    shape = imArray.shape
    if len(shape) == 3:
        if shape[0] == 1:
            return ImType.Im2D
        else:
            return ImType.GrayScale
    elif len(shape) == 4 and shape[3]==3 and imArray.dtype == np.uint8:
        return ImType.RGB
    elif len(shape) ==4:
        return ImType.Composite
    else: return ImType.UnHandeled

def getFileTagSiblings(imPath, channelTags):
    name, parent = imPath.name, imPath.parent
    if channelTags is None or len(channelTags)<2 or channelTags[0] not in name:
        return [imPath], ['']
    paths, localTags = [imPath], [channelTags[0]]
    for tag in channelTags[1:]:
        path = parent / rreplace(name,channelTags[0], tag,1)
        if path.exists(): 
            paths.append(path), localTags.append(tag)
        # else: return paths
    return paths,localTags


def ImageFolderInfos(imFolder, channelTags=CHANELTAGS, FileExtension = ".tif*" , onlyHandeled=True):
    imPaths = list(Path(imFolder).glob("*"+FileExtension),)
    infos = [ImInfo(imPath,channelTags) for imPath in imPaths]
    if onlyHandeled:
        infos = [info for info in infos if info.IsHandeled()]
    secondary = sum([info.paths[1:] for info in infos if info.type is ImType.ChannelFiles],[])
    infos = [info for info in infos if info.paths[0] not in secondary]
    return infos
    
def rreplace(s: str, old :str, new :str, occurrence=1):
    """right replace
    Args:
        s (str): old string
        old (str): str tag to replace
        new (str):new strt ag
        occurrence (int, optional): nuber of replacement. Defaults to 1.
    Returns:
        str: right replaced string
    """
    li = s.rsplit(old, occurrence)
    return new.join(li)

def nameOfChannelFile(name,tag):
    sep = ["_","-",""]
    tags = sum([["%s%s"%(tag,s),"%s%s"%(s,tag)] for s in sep],[])
    for t in tags:
        if t in name:
            return rreplace(name,t,"",1)


if __name__ == "__main__":
    print("Hello")
    rgbPath = r"statics\data\multi_im_RGB\RGB_test.tif"
    grayPath = r"statics\data\multi_im_monocan\P5-30X-_A04_G004_0001.oir_cropped.tif"
    multiCHanPath = r"statics\data\multi_im_bicannal\C1-P24-A2-30X-88pos_A02_G002_0001.tif"
    compositePath = r"statics\data\im_composit\test.tif"
    randomTif = r"statics\data\im_random_tif\random.tiff"
    im2D = r"statics\data\im_random_tif\im2D.tif"
    for path in [im2D,randomTif, compositePath, rgbPath, grayPath, multiCHanPath]:
        imInfo =ImInfo(path)
        # description = {el.split("=")[0]:el.split("=")[1] for el in imInfo['ImageDescription'].split("\n")}
        # print(imInfo["BitsPerSample"], description["images"], description["slices"])
        # print(descriptiion['channels'])
        # print(imInfo)
        print(imInfo.name, imInfo.type, imInfo.channel)

    # print([(str(el),el.type.name) for el in ImageFolderInfos(Path(multiCHanPath).parent)])

