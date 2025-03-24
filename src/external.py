
#SYSTEM - PYTHON 3.8.10 -
from pathlib import Path 
# import pickle 
# import shutil 
# import re 
import time 
# import copy 
import webbrowser 
import tempfile 
# import random 
# import math 
# import base64
from io import BytesIO
# import threading
# import multiprocessing as mp
# import platform
import sys
import tqdm
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass

#Pyside
from PySide6 import QtWidgets
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6 import QtSvg
from PySide6 import QtSvgWidgets

# #SCIPY
# from scipy.ndimage import maximum_filter 
from scipy.stats import ttest_ind, mannwhitneyu
import scipy.cluster.hierarchy as spc
from scipy.stats import f_oneway
from scipy.spatial.distance import squareform
# import scipy.ndimage as ndi

# #SKIMAGE 
# from skimage.transform import rescale 
# from skimage.segmentation import find_boundaries, watershed
# from skimage.measure import regionprops, regionprops_table, label, shannon_entropy, euler_number
# from skimage.filters import gaussian, laplace
# from skimage.filters.rank import median
# from skimage.morphology import remove_small_objects, skeletonize, closing, disk
# from skimage.feature import graycomatrix, graycoprops

#SKLEARN 
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as sklearn_LDA
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.cluster import AgglomerativeClustering

#MATPLOTLIB 
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.cm import hsv
from matplotlib import cm as colormap
import matplotlib.ticker as mticker
import matplotlib.patheffects as path_effects 

#Others 
import pandas as pd # BSD 3-Clause "New" or "Revised" License  - https://github.com/pandas-dev/pandas/blob/ac648eeaf5c27ab957e8cd284eb7e49a45232f00/LICENSE
import numpy as np # BSD 3-Clause "New" or "Revised" License - https://numpy.org/doc/stable/license.html

# import PySimpleGUI as sg # GNU Lesser General Public License v3.0 - https://github.com/PySimpleGUI/PySimpleGUI/blob/5fe61b7499f6e64bd4a5e3c271f7b644d6030210/license.txt
from PIL import Image # Open source HPND License - https://github.com/python-pillow/Pillow/blob/1d1a22bde37baaf162f4dd26e7b94cd96d3116a2/LICENSE
from tifffile import imwrite # BSD 3-Clause "New" or "Revised" License - https://github.com/cgohlke/tifffile/blob/eb79f64e65fdd17fa5722dac9ce32ec6dd224f5d/LICENSE
from imageio import imread # - BSD 2-Clause "Simplified" License - https://github.com/imageio/imageio/blob/c3e83ce24a366f849471922f5a76f1fa1bf94535/LICENSE
# import ctypes # https://github.com/Legrandin/ctypes/blob/f47a350a17cb2319695e71bf444a960946dba68c/LICENSE.txt

# from bs4 import BeautifulSoup #- MIT License - https://github.com/akalongman/python-beautifulsoup/blob/95e760c75e517226f668901ca9c83401c131d94c/LICENSE
# from PyPDF4 import PdfFileMerger #- BSD 3-Clause "New" or "Revised" License - https://github.com/claird/PyPDF4/blob/9c60d9df3a56edd32226c9e76695018f997fafe6/LICENSE.md
