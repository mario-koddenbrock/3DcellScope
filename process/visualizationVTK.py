import vtk
from tifffile import imread
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import sys
import vtkmodules.util.numpy_support as numpy_support
import numpy as np
import matplotlib.pyplot as plt
import colorsys

def numpyToVTK(data, spacingX=1, spacingY=1, spacingZ=1):
    if len(data.shape) == 3:
        data = np.reshape(data, (data.shape[2], data.shape[1], data.shape[0])) #from (z, y, x) to (x, y , z)
    elif len(data.shape) == 4:
        data = np.reshape(data, (data.shape[2], data.shape[1], data.shape[0], data.shape[3])) #from (z, y, x, c) to (x, y , z, c)
    data_type = vtk.VTK_FLOAT
    shape = data.shape

    flat_data_array = data.flatten()
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
    
    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetDimensions(shape[0], shape[1], shape[2])
    img.SetSpacing(spacingX, spacingY, spacingZ)
    return img

def getSurface(image_data, max_value = 151, opacity = 1, transparency = 0.0):
    # Create vtkDiscreteMarchingCubes
    discrete_marching_cubes = vtk.vtkDiscreteMarchingCubes()
    discrete_marching_cubes.SetInputData(image_data)
    discrete_marching_cubes.GenerateValues(max_value, 1, max_value)

    # Smooth the surface using vtkWindowedSincPolyDataFilter
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(discrete_marching_cubes.GetOutputPort())
    # Set smoothing parameters if needed
    smoother.SetNumberOfIterations(15)
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(120.0)
    smoother.SetPassBand(0.01)
                    
    smoother.Update()

    # Create vtkOpenGLPolyDataMapper
    mapper = vtk.vtkOpenGLPolyDataMapper()
    mapper.SetInputConnection(smoother.GetOutputPort())

    lut = make_colors_random(max_value) #CreateLookupTableVTKWhite(151)
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(0, lut.GetNumberOfColors())  # Assuming you want scalar values in [0, 1]

    # Create vtkActor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)

    return actor

def getVolumes(image_data, min_value = 50, max_value = 200, transparency = 1.0, color = (0,0,255)):
    r,g,b = color[0], color[1], color[2]
    # Appliquer un seuil aux scalaires des données
    # threshold = vtk.vtkImageThreshold()
    # threshold.SetInputData(image_data)
    # threshold.ThresholdBetween(0, 100)  # Appliquer le seuil d'intensité
    # threshold.Update()
    
    mapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
    mapper.SetInputData(image_data)
    
    volume = vtk.vtkVolume()

    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(min_value, 0.0)
    opacityTransferFunction.AddPoint(max_value, transparency)

    colorTransferFunction = vtk.vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(min_value, 0.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(min_value + (max_value - min_value) / 2.0, r / 255.0, g / 255.0, b / 255.0)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.SetInterpolationTypeToLinear()

    volume.SetMapper(mapper)
    volume.SetProperty(volumeProperty)
        
    return volume 

class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.frame = QFrame()

        self.vl = QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        # image_data = numpyToVTK(imread("D:/Elise/image_bank/OsmoticShock/A1_ctrl_s1_nms0.3_prob0.3_resizeX0.5179005059266462Y0.5179005059266462Z0.5179005059266462/A1_ctrl_s1_nuclei-mask.tif"))
        ###
        image_data2 = numpyToVTK(imread("D:/Elise/image_bank/OsmoticShock/A1_ctrl_C1_s1.tif"))
        image_data3 = numpyToVTK(imread("D:/Elise/image_bank/OsmoticShock/A1_ctrl_C2_s1.tif"))
        volume1 = getVolumes(image_data2,140,4095,1,(255,0,0))
        # volume2 = getVolumes(image_data3, 140,4095,1,0,255,0)
        self.ren.AddVolume(volume1)
        # self.ren.AddVolume(volume2)
        ###
        # actor = getSurface(image_data)
        
        # self.ren.AddActor(actor)

        self.ren.ResetCamera()

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.show()
        self.iren.Initialize()

def make_vivid_hsv_lut_vtk(n):
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n)
    lut.Build()

    for i in range(n):
        hue = i / n
        saturation = 1.0
        value = 1.0

        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        r, g, b = [int(c * 255) for c in rgb]
        lut.SetTableValue(i, r / 255.0, g / 255.0, b / 255.0, 1.0)

    return lut

def make_colors_random(n, transparency=1.0):
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfColors(n + 1)
    lut.SetTableRange(0, n)
    lut.SetScaleToLinear()
    lut.Build()
    lut.SetTableValue(0, 0, 0, 0, transparency)

    randomSequence = vtk.vtkMinimalStandardRandomSequence()
    randomSequence.SetSeed(4355412)
    for i in range(1, n + 1):
        r = randomSequence.GetRangeValue(0.4, 1.0)
        randomSequence.Next()
        g = randomSequence.GetRangeValue(0.4, 1.0)
        randomSequence.Next()
        b = randomSequence.GetRangeValue(0.4, 1.0)
        randomSequence.Next()
        lut.SetTableValue(i, r, g, b, 1.0)

    return lut

if __name__ == "__main__":
    
    app = QApplication(sys.argv)

    window = MainWindow()

    sys.exit(app.exec_())
