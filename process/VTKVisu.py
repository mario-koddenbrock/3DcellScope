import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkInteractionStyle
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import (vtkActor, vtkPolyDataMapper, vtkRenderer)
from OS_Window import OrganoSegmenterWindow
import numpy as np
import matplotlib.pyplot as plt
from vtk import *

my_window = OrganoSegmenterWindow()

class VTKvisu():
    def __init__(self):
        pass

    def numpyToVTK(data, voxel_size = (1, 1, 1)):
        if len(data.shape) == 3:
            data = np.reshape(data, (data.shape[2], data.shape[1], data.shape[0])) #from (z, y, x) to (x, y , z)
        elif len(data.shape) == 4:
            data = np.reshape(data, (data.shape[2], data.shape[1], data.shape[0], data.shape[3])) #from (z, y, x, c) to (x, y , z, c)
        data_type = VTK_FLOAT
        shape = data.shape

        flat_data_array = data.flatten()
        vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
        
        img = vtkImageData()
        img.GetPointData().SetScalars(vtk_data)
        img.SetDimensions(shape[0], shape[1], shape[2])
        img.SetSpacing(voxel_size[2], voxel_size[1], voxel_size[0])
        return img
    
    def setVTKImage(self) -> None:
        if self.ImageFrame.layout() is not None:
            layout = self.ImageFrame.layout()
        else :
            layout = QtWidgets.QVBoxLayout(self.ImageFrame)
            
        self.vtkWidget = QVTKRenderWindowInteractor(self.ImageFrame)
        layout.addWidget(self.vtkWidget)

        self.ren = vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        
    def addVTKImage(self, actor = None, volume = None):
        if actor is not None:
            self.ren.AddActor(actor)
        elif volume is not None:
            self.ren.AddVolume(volume) 
   
    def finishVisuVTK(self):
        self.ren.ResetCamera()
        self.show()
        self.iren.Initialize()   