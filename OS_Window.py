from OS_Ui import Ui_OrganoSegmenterWindow
from OS_Popups import Ui_Dialog,Ui_VolumeWindow, Ui_3dParametersWindow, ProgressBarWindow
#Pyside
from PySide6 import QtWidgets
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtUiTools import QUiLoader
from PySide6 import QtSvg
from PySide6 import QtSvgWidgets
import numpy as np
import matplotlib
from matplotlib import style
import  matplotlib.backends.qt_compat as qt_compat
qt_compat.QT_API_ENV = qt_compat.QT_API_PYSIDE6
matplotlib.use("qt5agg")
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_tools import Cursors
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.backends.backend_svg
import pandas as pd

import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkInteractionStyle
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import (vtkActor, vtkPolyDataMapper, vtkRenderer)

class OrganoSegmenterWindow(QMainWindow, Ui_OrganoSegmenterWindow):
    def __init__(self):
        super(OrganoSegmenterWindow,self).__init__()
        super(Ui_OrganoSegmenterWindow,self).__init__()
        self.setupUi(self)
        self.app_version = 'XXX'
        try:
            with open('file_version_info.txt', 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if 'ProductVersion' in line:
                        self.app_version = line.split("\'ProductVersion\',")[1].split(' ')[1].split('\'')[1]
        except:
            self.app_version = 'XXX'
        self.setWindowTitle(f'OrganoSegmenter (version {self.app_version}) - Quantacell - ')
        self.setImageWindows()
        self.setPlotWindows()
        
        #add informations
        self.BrowseParametersButton.setToolTip("Load json file to update parameters")
        self.InfoButton.setToolTip("Open user manual")
        self.BrowseFolderButton.setToolTip("Load a folder containing 3D images")
        self.ChangeModelButton.setToolTip("Load a custom model folder")
        self.AddPreprocessingButton.setToolTip("Add this preprocessing")
        self.RemovePreprocessingButton.setToolTip("Remove selected preprocessing")
        self.AutoResizeWindowButton.setToolTip("Open window to get resample values from voxel sizes")
        self.View3DPushButton.setToolTip("Open window to set 3d view custom settings")
        
        self.spacingX = None
        self.spacingY = None
        self.spacingZ = None
        self.opacityValueSlider = None
        self.ValuesList = None
    
    def open3dParametersWindow(self, FileInfo, spacingX, spacingY, spacingZ, opacityValueSlider=None, ValuesList=None):
        popup = Ui_3dParametersWindow(FileInfo, spacingX, spacingY, spacingZ, self.opacityValueSlider, self.ValuesList)
        popup.setModal(True) 
        popup.exec_()
        
        self.spacingX = popup.spacingX
        self.spacingY = popup.spacingY
        self.spacingZ = popup.spacingZ
        self.opacityValueSlider = popup.opacityValueSlider
        self.ValuesList = popup.ValuesList
        
    def openAutoResizeWindow(self):
        popup = Ui_Dialog((self.DeltaXLineEdit.text(), self.DeltaYLineEdit.text(), self.DeltaZLineEdit.text()), self.DefaultModelComboBox.currentText())
        popup.setModal(True) 
        popup.exec_()

        outputResize = popup.outputResize
        if len(outputResize) < 3:
            return
        self.ResampleXLineEdit.setText(str(outputResize[0]))
        self.ResampleYLineEdit.setText(str(outputResize[1]))
        self.ResampleZLineEdit.setText(str(outputResize[2]))
        self.ResampleXLineEdit.setCursorPosition(0)
        self.ResampleYLineEdit.setCursorPosition(0)
        self.ResampleZLineEdit.setCursorPosition(0)
      
        
    def openVolumeWindow(self, montageNuclei, maskNuclei, dt):
        popup = Ui_VolumeWindow(montageNuclei, maskNuclei, dt)
        popup.setModal(True) 
        popup.exec_()
        minVolume = popup.MinVolumeValueLabel.text()
        maxVolume = popup.MaxVolumeValueLabel.text()
        if (len(minVolume)>0) and minVolume.isdigit():
            self.VolumeMinLineEdit.setText(minVolume)
            self.VolumeMinLineEdit.setCursorPosition(0)
        if (len(maxVolume)>0) and maxVolume.isdigit():
            self.VolumeMaxLineEdit.setText(maxVolume)
            self.VolumeMaxLineEdit.setCursorPosition(0)
    
    def setImageWindows(self)->None:
        if self.ImageFrame.layout() is not None:
            layout = self.ImageFrame.layout()
        else :
            layout = QtWidgets.QVBoxLayout(self.ImageFrame)
            
        self.image_fig, self.image_ax = plt.subplots(figsize=(8,8))
        self.image_ax.axis("off")
        self.image_fig.tight_layout()
        self.image_fig.set_facecolor((24/255, 24/255, 24/255))
        static_canvas = FigureCanvas(self.image_fig)
        navbar = NavigationToolbar(static_canvas, parent= None)
        ### change icons toolbar matplotlib
        icons_buttons = {
            "Home": QIcon("./logos/toolbar/home-svgrepo-com.svg"),
            "Back" : QIcon("./logos/toolbar/back-svgrepo-com.svg"),
            "Forward" : QIcon("./logos/toolbar/forward-svgrepo-com.svg"),
            "move": QIcon("./logos/toolbar/move-alt-svgrepo-com.svg"),
            "Zoom": QIcon("./logos/toolbar/zoom-svgrepo-com.svg"),
            "Subplots" : QIcon("./logos/toolbar/sliders-svgrepo-com.svg"),
            "Customize" : QIcon("./logos/toolbar/chart-line-up-duotone-svgrepo-com.svg"), 
            "Save": QIcon("./logos/toolbar/save-svgrepo-com.svg"),
            "Pan": QIcon("./logos/toolbar/move-alt-svgrepo-com.svg"),
        }
        for action in navbar.actions():
            if action.text() in icons_buttons:
                action.setIcon(icons_buttons.get(action.text(), QIcon()))
        ###
        navbar.update()
        layout.addWidget(navbar)
        layout.addWidget(static_canvas)
        self.image_canvas = static_canvas
        self.image_navbar = navbar
        
    def clearImageFrame(self) -> None:
        for i in reversed(range(self.ImageFrame.layout().count())):
            widgetToRemove = self.ImageFrame.layout().itemAt(i).widget()
            if widgetToRemove is not None:
                # widgetToRemove.setParent(None)
                widgetToRemove.deleteLater()
            else:
                self.clearImageFrame(self.ImageFrame.layout().itemAt(i).layout())

    def clear3DWindow(self):
        self.ren.RemoveAllViewProps()
        self.vtkWidget.GetRenderWindow().Finalize()


        
    def setVTKImage(self) -> None:
        if self.ImageFrame.layout() is not None:
            layout = self.ImageFrame.layout()
        else :
            layout = QtWidgets.QVBoxLayout(self.ImageFrame)
            
        self.vtkWidget = QVTKRenderWindowInteractor(self.ImageFrame)
        layout.addWidget(self.vtkWidget)

        self.ren = vtkRenderer()
        
        #self.ren.SetBackground(255 / 255.0, 255 / 255.0, 255 / 255.0) #white background
        
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
           
    def setPlotWindows(self)->None:
        layout = QtWidgets.QVBoxLayout(self.PlotFrame)
        self.plot_fig, self.plot_ax = plt.subplots(figsize=(8,8))
        self.plot_ax.axis("on")
        ##
        self.plot_ax.tick_params(axis='x', colors='white')    
        self.plot_ax.tick_params(axis='y', colors='white') 

        self.plot_ax.spines['left'].set_color('white')  
        self.plot_ax.spines['right'].set_color('white')       
        self.plot_ax.spines['top'].set_color('white')  
        self.plot_ax.spines['bottom'].set_color('white')       
        ##
        # self.plot_fig.tight_layout()
        self.plot_fig.set_facecolor((24/255, 24/255, 24/255))
        static_canvas = FigureCanvas(self.plot_fig)
        navbar = NavigationToolbar(static_canvas, parent= None)
        ### change icons toolbar matplotlib
        icons_buttons = {
            "Home": QIcon("./logos/toolbar/home-svgrepo-com.svg"),
            "Back" : QIcon("./logos/toolbar/back-svgrepo-com.svg"),
            "Forward" : QIcon("./logos/toolbar/forward-svgrepo-com.svg"),
            "move": QIcon("./logos/toolbar/move-alt-svgrepo-com.svg"),
            "Zoom": QIcon("./logos/toolbar/zoom-svgrepo-com.svg"),
            "Subplots" : QIcon("./logos/toolbar/sliders-svgrepo-com.svg"),
            "Customize" : QIcon("./logos/toolbar/chart-line-up-duotone-svgrepo-com.svg"), 
            "Save": QIcon("./logos/toolbar/save-svgrepo-com.svg"),
            "Pan": QIcon("./logos/toolbar/move-alt-svgrepo-com.svg"),
        }
        for action in navbar.actions():
            if action.text() in icons_buttons:
                action.setIcon(icons_buttons.get(action.text(), QIcon()))
        ###
        navbar.update()
        layout.addWidget(navbar)
        layout.addWidget(static_canvas)
        self.plot_canvas = static_canvas
        self.plot_navbar = navbar

    def setFileList(self, fileListNames):
        self.FileListWidget.clear()  
        self.FileListWidget.addItems(fileListNames)
    
    def setChannelsComboBox(self, FileInfo):
        self.ChannelVisualisationComboBox.setEnabled(True)
        ComboBoxListAll = [self.ChannelOrganoidComboBox,self.ChannelVisualisationComboBox]
        ComboBoxList = [self.ChannelPreprocessingComboBox, self.ChannelNucleiComboBox, 
                        self.ChannelCellsComboBox]
        channelsList = [str(i+1) for i in range(FileInfo.channel)]
        for obj in ComboBoxList:
            obj.clear()
            obj.addItems(channelsList)
        for obj2 in ComboBoxListAll:
            obj2.clear()
            obj2.addItems(['all'] + channelsList)
            
    def setMasksComboBox(self, dictPredicted):
        if dictPredicted is not None:
            self.MaskVisualisationComboBox.setEnabled(True)
            maskList = ["Nuclei"]
            if dictPredicted["dataframes"]["dt_cell"] is not None:
                maskList.append("Cells")
                maskList.append("Nuclei + Cells")
            if dictPredicted["images_paths"]["mask_organo"] is not None : 
                if dictPredicted["images_paths"]["mask_organo"] != []:
                    maskList.append("Organoid")
            maskList.append("None")
        else:
            self.MaskVisualisationComboBox.setEnabled(False)
            maskList= []
        self.MaskVisualisationComboBox.clear()
        self.MaskVisualisationComboBox.addItems(maskList)
            
    def setVoxelSizes(self, FileInfo):
        self.DeltaXLineEdit.clear()
        self.DeltaYLineEdit.clear()
        self.DeltaZLineEdit.clear()
        self.DeltaXLineEdit.setText(str(FileInfo.resolution[2]))
        self.DeltaYLineEdit.setText(str(FileInfo.resolution[1]))
        self.DeltaZLineEdit.setText(str(FileInfo.resolution[0]))
        self.DeltaXLineEdit.setCursorPosition(0)
        self.DeltaYLineEdit.setCursorPosition(0)
        self.DeltaZLineEdit.setCursorPosition(0)
    
    def setSliderVisualisation(self, zShape):
        self.ImageViewerSlider.setMaximum(zShape)
        self.ImageViewerSlider.setValue((zShape)//2)
        self.ZValueLabel.setText(str((zShape)//2))
        
    def showImage(self, selectedImage, layer = None, z= None, channel = None):
        if z is None:
            z = self.ImageViewerSlider.value()
        if channel is None:
            channel = self.ChannelVisualisationComboBox.currentText()
        if channel != "all":
            ch = int(channel) - 1
        else:
            ch = None
        self.image_ax.clear()
        im_np = selectedImage[z] if ch is None else selectedImage[z,:,:,ch]
        try:
            self.image_ax.imshow(im_np.astype('uint8'))
        except TypeError:
            self.image_ax.imshow(im_np[..., :3].astype('uint8'))
        if layer is not None:
            self.image_ax.imshow(layer[z].astype('uint8'))
        self.image_ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)      
        self.image_canvas.draw_idle()
        self.image_navbar._update_view()
        self.PreprocessingCheckBox.setEnabled(True)
    
    def showPlot(self, dt, columnName = None, typeGraph = None):
        if dt is not None:
            self.plot_ax.clear()
            self.plot_ax.axis("on")
            fnames_no_ext=list(pd.Index(dt['file']).unique())
            if typeGraph == "Density repartition" and (len(fnames_no_ext)<0.5*len(dt)):
                for name in fnames_no_ext:
                    dt_file = dt[dt['file'] == name]
                    s = pd.Series(dt_file[columnName])
                    s.plot.kde(ax = self.plot_ax)
                    self.plot_ax.set_xlabel(columnName, color=(250/255, 250/255, 250/255))
                    self.plot_ax.set_ylabel('density repartition', color=(250/255, 250/255, 250/255))
                self.plot_ax.legend(fnames_no_ext, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='xx-small')
                # self.plot_ax.subplots_adjust(right=0.7)
            elif typeGraph == "Boxplot":
                for name in fnames_no_ext:
                    dt_file = dt[dt['file'] == name]
                    self.plot_ax.boxplot(dt_file[columnName], positions = [fnames_no_ext.index(name)])
                    self.plot_ax.set_ylabel(columnName, color=(250/255, 250/255, 250/255))
                self.plot_ax.set_xticklabels(fnames_no_ext, color=(250/255, 250/255, 250/255))
                
            # plt.subplots_adjust(left=0.3, right=1, top=1, bottom=0.3)
            self.plot_fig.tight_layout()  
            self.plot_canvas.draw_idle()
            self.plot_navbar._update_view()
        else :
            self.plot_ax.clear()


    def getPreprocessingParam(self):
        channelPreprocessing = self.ChannelPreprocessingComboBox.currentText()
        methodPreprocessing = self.MethodPreprocessingComboBox.currentText()
        parameter1 = None
        parameter2 = None
        if methodPreprocessing in ["gaussian difference", "gaussian division"]:
            parameter1 = self.Parameter1PreprocessingLineEdit.text()
            try:
                int(parameter1)
            except ValueError:
                QMessageBox.critical(self, "Parameter Value Warning", "Please enter a sigma (integer value) to add this preprocessing", QMessageBox.StandardButton.Ok)
                return (None, None, None, None)
        elif methodPreprocessing == "unsharp mask" : 
            parameter1 = self.Parameter1PreprocessingLineEdit.text()
            parameter2 = self.Parameter2PreprocessingLineEdit.text()
            try:
                int(parameter1)
                assert 0 <= float(parameter2) <= 1
            except :
                QMessageBox.critical(self, "Parameters Values Warning", "Please enter a radius (integer value) and an amount(between 0 and 1) to add this preprocessing", QMessageBox.StandardButton.Ok)
                return (None, None, None, None)
        return (channelPreprocessing, methodPreprocessing, parameter1, parameter2)
    
    def setPreprocessingItem(self, item):
        self.PreprocessingListWidget.addItem(item)
        
    def setStatisticsTableInit(self, dictPredicted):   
        if dictPredicted is not None:
            itemsList = ["Nuclei"] 
            if dictPredicted["dataframes"]["dt_cell"] is not None:
                itemsList.append("Cells")
                itemsList.append("Cytoplasm")
            if dictPredicted["images_paths"]["mask_organo"] is not None : 
                if dictPredicted["images_paths"]["mask_organo"] != []:
                    itemsList.append("Organoid")
        else:
            itemsList= []
        self.TableComboBox.clear()
        self.TableComboBox.addItems(itemsList)
            
    def getPreprocessingList(self):
        nPreproc = self.PreprocessingListWidget.count()
        list_preprocessings = ['' for i in range(nPreproc)]
        for index in range(nPreproc):
            item_text = self.PreprocessingListWidget.item(index).text()
            list_preprocessings[index] = item_text
        return list_preprocessings
    

    def getFolderPath(self):
        return self.ImageFolder.toPlainText()
    
    def getExportParam(self): 
        return {
            "exportChannels" : self.ExportChannelsCheckBox.isChecked(),
            "exportOverlays" : self.ExportOverlaysCheckBox.isChecked(),
            "exportParameters" : self.ExportParametersCheckBox.isChecked(),
            "exportComposite" :  self.ExportCompositeCheckBox.isChecked()   
        }
    
    def getNucleiParam(self):
        channelNuclei = self.ChannelNucleiComboBox.currentText()
        modelPath = self.ChangeModelTextBrowser.toPlainText()
        defaultModel = self.DefaultModelComboBox.currentText()
        resizeX = self.ResampleXLineEdit.text()
        resizeY = self.ResampleYLineEdit.text()
        resizeZ = self.ResampleZLineEdit.text()
        NmsThreshold = self.NmsSlider.value() / 10
        ProbThreshold = self.ProbSlider.value() / 10
        volumeMin = self.VolumeMinLineEdit.text()
        volumeMax = self.VolumeMaxLineEdit.text()
        return {"channelNuclei" : channelNuclei,
        "modelPath" : modelPath,
        "defaultModel" : defaultModel,
        "resizeX" : resizeX,
        "resizeY" : resizeY,
        "resizeZ" : resizeZ,
        "NmsThreshold" : NmsThreshold,
        "ProbThreshold" : ProbThreshold,
        "volumeMin" : volumeMin,
        "volumeMax" : volumeMax}

    def getCellsParam(self):
        doCells = self.DetectCellsCheckBox.isChecked()
        channelCells = self.ChannelCellsComboBox.currentText()
        methodCells = self.MethodCellsComboBox.currentText()
        cellStains = self.cellStainsRadioButton.isChecked()
        cytoStains = self.cytoStainsRadioButton.isChecked()
        distMax = self.DistMaxLineEdit.text()
        radius = self.RadiusLineEdit.text()
        return {"doCells" : doCells,
                "channelCells" : channelCells,
                "methodCells" : methodCells,
                "cellStains" : cellStains,
                "cytoStains" : cytoStains,
                "distMax" : distMax, 
                "radius" : radius}
    
    def getOrganoidParam(self):
        return {"doOrganoid" : self.DetectOrganoidCheckBox.isChecked(), 
                "channelOrganoid" : self.ChannelOrganoidComboBox.currentText(),
                "methodOrganoid" : self.MethodOrganoidComboBox.currentText(),
                "parameterOrganoid" : float(self.Parameter1OrganoidValueLabel.text()), 
                "keepLargestOrganoid" : self.KeepLargestOrganoidRadioButton.isChecked(),
                "keepMultipleOrganoids" : self.KeepMultipleOragnoidRadioButton.isChecked(),
                "volumeMinOrganoid" : self.VolumeMinOrganoidLineEdit.text() if self.KeepMultipleOragnoidRadioButton.isChecked() else None
        }
    
    def getImageParameters(self):
        return{"dx" : self.DeltaXLineEdit.text(),
               "dy" : self.DeltaYLineEdit.text(),
               "dz" : self.DeltaZLineEdit.text()}
    
    def getImageClickEventPosition(self, event : QMouseEvent):
        z = self.ImageViewerSlider.value()
        x = int(event.xdata if event.xdata is not None else -1)
        y = int(event.ydata if event.ydata is not None else -1)
        return (z,x,y)
    
    def setAllParameters(self, data):
        try:
            NucleiParameters = data["NucleiParameters"]
            CellsParameters = data["CellsParameters"]
            OrganoidParameters = data["OrganoidParameters"]
            ImageParameters = data["ImageParameters"]
            PreprocessingParameters = data["PreprocessingParameters"]
            ExportParameters = data["ExportParameters"]
        except:
            pass   
        self.PreprocessingListWidget.clear()
        self.PreprocessingListWidget.addItems(PreprocessingParameters["listPreprocessing"])
        
        self.ChannelNucleiComboBox.setCurrentText(NucleiParameters["channelNuclei"])
        self.ChangeModelTextBrowser.setPlainText(NucleiParameters["modelPath"])
        self.DefaultModelComboBox.setCurrentText(NucleiParameters["defaultModel"])
        self.ResampleXLineEdit.setText(NucleiParameters["resizeX"])
        self.ResampleYLineEdit.setText(NucleiParameters["resizeY"])
        self.ResampleZLineEdit.setText(NucleiParameters["resizeZ"])
        self.ResampleXLineEdit.setCursorPosition(0)
        self.ResampleYLineEdit.setCursorPosition(0)
        self.ResampleZLineEdit.setCursorPosition(0)
        self.NmsSlider.setValue(int(NucleiParameters["NmsThreshold"] * 10))
        self.ProbSlider.setValue(int(NucleiParameters["ProbThreshold"] * 10))
        self.VolumeMinLineEdit.setText(NucleiParameters["volumeMin"])
        self.VolumeMaxLineEdit.setText(NucleiParameters["volumeMax"])
        self.VolumeMinLineEdit.setCursorPosition(0)
        self.VolumeMaxLineEdit.setCursorPosition(0)
        
        self.DetectCellsCheckBox.setChecked(CellsParameters["doCells"])
        self.ChannelCellsComboBox.setCurrentText(CellsParameters["channelCells"])
        self.MethodCellsComboBox.setCurrentText(CellsParameters["methodCells"])
        try:
            self.cellStainsRadioButton.setChecked(CellsParameters["cellStains"])
            self.cytoStainsRadioButton.setChecked(CellsParameters["cytoStains"])
        except:
            pass
        self.DistMaxLineEdit.setText(CellsParameters["distMax"])
        self.RadiusLineEdit.setText(CellsParameters["radius"])
        self.DistMaxLineEdit.setCursorPosition(0)
        self.RadiusLineEdit.setCursorPosition(0)
        
        self.DetectOrganoidCheckBox.setChecked(OrganoidParameters["doOrganoid"])
        self.ChannelOrganoidComboBox.setCurrentText(OrganoidParameters["channelOrganoid"])
        self.MethodOrganoidComboBox.setCurrentText(OrganoidParameters["methodOrganoid"])
        self.Parameter1OrganoidSlider.setValue(OrganoidParameters["parameterOrganoid"])
        self.KeepLargestOrganoidRadioButton.setChecked(OrganoidParameters["keepLargestOrganoid"])
        self.KeepMultipleOragnoidRadioButton.setChecked(OrganoidParameters["keepMultipleOrganoids"])
        self.VolumeMinOrganoidLineEdit.setText(OrganoidParameters["volumeMinOrganoid"])
        self.VolumeMinOrganoidLineEdit.setCursorPosition(0)

        
        self.DeltaXLineEdit.setText(ImageParameters["dx"])
        self.DeltaYLineEdit.setText(ImageParameters["dy"])
        self.DeltaZLineEdit.setText(ImageParameters["dz"])
        self.DeltaXLineEdit.setCursorPosition(0)
        self.DeltaYLineEdit.setCursorPosition(0)
        self.DeltaZLineEdit.setCursorPosition(0)
        
        self.ExportChannelsCheckBox.setChecked(ExportParameters["exportChannels"])
        self.ExportOverlaysCheckBox.setChecked(ExportParameters["exportOverlays"])
        self.ExportParametersCheckBox.setChecked(ExportParameters["exportParameters"])
        self.ExportCompositeCheckBox.setChecked(ExportParameters["exportComposite"])   
        
    