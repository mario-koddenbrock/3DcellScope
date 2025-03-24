# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'autoResizeWindowGIEsKR.ui'
##
## Created by: Qt User Interface Compiler version 6.5.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################
import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect, QTimer, QRectF,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QMouseEvent, QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter, QPainterPath, QPen,
    QPalette, QPixmap, QRadialGradient, QTransform,QStandardItemModel, QStandardItem)
from PySide6.QtWidgets import (QApplication, QMainWindow, QProgressBar, QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QListWidget, QListWidgetItem, QPushButton, QSizePolicy, QWidget, QFrame, QSlider, QToolBox, QVBoxLayout, QSpacerItem,QTableView, QHeaderView)
# from cryptlex.lexactivator import LexActivator, LexStatusCodes, PermissionFlags, LexActivatorException

class Ui_Dialog(QDialog):
    def __init__(self, InitialResizeTuple, defaultModel):
        super(Ui_Dialog,self).__init__()
        self.setupUi(self)
        self.InputImageXLineEdit.setText(InitialResizeTuple[0])
        self.InputImageYLineEdit.setText(InitialResizeTuple[1])
        self.InputImageZLineEdit.setText(InitialResizeTuple[2])
        if defaultModel == "Default Model":
            self.TrainingDataXLineEdit.setText("0.8")
            self.TrainingDataYLineEdit.setText("0.8")
            self.TrainingDataZLineEdit.setText("1")
        elif defaultModel == "Big Nuclei":
            self.TrainingDataXLineEdit.setText("0.65")
            self.TrainingDataYLineEdit.setText("0.65")
            self.TrainingDataZLineEdit.setText("1.2")
        self.InputImageXLineEdit.setCursorPosition(0)
        self.InputImageYLineEdit.setCursorPosition(0)
        self.InputImageZLineEdit.setCursorPosition(0)
        self.ApplyAutoResizeButton.clicked.connect(self.ComputeValues)
        self.outputResize = []

    def ComputeValues(self):
        VS_trainX = float(self.TrainingDataXLineEdit.text())
        VS_trainY = float(self.TrainingDataYLineEdit.text())
        VS_trainZ = float(self.TrainingDataZLineEdit.text())
        
        VS_testX = float(self.InputImageXLineEdit.text())
        VS_testY = float(self.InputImageYLineEdit.text())
        VS_testZ = float(self.InputImageZLineEdit.text())
        
        resize_valueX = float(VS_testX/VS_trainX)
        resize_valueY = float(VS_testY/VS_trainY)
        resize_valueZ = float(VS_testZ/VS_trainZ)

        self.ResampleXValueLabel.setText(str(resize_valueX))
        self.ResampleYValueLabel.setText(str(resize_valueY))
        self.ResampleZValueLabel.setText(str(resize_valueZ))
        
        self.outputResize = [resize_valueX, resize_valueY, resize_valueZ]
        
        
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(440, 236)
        icon = QIcon()
        icon.addFile(u"OSlogo.ico", QSize(), QIcon.Normal, QIcon.Off)
        Dialog.setWindowIcon(icon)
        self.InputImageYLabel = QLabel(Dialog)
        self.InputImageYLabel.setObjectName(u"InputImageYLabel")
        self.InputImageYLabel.setGeometry(QRect(250, 58, 16, 22))
        self.InputImageZLineEdit = QLineEdit(Dialog)
        self.InputImageZLineEdit.setObjectName(u"InputImageZLineEdit")
        self.InputImageZLineEdit.setGeometry(QRect(370, 58, 55, 22))
        self.ApplyAutoResizeButton = QPushButton(Dialog)
        self.ApplyAutoResizeButton.setObjectName(u"ApplyAutoResizeButton")
        self.ApplyAutoResizeButton.setGeometry(QRect(20, 190, 161, 31))
        self.InputImageZLabel = QLabel(Dialog)
        self.InputImageZLabel.setObjectName(u"InputImageZLabel")
        self.InputImageZLabel.setGeometry(QRect(350, 58, 16, 22))
        self.ResampleXLabel = QLabel(Dialog)
        self.ResampleXLabel.setObjectName(u"ResampleXLabel")
        self.ResampleXLabel.setGeometry(QRect(20, 100, 71, 16))
        self.ResampleYLabel = QLabel(Dialog)
        self.ResampleYLabel.setObjectName(u"ResampleYLabel")
        self.ResampleYLabel.setGeometry(QRect(20, 130, 71, 16))
        self.TrainingDataZLabel = QLabel(Dialog)
        self.TrainingDataZLabel.setObjectName(u"TrainingDataZLabel")
        self.TrainingDataZLabel.setGeometry(QRect(350, 18, 16, 22))
        self.TrainingDataZLineEdit = QLineEdit(Dialog)
        self.TrainingDataZLineEdit.setObjectName(u"TrainingDataZLineEdit")
        self.TrainingDataZLineEdit.setGeometry(QRect(370, 18, 55, 22))
        self.InputImageYLineEdit = QLineEdit(Dialog)
        self.InputImageYLineEdit.setObjectName(u"InputImageYLineEdit")
        self.InputImageYLineEdit.setGeometry(QRect(270, 58, 55, 22))
        self.TrainingDataXLineEdit = QLineEdit(Dialog)
        self.TrainingDataXLineEdit.setObjectName(u"TrainingDataXLineEdit")
        self.TrainingDataXLineEdit.setGeometry(QRect(180, 18, 55, 22))
        self.InputImageXLineEdit = QLineEdit(Dialog)
        self.InputImageXLineEdit.setObjectName(u"InputImageXLineEdit")
        self.InputImageXLineEdit.setGeometry(QRect(180, 58, 55, 22))
        self.InputImageXLabel = QLabel(Dialog)
        self.InputImageXLabel.setObjectName(u"InputImageXLabel")
        self.InputImageXLabel.setGeometry(QRect(160, 58, 16, 22))
        self.ResampleZLabel = QLabel(Dialog)
        self.ResampleZLabel.setObjectName(u"ResampleZLabel")
        self.ResampleZLabel.setGeometry(QRect(20, 160, 71, 16))
        self.TrainingDataXLabel = QLabel(Dialog)
        self.TrainingDataXLabel.setObjectName(u"TrainingDataXLabel")
        self.TrainingDataXLabel.setGeometry(QRect(160, 18, 16, 22))
        self.TrainingDataYLineEdit = QLineEdit(Dialog)
        self.TrainingDataYLineEdit.setObjectName(u"TrainingDataYLineEdit")
        self.TrainingDataYLineEdit.setGeometry(QRect(270, 18, 55, 22))
        self.TrainingDataLabel = QLabel(Dialog)
        self.TrainingDataLabel.setObjectName(u"TrainingDataLabel")
        self.TrainingDataLabel.setGeometry(QRect(20, 20, 141, 16))
        self.TrainingDataYLabel = QLabel(Dialog)
        self.TrainingDataYLabel.setObjectName(u"TrainingDataYLabel")
        self.TrainingDataYLabel.setGeometry(QRect(250, 18, 16, 22))
        self.InputImageLabel = QLabel(Dialog)
        self.InputImageLabel.setObjectName(u"InputImageLabel")
        self.InputImageLabel.setGeometry(QRect(20, 60, 141, 16))
        self.ResampleXValueLabel = QLabel(Dialog)
        self.ResampleXValueLabel.setObjectName(u"ResampleXValueLabel")
        self.ResampleXValueLabel.setGeometry(QRect(110, 100, 311, 16))
        self.ResampleYValueLabel = QLabel(Dialog)
        self.ResampleYValueLabel.setObjectName(u"ResampleYValueLabel")
        self.ResampleYValueLabel.setGeometry(QRect(110, 130, 311, 16))
        self.ResampleZValueLabel = QLabel(Dialog)
        self.ResampleZValueLabel.setObjectName(u"ResampleZValueLabel")
        self.ResampleZValueLabel.setGeometry(QRect(110, 160, 311, 16))

        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Auto Resize - Voxel Size -", None))
        self.InputImageYLabel.setText(QCoreApplication.translate("Dialog", u"Y", None))
        self.InputImageZLineEdit.setText(QCoreApplication.translate("Dialog", u"1.0", None))
        self.ApplyAutoResizeButton.setText(QCoreApplication.translate("Dialog", u"Apply Auto Resize", None))
        self.InputImageZLabel.setText(QCoreApplication.translate("Dialog", u"Z", None))
        self.ResampleXLabel.setText(QCoreApplication.translate("Dialog", u"Resample X :", None))
        self.ResampleYLabel.setText(QCoreApplication.translate("Dialog", u"Resample Y :", None))
        self.TrainingDataZLabel.setText(QCoreApplication.translate("Dialog", u"Z", None))
        self.TrainingDataZLineEdit.setText(QCoreApplication.translate("Dialog", u"1", None))
        self.InputImageYLineEdit.setText(QCoreApplication.translate("Dialog", u"1.0", None))
        self.TrainingDataXLineEdit.setText(QCoreApplication.translate("Dialog", u"0.8", None))
        self.InputImageXLineEdit.setText(QCoreApplication.translate("Dialog", u"1.0", None))
        self.InputImageXLabel.setText(QCoreApplication.translate("Dialog", u"X", None))
        self.ResampleZLabel.setText(QCoreApplication.translate("Dialog", u"Resample Z :", None))
        self.TrainingDataXLabel.setText(QCoreApplication.translate("Dialog", u"X", None))
        self.TrainingDataYLineEdit.setText(QCoreApplication.translate("Dialog", u"0.8", None))
        self.TrainingDataLabel.setText(QCoreApplication.translate("Dialog", u"Training Data Voxel Size :", None))
        self.TrainingDataYLabel.setText(QCoreApplication.translate("Dialog", u"Y", None))
        self.InputImageLabel.setText(QCoreApplication.translate("Dialog", u"Input Image Voxel Size :", None))
        self.ResampleXValueLabel.setText("")
        self.ResampleYValueLabel.setText("")
        self.ResampleZValueLabel.setText("")
    # retranslateUi


class Ui_VolumeWindow(QDialog):
    def __init__(self, montageNuclei, maskNuclei, dt):
        super(Ui_VolumeWindow,self).__init__()
        self.setupUi(self)
        self.montageNuclei = montageNuclei
        self.maskNuclei = maskNuclei
        self.dt = dt
        self.labelsSelected = []
        self.volumesSelected = []
        self.layer = montageNuclei.copy()
        self.setImageWindows(montageNuclei.shape[0]-1)
        self.MontageVisuHorizontalSlider.valueChanged.connect(self.showImage)
        self.ConfirmSmallButton.clicked.connect(self.computeMinVolume)
        self.ConfirmBigButton.clicked.connect(self.computeMaxVolume)
        self.image_fig.canvas.mpl_connect('button_press_event', self.changeLayer)
    
    def setImageWindows(self, zShape)->None:
        self.MontageVisuHorizontalSlider.setValue((zShape)//2)
        self.MontageVisuHorizontalSlider.setMaximum(zShape)
        self.SliderValueLabel.setText(str((zShape)//2))
        layout = QVBoxLayout(self.MontageVisualisationFrame)
        self.image_fig, self.image_ax = plt.subplots(figsize=(8,8))
        self.image_ax.axis("off")
        self.image_fig.tight_layout()
        self.image_fig.set_facecolor((24/255, 24/255, 24/255))
        static_canvas = FigureCanvas(self.image_fig)
        navbar = NavigationToolbar(static_canvas, parent= None)
        navbar.update()
        layout.addWidget(navbar)
        layout.addWidget(static_canvas)
        self.image_canvas = static_canvas
        self.image_navbar = navbar
        self.showImage(z= (zShape)//2)
        
    def changeLayer(self, event : QMouseEvent):
        z = self.MontageVisuHorizontalSlider.value()
        x = int(event.xdata)
        y = int(event.ydata)
        currentLabel = self.maskNuclei[z, y, x]
        if currentLabel!= 0:
            volumeLab = self.dt[self.dt["cell_Id"] == float(currentLabel)]["volume_um3"].values[0]
            #select a nuclei
            if currentLabel not in self.labelsSelected: #complete list of selected labels (initialized when window 2 was opened)
                self.labelsSelected.append(currentLabel)     
                self.volumesSelected.append(volumeLab)
                in_label = self.maskNuclei == currentLabel
                in_label_rgb = np.repeat(in_label[..., np.newaxis], 3, axis=-1).astype('uint8')
                color_values = np.array([104, 178, 177]).astype('uint8')
                for c, val in enumerate(color_values):
                    in_label_rgb[..., c] *= val 
                z_pos = np.unique(np.where(in_label)[0])
                for iz in z_pos:
                    in_lbl_slice = in_label[iz]
                    self.layer[iz][in_lbl_slice] = in_label_rgb[iz][in_lbl_slice]  
            #unselect nuclei
            elif currentLabel in self.labelsSelected: #complete list of selected labels (initialized when window 2 was opened)
                self.labelsSelected.remove(currentLabel)
                self.volumesSelected.remove(volumeLab)
                in_label = self.maskNuclei == currentLabel
                self.layer[in_label] = self.montageNuclei[in_label] #normalize_data(montage)
            self.showImage()
            text = ""
            for vol in self.volumesSelected:
                text += str(int(vol)) + ", "           
            self.ListVolumesLabel.setText(text)
        
    def computeMinVolume(self):
        if len(self.labelsSelected)>0:
            self.layer = self.montageNuclei.copy()   
            vol_average_min = int(max(self.volumesSelected))
            self.MinVolumeValueLabel.setText(str(vol_average_min))
            self.showImage()
            self.labelsSelected = []
            self.volumesSelected = []
            self.ListVolumesLabel.setText("")
                    
    def computeMaxVolume(self):
        if len(self.labelsSelected)>0:
            self.layer = self.montageNuclei.copy()   
            vol_average_max = int(min(self.volumesSelected))
            self.MaxVolumeValueLabel.setText(str(vol_average_max))
            self.showImage()
            self.labelsSelected = []
            self.volumesSelected = []
            self.ListVolumesLabel.setText("")
        
    def showImage(self, z= None):
        if z is None:
            z = self.MontageVisuHorizontalSlider.value()
        self.image_ax.clear()
        im_np = self.montageNuclei[z]
        self.image_ax.imshow(im_np.astype('uint8'))
        self.image_ax.imshow(self.layer[z].astype('uint8'), alpha = 0.5)
        self.image_ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)      
        self.image_canvas.draw_idle()
        self.image_navbar._update_view()
        
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(797, 586)
        icon = QIcon()
        icon.addFile(u"OSlogo.ico", QSize(), QIcon.Normal, QIcon.Off)
        Dialog.setWindowIcon(icon)
        self.MontageVisualisationFrame = QFrame(Dialog)
        self.MontageVisualisationFrame.setObjectName(u"MontageVisualisationFrame")
        self.MontageVisualisationFrame.setGeometry(QRect(19, 20, 481, 481))
        self.MontageVisualisationFrame.setFrameShape(QFrame.StyledPanel)
        self.MontageVisualisationFrame.setFrameShadow(QFrame.Raised)
        self.MontageVisuHorizontalSlider = QSlider(Dialog)
        self.MontageVisuHorizontalSlider.setObjectName(u"MontageVisuHorizontalSlider")
        self.MontageVisuHorizontalSlider.setGeometry(QRect(20, 530, 441, 22))
        self.MontageVisuHorizontalSlider.setOrientation(Qt.Horizontal)
        self.ChooseVolumeToolBox = QToolBox(Dialog)
        self.ChooseVolumeToolBox.setObjectName(u"ChooseVolumeToolBox")
        self.ChooseVolumeToolBox.setGeometry(QRect(510, 20, 261, 521))
        self.VolumeSelectionPage = QWidget()
        self.VolumeSelectionPage.setObjectName(u"VolumeSelectionPage")
        self.VolumeSelectionPage.setGeometry(QRect(0, 0, 261, 491))
        self.MinVolumeValueLabel = QLabel(self.VolumeSelectionPage)
        self.MinVolumeValueLabel.setObjectName(u"MinVolumeValueLabel")
        self.MinVolumeValueLabel.setGeometry(QRect(170, 190, 71, 20))
        self.SelectedVolumesLabel = QLabel(self.VolumeSelectionPage)
        self.SelectedVolumesLabel.setObjectName(u"SelectedVolumesLabel")
        self.SelectedVolumesLabel.setGeometry(QRect(10, 10, 111, 16))
        self.ConfirmSmallButton = QPushButton(self.VolumeSelectionPage)
        self.ConfirmSmallButton.setObjectName(u"ConfirmSmallButton")
        self.ConfirmSmallButton.setGeometry(QRect(10, 150, 211, 24))
        self.label = QLabel(self.VolumeSelectionPage)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 190, 161, 16))
        self.MaxVolumeValueLabel = QLabel(self.VolumeSelectionPage)
        self.MaxVolumeValueLabel.setObjectName(u"MaxVolumeValueLabel")
        self.MaxVolumeValueLabel.setGeometry(QRect(170, 340, 71, 20))
        self.label_2 = QLabel(self.VolumeSelectionPage)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(10, 340, 161, 16))
        self.ConfirmBigButton = QPushButton(self.VolumeSelectionPage)
        self.ConfirmBigButton.setObjectName(u"ConfirmBigButton")
        self.ConfirmBigButton.setGeometry(QRect(10, 300, 211, 24))
        self.ListVolumesLabel = QLabel(self.VolumeSelectionPage)
        self.ListVolumesLabel.setObjectName(u"ListVolumesLabel")
        self.ListVolumesLabel.setGeometry(QRect(10, 40, 231, 16))
        self.ChooseVolumeToolBox.addItem(self.VolumeSelectionPage, u"Volume Selection")
        self.SliderValueLabel = QLabel(Dialog)
        self.SliderValueLabel.setObjectName(u"SliderValueLabel")
        self.SliderValueLabel.setGeometry(QRect(470, 530, 49, 16))

        self.retranslateUi(Dialog)
        self.MontageVisuHorizontalSlider.valueChanged.connect(self.SliderValueLabel.setNum)

        self.ChooseVolumeToolBox.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Optimal Volumes ", None))
        self.MinVolumeValueLabel.setText("")
        self.SelectedVolumesLabel.setText(QCoreApplication.translate("Dialog", u"Selected volumes :  ", None))
        self.ConfirmSmallButton.setText(QCoreApplication.translate("Dialog", u"Confirm Selection Small Nuclei", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"Min Volume Recommended :", None))
        self.MaxVolumeValueLabel.setText("")
        self.label_2.setText(QCoreApplication.translate("Dialog", u"Max Volume Recommended :", None))
        self.ConfirmBigButton.setText(QCoreApplication.translate("Dialog", u"Confirm Selection Big Nuclei", None))
        self.ListVolumesLabel.setText("")
        self.ChooseVolumeToolBox.setItemText(self.ChooseVolumeToolBox.indexOf(self.VolumeSelectionPage), QCoreApplication.translate("Dialog", u"Volume Selection", None))
        self.SliderValueLabel.setText("")
    # retranslateUi
  
  
     
class Ui_3dParametersWindow(QDialog):
    def __init__(self, FileInfo, spacingX, spacingY, spacingZ, opacityValueSlider=None, ValuesList=None):
        super(Ui_3dParametersWindow,self).__init__()
        self.setupUi(self)
        self.opacityValueSlider = 10 if opacityValueSlider is None else opacityValueSlider
        self.spacingX = spacingX
        self.spacingY = spacingY
        self.spacingZ = spacingZ
        self.FileInfo = FileInfo
        self.ValuesList = [] if ValuesList is None else ValuesList
        
        #set
        self.opacityMask3dHorizontalSlider.setValue(self.opacityValueSlider)
        self.opacityMaskValue3dLabel.setText(str(self.opacityValueSlider/10))
        
        self.spacingX3dLineEdit.setText(str(self.spacingX))
        self.spacingX3dLineEdit.setCursorPosition(0)
        
        self.spacingY3dLineEdit.setText(str(self.spacingY))
        self.spacingY3dLineEdit.setCursorPosition(0)
        
        self.spacingZ3dLineEdit.setText(str(self.spacingZ))
        self.spacingZ3dLineEdit.setCursorPosition(0)
        
        self.channels3dListWidget.addItems(self.ValuesList)
        
        self.setChannelsComboBox(self.FileInfo)
        self.getCurrentImageMinMax()
        #connections
        self.opacityMask3dHorizontalSlider.valueChanged.connect(self.updateNmsLabel)
        self.channel3dComboBox.currentTextChanged.connect(self.getCurrentImageMinMax)
        self.ok3dPushButton.clicked.connect(self.fixAllParameters)
        self.channelsAdd3dButton.clicked.connect(self.addChannelMinMax)
        self.channelsRemove3dButton.clicked.connect(self.removeThresholding)
        
    def fixAllParameters(self):
        self.spacingX = float(self.spacingX3dLineEdit.text())
        self.spacingY = float(self.spacingY3dLineEdit.text())
        self.spacingZ = float(self.spacingZ3dLineEdit.text())
        self.opacityValueSlider = self.opacityMask3dHorizontalSlider.value()
        self.ValuesList = [self.channels3dListWidget.item(x).text() for x in range(self.channels3dListWidget.count())]
        self.close()
    
    def addChannelMinMax(self):
        ValuesList = [self.channels3dListWidget.item(x).text() for x in range(self.channels3dListWidget.count())]
        channel = self.channel3dComboBox.currentText()
        min = self.Min3dLineEdit.text()
        max = self.Max3dLineEdit.text()
        try:
            float(min)
            item = 'C' + channel + ' min ' + min
            
            for i, el in enumerate(ValuesList):
                if 'C' + channel + ' min ' in el:
                    self.channels3dListWidget.takeItem(i)
            self.channels3dListWidget.addItem(item) 
        except:
            self.Min3dLineEdit.setText('')
        ValuesList = [self.channels3dListWidget.item(x).text() for x in range(self.channels3dListWidget.count())]    
        try:
            float(max)
            item = 'C' + channel + ' max ' + max
            
            for i, el in enumerate(ValuesList):
                if 'C' + channel + ' max ' in el:
                    self.channels3dListWidget.takeItem(i)
            self.channels3dListWidget.addItem(item)
        except:
            self.Max3dLineEdit.setText('')
        
    def removeThresholding(self):
        self.channels3dListWidget.takeItem(self.channels3dListWidget.currentRow())
        
    def getCurrentImageMinMax(self, ch = None):    
        if ch is None:
            ch = self.channel3dComboBox.currentText()
        channel = int(ch) - 1 if ch != '' else None
        img = self.FileInfo.Display()[:,:,:,channel] if channel is not None else self.FileInfo.Display()
        self.actualMin3dLabel.setText("Min : " + str(img.min()))
        self.actualMedian3dLabel.setText("Median : " + str(np.median(img)))
        self.actualMax3dLabel.setText("Max : " + str(img.max()))

    def setChannelsComboBox(self, FileInfo):
        channelsList = [str(i+1) for i in range(FileInfo.channel)]
        self.channel3dComboBox.clear()
        self.channel3dComboBox.addItems(channelsList)

    
    def updateNmsLabel(self, value):
        self.opacityMaskValue3dLabel.setText(str(value/10))
        
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(417, 468)
        icon = QIcon()
        icon.addFile(u"OSlogo.ico", QSize(), QIcon.Normal, QIcon.Off)
        Dialog.setWindowIcon(icon)
        self.verticalLayout_2 = QVBoxLayout(Dialog)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.general3dToolBox = QToolBox(Dialog)
        self.general3dToolBox.setObjectName(u"general3dToolBox")
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.page.setGeometry(QRect(0, 0, 397, 69))
        self.spacingGeneral3dLabel = QLabel(self.page)
        self.spacingGeneral3dLabel.setObjectName(u"spacingGeneral3dLabel")
        self.spacingGeneral3dLabel.setGeometry(QRect(10, 10, 61, 21))
        self.spacingY3dLabel = QLabel(self.page)
        self.spacingY3dLabel.setObjectName(u"spacingY3dLabel")
        self.spacingY3dLabel.setGeometry(QRect(180, 10, 16, 21))
        self.spacingZ3dLabel = QLabel(self.page)
        self.spacingZ3dLabel.setObjectName(u"spacingZ3dLabel")
        self.spacingZ3dLabel.setGeometry(QRect(290, 10, 16, 21))
        self.spacingX3dLabel = QLabel(self.page)
        self.spacingX3dLabel.setObjectName(u"spacingX3dLabel")
        self.spacingX3dLabel.setGeometry(QRect(70, 10, 16, 21))
        self.spacingZ3dLineEdit = QLineEdit(self.page)
        self.spacingZ3dLineEdit.setObjectName(u"spacingZ3dLineEdit")
        self.spacingZ3dLineEdit.setGeometry(QRect(310, 10, 60, 21))
        self.spacingY3dLineEdit = QLineEdit(self.page)
        self.spacingY3dLineEdit.setObjectName(u"spacingY3dLineEdit")
        self.spacingY3dLineEdit.setGeometry(QRect(200, 10, 60, 21))
        self.spacingX3dLineEdit = QLineEdit(self.page)
        self.spacingX3dLineEdit.setObjectName(u"spacingX3dLineEdit")
        self.spacingX3dLineEdit.setGeometry(QRect(90, 10, 60, 21))
        self.general3dToolBox.addItem(self.page, u"General")

        self.verticalLayout.addWidget(self.general3dToolBox)

        self.mask3dToolBox = QToolBox(Dialog)
        self.mask3dToolBox.setObjectName(u"mask3dToolBox")
        self.page_4 = QWidget()
        self.page_4.setObjectName(u"page_4")
        self.page_4.setGeometry(QRect(0, 0, 397, 69))
        self.opacityMask3dLabel = QLabel(self.page_4)
        self.opacityMask3dLabel.setObjectName(u"opacityMask3dLabel")
        self.opacityMask3dLabel.setGeometry(QRect(10, 20, 61, 21))
        self.opacityMask3dHorizontalSlider = QSlider(self.page_4)
        self.opacityMask3dHorizontalSlider.setObjectName(u"opacityMask3dHorizontalSlider")
        self.opacityMask3dHorizontalSlider.setGeometry(QRect(70, 20, 281, 22))
        self.opacityMask3dHorizontalSlider.setMaximum(10)
        self.opacityMask3dHorizontalSlider.setValue(10)
        self.opacityMask3dHorizontalSlider.setOrientation(Qt.Horizontal)
        self.opacityMaskValue3dLabel = QLabel(self.page_4)
        self.opacityMaskValue3dLabel.setObjectName(u"opacityMaskValue3dLabel")
        self.opacityMaskValue3dLabel.setGeometry(QRect(360, 20, 21, 21))
        self.mask3dToolBox.addItem(self.page_4, u"Mask")

        self.verticalLayout.addWidget(self.mask3dToolBox)

        self.channels3dToolBox = QToolBox(Dialog)
        self.channels3dToolBox.setObjectName(u"channels3dToolBox")
        self.page_5 = QWidget()
        self.page_5.setObjectName(u"page_5")
        self.page_5.setGeometry(QRect(0, 0, 397, 176))
        self.channel3dComboBox = QComboBox(self.page_5)
        self.channel3dComboBox.setObjectName(u"channel3dComboBox")
        self.channel3dComboBox.setGeometry(QRect(10, 10, 91, 22))
        self.Min3dLabel = QLabel(self.page_5)
        self.Min3dLabel.setObjectName(u"Min3dLabel")
        self.Min3dLabel.setGeometry(QRect(107, 11, 31, 20))
        self.Max3dLabel = QLabel(self.page_5)
        self.Max3dLabel.setObjectName(u"Max3dLabel")
        self.Max3dLabel.setGeometry(QRect(227, 11, 31, 20))
        self.Min3dLineEdit = QLineEdit(self.page_5)
        self.Min3dLineEdit.setObjectName(u"Min3dLineEdit")
        self.Min3dLineEdit.setGeometry(QRect(138, 10, 81, 21))
        self.Max3dLineEdit = QLineEdit(self.page_5)
        self.Max3dLineEdit.setObjectName(u"Max3dLineEdit")
        self.Max3dLineEdit.setGeometry(QRect(260, 10, 81, 21))
        self.channels3dListWidget = QListWidget(self.page_5)
        self.channels3dListWidget.setObjectName(u"channels3dListWidget")
        self.channels3dListWidget.setGeometry(QRect(10, 70, 321, 101))
        self.channelsAdd3dButton = QPushButton(self.page_5)
        self.channelsAdd3dButton.setObjectName(u"channelsAdd3dButton")
        self.channelsAdd3dButton.setGeometry(QRect(350, 10, 41, 22))
        self.channelsRemove3dButton = QPushButton(self.page_5)
        self.channelsRemove3dButton.setObjectName(u"channelsRemove3dButton")
        self.channelsRemove3dButton.setGeometry(QRect(340, 90, 51, 51))
        self.actualMin3dLabel = QLabel(self.page_5)
        self.actualMin3dLabel.setObjectName(u"actualMin3dLabel")
        self.actualMin3dLabel.setEnabled(True)
        self.actualMin3dLabel.setGeometry(QRect(42, 40, 90, 21))
        self.actualMedian3dLabel = QLabel(self.page_5)
        self.actualMedian3dLabel.setObjectName(u"actualMedian3dLabel")
        self.actualMedian3dLabel.setEnabled(True)
        self.actualMedian3dLabel.setGeometry(QRect(150, 40, 90, 21))
        self.actualMax3dLabel = QLabel(self.page_5)
        self.actualMax3dLabel.setObjectName(u"actualMax3dLabel")
        self.actualMax3dLabel.setEnabled(True)
        self.actualMax3dLabel.setGeometry(QRect(260, 40, 90, 21))
        self.channels3dToolBox.addItem(self.page_5, u"Channels")

        self.verticalLayout.addWidget(self.channels3dToolBox)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, -1, 0, -1)
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.ok3dPushButton = QPushButton(Dialog)
        self.ok3dPushButton.setObjectName(u"ok3dPushButton")

        self.horizontalLayout_2.addWidget(self.ok3dPushButton)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.verticalLayout.setStretch(2, 2)

        self.verticalLayout_2.addLayout(self.verticalLayout)


        self.retranslateUi(Dialog)

        self.general3dToolBox.setCurrentIndex(0)
        self.mask3dToolBox.setCurrentIndex(0)
        self.channels3dToolBox.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"3D Parameters", None))
        self.spacingGeneral3dLabel.setText(QCoreApplication.translate("Dialog", u"Spacing : ", None))
        self.spacingY3dLabel.setText(QCoreApplication.translate("Dialog", u"Y", None))
        self.spacingZ3dLabel.setText(QCoreApplication.translate("Dialog", u"Z", None))
        self.spacingX3dLabel.setText(QCoreApplication.translate("Dialog", u"X", None))
        self.spacingZ3dLineEdit.setText(QCoreApplication.translate("Dialog", u"1", None))
        self.spacingY3dLineEdit.setText(QCoreApplication.translate("Dialog", u"1", None))
        self.spacingX3dLineEdit.setText(QCoreApplication.translate("Dialog", u"1", None))
        self.general3dToolBox.setItemText(self.general3dToolBox.indexOf(self.page), QCoreApplication.translate("Dialog", u"General", None))
        self.opacityMask3dLabel.setText(QCoreApplication.translate("Dialog", u"Opacity :", None))
        self.opacityMaskValue3dLabel.setText(QCoreApplication.translate("Dialog", u"1", None))
        self.mask3dToolBox.setItemText(self.mask3dToolBox.indexOf(self.page_4), QCoreApplication.translate("Dialog", u"Mask", None))
        self.Min3dLabel.setText(QCoreApplication.translate("Dialog", u"Min : ", None))
        self.Max3dLabel.setText(QCoreApplication.translate("Dialog", u"Max : ", None))
        self.Min3dLineEdit.setText("")
        self.Max3dLineEdit.setText("")
        self.channelsAdd3dButton.setText(QCoreApplication.translate("Dialog", u"Add", None))
        self.channelsRemove3dButton.setText(QCoreApplication.translate("Dialog", u"Remove", None))
        self.actualMin3dLabel.setText(QCoreApplication.translate("Dialog", u"Min : ", None))
        self.actualMedian3dLabel.setText(QCoreApplication.translate("Dialog", u"Median : ", None))
        self.actualMax3dLabel.setText(QCoreApplication.translate("Dialog", u"Max : ", None))
        self.channels3dToolBox.setItemText(self.channels3dToolBox.indexOf(self.page_5), QCoreApplication.translate("Dialog", u"Channels", None))
        self.ok3dPushButton.setText(QCoreApplication.translate("Dialog", u"OK", None))
        
    # retranslateUi

class ProgressBarWindow(QDialog):
    def __init__(self, n_steps=5):
        super().__init__()

        self.setWindowTitle('Pipeline Progress')
        self.setGeometry(100, 100, 400, 100)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(30, 40, 340, 25)

        self.show()
        self.btn_cancel = QPushButton('Cancel')
        self.btn_cancel.clicked.connect(self.close)

        # Simulation de l'avancement du pipeline
        self.pipeline_steps = n_steps
        self.current_step = 0

        # Démarre le timer pour simuler le chargement progressif
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.updateProgressBar)
        # self.timer.start(1000)  # Mettez à jour toutes les secondes pour simuler les étapes        

    def updateProgressBar(self):
        self.current_step += 1
        if self.current_step <= self.pipeline_steps:
            self.progress_bar.setValue((self.current_step / self.pipeline_steps) * 100)
        # else:
        #     self.timer.stop()  # Arrête le timer lorsque le pipeline est terminé
        

class CPBar(QWidget):
    def __init__(self):
        super().__init__()
        self.p=0
        self.setMinimumSize(208, 208)

    def upd(self,pp):
        if self.p == pp:
            return
        self.p = pp
        self.update()

    def paintEvent(self,e):
        pd = self.p * 360
        rd = 360 - pd
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.white)
        p.translate(4, 4)
        p.setRenderHint(QPainter.Antialiasing)
        path, path2 = QPainterPath(),QPainterPath()
        path.moveTo(100, 0)
        path.arcTo(QRectF(0, 0, 200, 200), 90, -pd)
        pen, pen2 = QPen(), QPen()
        pen.setCapStyle(Qt.FlatCap)
        pen.setColor(QColor("#30b7e0"))
        pen.setWidth(8)
        p.strokePath(path, pen)
        path2.moveTo(100, 0)
        pen2.setWidth(8)
        pen2.setColor(QColor("#d7d7d7"))
        pen2.setCapStyle(Qt.FlatCap)
        pen2.setDashPattern([0.5, 1.105]) # remove this line to have continue cercle line
        path2.arcTo(QRectF(0, 0, 200, 200), 90, rd)
        pen2.setDashOffset(2.2) # this one too
        p.strokePath(path2, pen2)


class ProgressBarWindow0(QWidget):
  def __init__(self):
      super().__init__()
      l = QVBoxLayout(self)
      p = CPBar()
      s = QSlider(Qt.Horizontal, self)
      s.setMinimum(0)
      s.setMaximum(100)
      l.addWidget(p)
      l.addWidget(s)
      self.setLayout(l)
      s.valueChanged.connect(lambda :p.upd(s.value()/s.maximum()))
