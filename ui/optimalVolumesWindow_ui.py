# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'optimalVolumesWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.5.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QLabel,
    QLayout, QPushButton, QSizePolicy, QSlider,
    QToolBox, QWidget)

class Ui_Dialog(object):
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

