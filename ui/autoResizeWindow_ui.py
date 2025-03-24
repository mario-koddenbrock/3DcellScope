# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'autoResizeWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QWidget)

class Ui_Dialog(object):
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

