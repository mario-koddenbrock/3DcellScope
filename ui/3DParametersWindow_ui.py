# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file '3DParametersWindow.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QSizePolicy, QSlider, QSpacerItem,
    QToolBox, QVBoxLayout, QWidget)

class Ui_Dialog(object):
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

