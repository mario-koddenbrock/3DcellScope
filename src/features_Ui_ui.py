# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'features_Ui.ui'
##
## Created by: Qt User Interface Compiler version 6.5.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
    QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMenuBar,
    QPushButton, QRadioButton, QSizePolicy, QSpacerItem,
    QSpinBox, QStatusBar, QTableWidget, QTableWidgetItem,
    QToolBox, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(733, 1053)
        MainWindow.setMinimumSize(QSize(0, 700))
        self.actionload = QAction(MainWindow)
        self.actionload.setObjectName(u"actionload")
        self.actionimport = QAction(MainWindow)
        self.actionimport.setObjectName(u"actionimport")
        self.actionsave = QAction(MainWindow)
        self.actionsave.setObjectName(u"actionsave")
        self.actionLoad_CSV = QAction(MainWindow)
        self.actionLoad_CSV.setObjectName(u"actionLoad_CSV")
        self.actionLoad_Project = QAction(MainWindow)
        self.actionLoad_Project.setObjectName(u"actionLoad_Project")
        self.actionLoad_Multiple = QAction(MainWindow)
        self.actionLoad_Multiple.setObjectName(u"actionLoad_Multiple")
        self.actionSave_Project = QAction(MainWindow)
        self.actionSave_Project.setObjectName(u"actionSave_Project")
        self.actionExport_CSV = QAction(MainWindow)
        self.actionExport_CSV.setObjectName(u"actionExport_CSV")
        self.actionAggregate_Data = QAction(MainWindow)
        self.actionAggregate_Data.setObjectName(u"actionAggregate_Data")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setMinimumSize(QSize(540, 690))
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.toolBox = QToolBox(self.centralwidget)
        self.toolBox.setObjectName(u"toolBox")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolBox.sizePolicy().hasHeightForWidth())
        self.toolBox.setSizePolicy(sizePolicy)
        self.toolBox.setMinimumSize(QSize(0, 230))
        self.toolBox.setMaximumSize(QSize(16777215, 300))
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.page_2.setGeometry(QRect(0, 0, 715, 240))
        sizePolicy.setHeightForWidth(self.page_2.sizePolicy().hasHeightForWidth())
        self.page_2.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QVBoxLayout(self.page_2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.tableWidget = QTableWidget(self.page_2)
        self.tableWidget.setObjectName(u"tableWidget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy1)
        self.tableWidget.setMinimumSize(QSize(0, 132))
        self.tableWidget.setMaximumSize(QSize(16777215, 132))
        font = QFont()
        font.setPointSize(9)
        self.tableWidget.setFont(font)
        self.tableWidget.setFrameShape(QFrame.StyledPanel)
        self.tableWidget.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.tableWidget.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.tableWidget.setAlternatingRowColors(False)
        self.tableWidget.setSelectionMode(QAbstractItemView.NoSelection)
        self.tableWidget.setTextElideMode(Qt.ElideNone)
        self.tableWidget.setGridStyle(Qt.NoPen)
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(0)
        self.tableWidget.verticalHeader().setStretchLastSection(False)

        self.verticalLayout_2.addWidget(self.tableWidget)

        self.hideIgnoredFeaturesCheckBox = QCheckBox(self.page_2)
        self.hideIgnoredFeaturesCheckBox.setObjectName(u"hideIgnoredFeaturesCheckBox")
        self.hideIgnoredFeaturesCheckBox.setMinimumSize(QSize(0, 15))
        self.hideIgnoredFeaturesCheckBox.setMaximumSize(QSize(16777215, 15))

        self.verticalLayout_2.addWidget(self.hideIgnoredFeaturesCheckBox)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer)

        self.toolBox.addItem(self.page_2, u"Features")
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.page.setGeometry(QRect(0, 0, 715, 240))
        self.page.setMinimumSize(QSize(0, 200))
        self.verticalLayout_3 = QVBoxLayout(self.page)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.descriptorListWidget = QListWidget(self.page)
        self.descriptorListWidget.setObjectName(u"descriptorListWidget")
        self.descriptorListWidget.setMinimumSize(QSize(0, 50))
        font1 = QFont()
        font1.setPointSize(8)
        self.descriptorListWidget.setFont(font1)
        self.descriptorListWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.verticalLayout_3.addWidget(self.descriptorListWidget)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.excludeSelectedDescriptorButton = QPushButton(self.page)
        self.excludeSelectedDescriptorButton.setObjectName(u"excludeSelectedDescriptorButton")

        self.horizontalLayout_3.addWidget(self.excludeSelectedDescriptorButton)

        self.keepOnlySelectedDescriptorButton = QPushButton(self.page)
        self.keepOnlySelectedDescriptorButton.setObjectName(u"keepOnlySelectedDescriptorButton")

        self.horizontalLayout_3.addWidget(self.keepOnlySelectedDescriptorButton)

        self.renameDescriptorButton = QPushButton(self.page)
        self.renameDescriptorButton.setObjectName(u"renameDescriptorButton")

        self.horizontalLayout_3.addWidget(self.renameDescriptorButton)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_2)

        self.resetDescriptorFilterButton = QPushButton(self.page)
        self.resetDescriptorFilterButton.setObjectName(u"resetDescriptorFilterButton")

        self.horizontalLayout_3.addWidget(self.resetDescriptorFilterButton)


        self.verticalLayout_3.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.ApllyDescriptorCorrelationButton = QPushButton(self.page)
        self.ApllyDescriptorCorrelationButton.setObjectName(u"ApllyDescriptorCorrelationButton")

        self.horizontalLayout_4.addWidget(self.ApllyDescriptorCorrelationButton)

        self.label_6 = QLabel(self.page)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_4.addWidget(self.label_6)

        self.distanceThresholdCorrelationSpinBox = QDoubleSpinBox(self.page)
        self.distanceThresholdCorrelationSpinBox.setObjectName(u"distanceThresholdCorrelationSpinBox")
        self.distanceThresholdCorrelationSpinBox.setMaximum(1.000000000000000)
        self.distanceThresholdCorrelationSpinBox.setSingleStep(0.010000000000000)
        self.distanceThresholdCorrelationSpinBox.setValue(0.100000000000000)

        self.horizontalLayout_4.addWidget(self.distanceThresholdCorrelationSpinBox)

        self.showCorrelationCheckBox = QCheckBox(self.page)
        self.showCorrelationCheckBox.setObjectName(u"showCorrelationCheckBox")

        self.horizontalLayout_4.addWidget(self.showCorrelationCheckBox)

        self.filterDescriptorWithCorrelationCheckBox = QCheckBox(self.page)
        self.filterDescriptorWithCorrelationCheckBox.setObjectName(u"filterDescriptorWithCorrelationCheckBox")

        self.horizontalLayout_4.addWidget(self.filterDescriptorWithCorrelationCheckBox)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_3)

        self.resetDescriptorCorelationButton = QPushButton(self.page)
        self.resetDescriptorCorelationButton.setObjectName(u"resetDescriptorCorelationButton")

        self.horizontalLayout_4.addWidget(self.resetDescriptorCorelationButton)


        self.verticalLayout_3.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.squareDistCorrelationRadioButton = QRadioButton(self.page)
        self.squareDistCorrelationRadioButton.setObjectName(u"squareDistCorrelationRadioButton")
        self.squareDistCorrelationRadioButton.setChecked(True)

        self.horizontalLayout_5.addWidget(self.squareDistCorrelationRadioButton)

        self.absDistCorrelationRadioButton = QRadioButton(self.page)
        self.absDistCorrelationRadioButton.setObjectName(u"absDistCorrelationRadioButton")

        self.horizontalLayout_5.addWidget(self.absDistCorrelationRadioButton)

        self.diffDistCorrelationRadioButton = QRadioButton(self.page)
        self.diffDistCorrelationRadioButton.setObjectName(u"diffDistCorrelationRadioButton")

        self.horizontalLayout_5.addWidget(self.diffDistCorrelationRadioButton)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_4)


        self.verticalLayout_3.addLayout(self.horizontalLayout_5)

        self.toolBox.addItem(self.page, u"Descriptor Selection")

        self.verticalLayout.addWidget(self.toolBox)

        self.toolBox_2 = QToolBox(self.centralwidget)
        self.toolBox_2.setObjectName(u"toolBox_2")
        self.toolBox_2.setMinimumSize(QSize(0, 170))
        self.page_3 = QWidget()
        self.page_3.setObjectName(u"page_3")
        self.page_3.setGeometry(QRect(0, 0, 715, 304))
        self.horizontalLayout_2 = QHBoxLayout(self.page_3)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.dataGroupWidget = QTableWidget(self.page_3)
        self.dataGroupWidget.setObjectName(u"dataGroupWidget")
        sizePolicy2 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.dataGroupWidget.sizePolicy().hasHeightForWidth())
        self.dataGroupWidget.setSizePolicy(sizePolicy2)
        self.dataGroupWidget.setMinimumSize(QSize(300, 100))
        self.dataGroupWidget.setFont(font)
        self.dataGroupWidget.setFrameShape(QFrame.StyledPanel)
        self.dataGroupWidget.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.dataGroupWidget.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.dataGroupWidget.setAlternatingRowColors(False)
        self.dataGroupWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.dataGroupWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.dataGroupWidget.setTextElideMode(Qt.ElideNone)
        self.dataGroupWidget.setGridStyle(Qt.NoPen)
        self.dataGroupWidget.setRowCount(0)
        self.dataGroupWidget.setColumnCount(0)
        self.dataGroupWidget.verticalHeader().setStretchLastSection(False)

        self.horizontalLayout_2.addWidget(self.dataGroupWidget)

        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.filterGroupButtonRemove = QPushButton(self.page_3)
        self.filterGroupButtonRemove.setObjectName(u"filterGroupButtonRemove")

        self.verticalLayout_4.addWidget(self.filterGroupButtonRemove)

        self.filterGroupButtonKeepOnly = QPushButton(self.page_3)
        self.filterGroupButtonKeepOnly.setObjectName(u"filterGroupButtonKeepOnly")

        self.verticalLayout_4.addWidget(self.filterGroupButtonKeepOnly)

        self.featureGateButton = QPushButton(self.page_3)
        self.featureGateButton.setObjectName(u"featureGateButton")

        self.verticalLayout_4.addWidget(self.featureGateButton)


        self.horizontalLayout.addLayout(self.verticalLayout_4)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.filterGroupButtonBack = QPushButton(self.page_3)
        self.filterGroupButtonBack.setObjectName(u"filterGroupButtonBack")
        sizePolicy3 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.filterGroupButtonBack.sizePolicy().hasHeightForWidth())
        self.filterGroupButtonBack.setSizePolicy(sizePolicy3)
        self.filterGroupButtonBack.setMinimumSize(QSize(20, 20))
        self.filterGroupButtonBack.setMaximumSize(QSize(20, 20))

        self.verticalLayout_5.addWidget(self.filterGroupButtonBack)

        self.filterGroupButtonReset = QPushButton(self.page_3)
        self.filterGroupButtonReset.setObjectName(u"filterGroupButtonReset")
        self.filterGroupButtonReset.setMaximumSize(QSize(20, 16777215))

        self.verticalLayout_5.addWidget(self.filterGroupButtonReset)


        self.horizontalLayout.addLayout(self.verticalLayout_5)


        self.verticalLayout_6.addLayout(self.horizontalLayout)

        self.aggregateDataButton = QPushButton(self.page_3)
        self.aggregateDataButton.setObjectName(u"aggregateDataButton")

        self.verticalLayout_6.addWidget(self.aggregateDataButton)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_6.addItem(self.verticalSpacer_2)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.EditGroupIndexValues = QPushButton(self.page_3)
        self.EditGroupIndexValues.setObjectName(u"EditGroupIndexValues")

        self.horizontalLayout_8.addWidget(self.EditGroupIndexValues)

        self.countsButton = QPushButton(self.page_3)
        self.countsButton.setObjectName(u"countsButton")

        self.horizontalLayout_8.addWidget(self.countsButton)


        self.verticalLayout_6.addLayout(self.horizontalLayout_8)


        self.horizontalLayout_2.addLayout(self.verticalLayout_6)

        self.toolBox_2.addItem(self.page_3, u"Data Groups")

        self.verticalLayout.addWidget(self.toolBox_2)

        self.toolBox_3 = QToolBox(self.centralwidget)
        self.toolBox_3.setObjectName(u"toolBox_3")
        self.toolBox_3.setMinimumSize(QSize(0, 250))
        self.toolBox_3.setMaximumSize(QSize(16777215, 310))
        self.page_4 = QWidget()
        self.page_4.setObjectName(u"page_4")
        self.page_4.setGeometry(QRect(0, 0, 715, 250))
        self.verticalLayout_7 = QVBoxLayout(self.page_4)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.barPlotButton = QPushButton(self.page_4)
        self.barPlotButton.setObjectName(u"barPlotButton")

        self.horizontalLayout_12.addWidget(self.barPlotButton)

        self.boxPlotButton = QPushButton(self.page_4)
        self.boxPlotButton.setObjectName(u"boxPlotButton")

        self.horizontalLayout_12.addWidget(self.boxPlotButton)

        self.densityPlotButton = QPushButton(self.page_4)
        self.densityPlotButton.setObjectName(u"densityPlotButton")

        self.horizontalLayout_12.addWidget(self.densityPlotButton)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_12.addItem(self.horizontalSpacer)

        self.label_2 = QLabel(self.page_4)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_12.addWidget(self.label_2)

        self.plotColorComboBox = QComboBox(self.page_4)
        self.plotColorComboBox.addItem("")
        self.plotColorComboBox.addItem("")
        self.plotColorComboBox.addItem("")
        self.plotColorComboBox.addItem("")
        self.plotColorComboBox.setObjectName(u"plotColorComboBox")
        self.plotColorComboBox.setMinimumSize(QSize(75, 0))
        self.plotColorComboBox.setMaximumSize(QSize(65, 16777215))
        self.plotColorComboBox.setEditable(True)

        self.horizontalLayout_12.addWidget(self.plotColorComboBox)

        self.label = QLabel(self.page_4)
        self.label.setObjectName(u"label")

        self.horizontalLayout_12.addWidget(self.label)

        self.stylePlotComboBox = QComboBox(self.page_4)
        self.stylePlotComboBox.addItem("")
        self.stylePlotComboBox.addItem("")
        self.stylePlotComboBox.setObjectName(u"stylePlotComboBox")
        self.stylePlotComboBox.setMinimumSize(QSize(65, 0))

        self.horizontalLayout_12.addWidget(self.stylePlotComboBox)


        self.verticalLayout_7.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.scatterPlotButton = QPushButton(self.page_4)
        self.scatterPlotButton.setObjectName(u"scatterPlotButton")

        self.horizontalLayout_6.addWidget(self.scatterPlotButton)

        self.pcaButton = QPushButton(self.page_4)
        self.pcaButton.setObjectName(u"pcaButton")

        self.horizontalLayout_6.addWidget(self.pcaButton)

        self.ldaButton = QPushButton(self.page_4)
        self.ldaButton.setObjectName(u"ldaButton")

        self.horizontalLayout_6.addWidget(self.ldaButton)

        self.horizontalSpacer_11 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_11)

        self.label_14 = QLabel(self.page_4)
        self.label_14.setObjectName(u"label_14")

        self.horizontalLayout_6.addWidget(self.label_14)

        self.maxPointLineEdit = QLineEdit(self.page_4)
        self.maxPointLineEdit.setObjectName(u"maxPointLineEdit")
        self.maxPointLineEdit.setMaximumSize(QSize(70, 16777215))

        self.horizontalLayout_6.addWidget(self.maxPointLineEdit)

        self.label_13 = QLabel(self.page_4)
        self.label_13.setObjectName(u"label_13")

        self.horizontalLayout_6.addWidget(self.label_13)

        self.ax1SpinBox = QSpinBox(self.page_4)
        self.ax1SpinBox.setObjectName(u"ax1SpinBox")
        self.ax1SpinBox.setMinimum(0)
        self.ax1SpinBox.setMaximum(10)
        self.ax1SpinBox.setValue(1)

        self.horizontalLayout_6.addWidget(self.ax1SpinBox)

        self.ax2SpinBox = QSpinBox(self.page_4)
        self.ax2SpinBox.setObjectName(u"ax2SpinBox")
        self.ax2SpinBox.setMinimum(0)
        self.ax2SpinBox.setMaximum(10)
        self.ax2SpinBox.setValue(2)

        self.horizontalLayout_6.addWidget(self.ax2SpinBox)

        self.ax3SpinBox = QSpinBox(self.page_4)
        self.ax3SpinBox.setObjectName(u"ax3SpinBox")
        self.ax3SpinBox.setMinimum(0)
        self.ax3SpinBox.setMaximum(10)
        self.ax3SpinBox.setValue(3)

        self.horizontalLayout_6.addWidget(self.ax3SpinBox)


        self.verticalLayout_7.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalSpacer_12 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_12)

        self.label_7 = QLabel(self.page_4)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_7.addWidget(self.label_7)

        self.brestCorrelationSpinBox = QSpinBox(self.page_4)
        self.brestCorrelationSpinBox.setObjectName(u"brestCorrelationSpinBox")
        self.brestCorrelationSpinBox.setMinimum(0)
        self.brestCorrelationSpinBox.setMaximum(10)
        self.brestCorrelationSpinBox.setValue(0)

        self.horizontalLayout_7.addWidget(self.brestCorrelationSpinBox)

        self.label_8 = QLabel(self.page_4)
        self.label_8.setObjectName(u"label_8")

        self.horizontalLayout_7.addWidget(self.label_8)

        self.pcaModeComboBox = QComboBox(self.page_4)
        self.pcaModeComboBox.addItem("")
        self.pcaModeComboBox.addItem("")
        self.pcaModeComboBox.addItem("")
        self.pcaModeComboBox.addItem("")
        self.pcaModeComboBox.addItem("")
        self.pcaModeComboBox.setObjectName(u"pcaModeComboBox")
        self.pcaModeComboBox.setEditable(False)

        self.horizontalLayout_7.addWidget(self.pcaModeComboBox)


        self.verticalLayout_7.addLayout(self.horizontalLayout_7)

        self.line = QFrame(self.page_4)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_7.addWidget(self.line)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.summaryPlottButton = QPushButton(self.page_4)
        self.summaryPlottButton.setObjectName(u"summaryPlottButton")

        self.horizontalLayout_9.addWidget(self.summaryPlottButton)

        self.crossTestingButton = QPushButton(self.page_4)
        self.crossTestingButton.setObjectName(u"crossTestingButton")

        self.horizontalLayout_9.addWidget(self.crossTestingButton)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_5)

        self.ssmdRadioButton = QRadioButton(self.page_4)
        self.ssmdRadioButton.setObjectName(u"ssmdRadioButton")
        self.ssmdRadioButton.setCheckable(True)
        self.ssmdRadioButton.setChecked(True)

        self.horizontalLayout_9.addWidget(self.ssmdRadioButton)

        self.pValueRadioButton = QRadioButton(self.page_4)
        self.pValueRadioButton.setObjectName(u"pValueRadioButton")

        self.horizontalLayout_9.addWidget(self.pValueRadioButton)

        self.uValueRadioButton = QRadioButton(self.page_4)
        self.uValueRadioButton.setObjectName(u"uValueRadioButton")

        self.horizontalLayout_9.addWidget(self.uValueRadioButton)

        self.maskSummaryCheckBox = QCheckBox(self.page_4)
        self.maskSummaryCheckBox.setObjectName(u"maskSummaryCheckBox")

        self.horizontalLayout_9.addWidget(self.maskSummaryCheckBox)

        self.correctionSummaryCheckBox = QCheckBox(self.page_4)
        self.correctionSummaryCheckBox.setObjectName(u"correctionSummaryCheckBox")

        self.horizontalLayout_9.addWidget(self.correctionSummaryCheckBox)


        self.verticalLayout_7.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.radarPlotButton = QPushButton(self.page_4)
        self.radarPlotButton.setObjectName(u"radarPlotButton")

        self.horizontalLayout_10.addWidget(self.radarPlotButton)

        self.replicateRadarButton = QPushButton(self.page_4)
        self.replicateRadarButton.setObjectName(u"replicateRadarButton")

        self.horizontalLayout_10.addWidget(self.replicateRadarButton)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_10.addItem(self.horizontalSpacer_9)

        self.label_9 = QLabel(self.page_4)
        self.label_9.setObjectName(u"label_9")

        self.horizontalLayout_10.addWidget(self.label_9)

        self.ssmdAxisLimitSpinBox = QSpinBox(self.page_4)
        self.ssmdAxisLimitSpinBox.setObjectName(u"ssmdAxisLimitSpinBox")
        self.ssmdAxisLimitSpinBox.setMinimum(2)
        self.ssmdAxisLimitSpinBox.setMaximum(10)
        self.ssmdAxisLimitSpinBox.setValue(6)

        self.horizontalLayout_10.addWidget(self.ssmdAxisLimitSpinBox)

        self.sortRadarCheckBox = QCheckBox(self.page_4)
        self.sortRadarCheckBox.setObjectName(u"sortRadarCheckBox")
        self.sortRadarCheckBox.setChecked(False)

        self.horizontalLayout_10.addWidget(self.sortRadarCheckBox)


        self.verticalLayout_7.addLayout(self.horizontalLayout_10)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.automaticAnalysisButton = QPushButton(self.page_4)
        self.automaticAnalysisButton.setObjectName(u"automaticAnalysisButton")

        self.horizontalLayout_11.addWidget(self.automaticAnalysisButton)

        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_11.addItem(self.horizontalSpacer_10)

        self.label_10 = QLabel(self.page_4)
        self.label_10.setObjectName(u"label_10")

        self.horizontalLayout_11.addWidget(self.label_10)

        self.significativityTresholdLineEdit = QLineEdit(self.page_4)
        self.significativityTresholdLineEdit.setObjectName(u"significativityTresholdLineEdit")
        self.significativityTresholdLineEdit.setMaximumSize(QSize(100, 16777215))

        self.horizontalLayout_11.addWidget(self.significativityTresholdLineEdit)


        self.verticalLayout_7.addLayout(self.horizontalLayout_11)

        self.toolBox_3.addItem(self.page_4, u"Plots")
        self.page_5 = QWidget()
        self.page_5.setObjectName(u"page_5")
        self.page_5.setGeometry(QRect(0, 0, 715, 250))
        self.verticalLayout_8 = QVBoxLayout(self.page_5)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalLayout_16.setContentsMargins(-1, 0, -1, -1)
        self.label_17 = QLabel(self.page_5)
        self.label_17.setObjectName(u"label_17")

        self.horizontalLayout_16.addWidget(self.label_17)

        self.bagChoiceList = QListWidget(self.page_5)
        self.bagChoiceList.setObjectName(u"bagChoiceList")
        self.bagChoiceList.setMinimumSize(QSize(50, 50))
        self.bagChoiceList.setFont(font1)

        self.horizontalLayout_16.addWidget(self.bagChoiceList)

        self.label_20 = QLabel(self.page_5)
        self.label_20.setObjectName(u"label_20")

        self.horizontalLayout_16.addWidget(self.label_20)

        self.preprocChoiceList = QListWidget(self.page_5)
        QListWidgetItem(self.preprocChoiceList)
        self.preprocChoiceList.setObjectName(u"preprocChoiceList")
        sizePolicy.setHeightForWidth(self.preprocChoiceList.sizePolicy().hasHeightForWidth())
        self.preprocChoiceList.setSizePolicy(sizePolicy)
        self.preprocChoiceList.setMinimumSize(QSize(50, 50))
        font2 = QFont()
        font2.setPointSize(8)
        font2.setItalic(True)
        self.preprocChoiceList.setFont(font2)
        self.preprocChoiceList.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.horizontalLayout_16.addWidget(self.preprocChoiceList)

        self.horizontalSpacer_13 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_16.addItem(self.horizontalSpacer_13)


        self.verticalLayout_8.addLayout(self.horizontalLayout_16)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.widget_7 = QWidget(self.page_5)
        self.widget_7.setObjectName(u"widget_7")
        sizePolicy4 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.widget_7.sizePolicy().hasHeightForWidth())
        self.widget_7.setSizePolicy(sizePolicy4)
        self.widget_7.setMinimumSize(QSize(150, 100))
        self.widget_7.setMaximumSize(QSize(125, 16777215))
        self.modelComboBox = QComboBox(self.widget_7)
        self.modelComboBox.addItem("")
        self.modelComboBox.setObjectName(u"modelComboBox")
        self.modelComboBox.setGeometry(QRect(0, 20, 111, 22))
        self.modelComboBox.setEditable(True)
        self.label_19 = QLabel(self.widget_7)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setGeometry(QRect(0, 0, 111, 20))
        self.modelTypeComboBox = QComboBox(self.widget_7)
        self.modelTypeComboBox.addItem("")
        self.modelTypeComboBox.addItem("")
        self.modelTypeComboBox.addItem("")
        self.modelTypeComboBox.addItem("")
        self.modelTypeComboBox.setObjectName(u"modelTypeComboBox")
        self.modelTypeComboBox.setGeometry(QRect(30, 71, 111, 21))
        self.modelTypeComboBox.setEditable(False)
        self.addModelButton = QPushButton(self.widget_7)
        self.addModelButton.setObjectName(u"addModelButton")
        self.addModelButton.setGeometry(QRect(0, 70, 21, 24))

        self.horizontalLayout_13.addWidget(self.widget_7)

        self.horizontalSpacer_6 = QSpacerItem(20, 42, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_13.addItem(self.horizontalSpacer_6)

        self.widget = QWidget(self.page_5)
        self.widget.setObjectName(u"widget")
        self.widget.setMinimumSize(QSize(50, 50))
        self.toogleEditModelButton = QPushButton(self.widget)
        self.toogleEditModelButton.setObjectName(u"toogleEditModelButton")
        self.toogleEditModelButton.setGeometry(QRect(10, 0, 41, 24))

        self.horizontalLayout_13.addWidget(self.widget)

        self.classifOptWidget = QWidget(self.page_5)
        self.classifOptWidget.setObjectName(u"classifOptWidget")
        self.classifOptWidget.setEnabled(False)
        sizePolicy.setHeightForWidth(self.classifOptWidget.sizePolicy().hasHeightForWidth())
        self.classifOptWidget.setSizePolicy(sizePolicy)
        self.classifOptWidget.setMinimumSize(QSize(420, 100))
        self.layoutWidget = QWidget(self.classifOptWidget)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(0, 0, 434, 100))
        self.ClassifOptLayout = QGridLayout(self.layoutWidget)
        self.ClassifOptLayout.setObjectName(u"ClassifOptLayout")
        self.ClassifOptLayout.setContentsMargins(0, 0, 0, 0)
        self.ClassifOpt3Widg = QWidget(self.layoutWidget)
        self.ClassifOpt3Widg.setObjectName(u"ClassifOpt3Widg")
        self.ClassifOpt3Widg.setMinimumSize(QSize(140, 0))
        self.comboBox_7 = QComboBox(self.ClassifOpt3Widg)
        self.comboBox_7.setObjectName(u"comboBox_7")
        self.comboBox_7.setGeometry(QRect(0, 20, 111, 22))
        self.comboBox_7.setEditable(True)
        self.label_16 = QLabel(self.ClassifOpt3Widg)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setGeometry(QRect(0, 0, 111, 20))

        self.ClassifOptLayout.addWidget(self.ClassifOpt3Widg, 0, 2, 1, 1)

        self.ClassifOpt2Widg = QWidget(self.layoutWidget)
        self.ClassifOpt2Widg.setObjectName(u"ClassifOpt2Widg")
        self.ClassifOpt2Widg.setMinimumSize(QSize(140, 0))
        self.comboBox_6 = QComboBox(self.ClassifOpt2Widg)
        self.comboBox_6.setObjectName(u"comboBox_6")
        self.comboBox_6.setGeometry(QRect(0, 20, 111, 22))
        self.comboBox_6.setEditable(True)
        self.label_15 = QLabel(self.ClassifOpt2Widg)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(0, 0, 111, 20))

        self.ClassifOptLayout.addWidget(self.ClassifOpt2Widg, 0, 1, 1, 1)

        self.ClassifOpt6Widg = QWidget(self.layoutWidget)
        self.ClassifOpt6Widg.setObjectName(u"ClassifOpt6Widg")
        self.ClassifOpt6Widg.setMinimumSize(QSize(140, 0))
        self.comboBox_5 = QComboBox(self.ClassifOpt6Widg)
        self.comboBox_5.setObjectName(u"comboBox_5")
        self.comboBox_5.setGeometry(QRect(0, 20, 111, 22))
        self.comboBox_5.setEditable(True)
        self.label_12 = QLabel(self.ClassifOpt6Widg)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(0, 0, 111, 20))

        self.ClassifOptLayout.addWidget(self.ClassifOpt6Widg, 1, 2, 1, 1)

        self.ClassifOpt5Widg = QWidget(self.layoutWidget)
        self.ClassifOpt5Widg.setObjectName(u"ClassifOpt5Widg")
        self.ClassifOpt5Widg.setMinimumSize(QSize(140, 0))
        self.comboBox_3 = QComboBox(self.ClassifOpt5Widg)
        self.comboBox_3.setObjectName(u"comboBox_3")
        self.comboBox_3.setGeometry(QRect(0, 20, 111, 22))
        self.comboBox_3.setEditable(True)
        self.label_4 = QLabel(self.ClassifOpt5Widg)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(0, 0, 111, 20))

        self.ClassifOptLayout.addWidget(self.ClassifOpt5Widg, 1, 1, 1, 1)

        self.ClassifOpt4Widg = QWidget(self.layoutWidget)
        self.ClassifOpt4Widg.setObjectName(u"ClassifOpt4Widg")
        self.ClassifOpt4Widg.setMinimumSize(QSize(140, 0))
        self.comboBox_9 = QComboBox(self.ClassifOpt4Widg)
        self.comboBox_9.setObjectName(u"comboBox_9")
        self.comboBox_9.setGeometry(QRect(0, 20, 111, 22))
        self.comboBox_9.setEditable(True)
        self.label_18 = QLabel(self.ClassifOpt4Widg)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setGeometry(QRect(0, 0, 111, 20))

        self.ClassifOptLayout.addWidget(self.ClassifOpt4Widg, 1, 0, 1, 1)

        self.ClassifOpt1Widg = QWidget(self.layoutWidget)
        self.ClassifOpt1Widg.setObjectName(u"ClassifOpt1Widg")
        self.ClassifOpt1Widg.setMinimumSize(QSize(140, 0))
        self.comboBox = QComboBox(self.ClassifOpt1Widg)
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setGeometry(QRect(0, 20, 111, 22))
        self.comboBox.setEditable(True)
        self.label_3 = QLabel(self.ClassifOpt1Widg)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(0, 0, 111, 20))

        self.ClassifOptLayout.addWidget(self.ClassifOpt1Widg, 0, 0, 1, 1)


        self.horizontalLayout_13.addWidget(self.classifOptWidget)


        self.verticalLayout_8.addLayout(self.horizontalLayout_13)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_8.addItem(self.verticalSpacer_3)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.horizontalLayout_15.setContentsMargins(-1, 0, -1, -1)
        self.classifOptimButton = QPushButton(self.page_5)
        self.classifOptimButton.setObjectName(u"classifOptimButton")

        self.horizontalLayout_15.addWidget(self.classifOptimButton)

        self.label_21 = QLabel(self.page_5)
        self.label_21.setObjectName(u"label_21")

        self.horizontalLayout_15.addWidget(self.label_21)

        self.classifOptimIterSpinBox = QSpinBox(self.page_5)
        self.classifOptimIterSpinBox.setObjectName(u"classifOptimIterSpinBox")
        self.classifOptimIterSpinBox.setMinimum(1)
        self.classifOptimIterSpinBox.setValue(20)

        self.horizontalLayout_15.addWidget(self.classifOptimIterSpinBox)

        self.crossValidateButton = QPushButton(self.page_5)
        self.crossValidateButton.setObjectName(u"crossValidateButton")

        self.horizontalLayout_15.addWidget(self.crossValidateButton)

        self.testClassifButton = QPushButton(self.page_5)
        self.testClassifButton.setObjectName(u"testClassifButton")

        self.horizontalLayout_15.addWidget(self.testClassifButton)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_15.addItem(self.horizontalSpacer_8)

        self.label_5 = QLabel(self.page_5)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setEnabled(True)

        self.horizontalLayout_15.addWidget(self.label_5)

        self.cvScorsComboBox = QComboBox(self.page_5)
        self.cvScorsComboBox.setObjectName(u"cvScorsComboBox")
        self.cvScorsComboBox.setEnabled(False)
        sizePolicy5 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.cvScorsComboBox.sizePolicy().hasHeightForWidth())
        self.cvScorsComboBox.setSizePolicy(sizePolicy5)

        self.horizontalLayout_15.addWidget(self.cvScorsComboBox)


        self.verticalLayout_8.addLayout(self.horizontalLayout_15)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_14.addItem(self.horizontalSpacer_7)

        self.modelFitButton = QPushButton(self.page_5)
        self.modelFitButton.setObjectName(u"modelFitButton")

        self.horizontalLayout_14.addWidget(self.modelFitButton)

        self.modelPredictButton = QPushButton(self.page_5)
        self.modelPredictButton.setObjectName(u"modelPredictButton")

        self.horizontalLayout_14.addWidget(self.modelPredictButton)


        self.verticalLayout_8.addLayout(self.horizontalLayout_14)

        self.toolBox_3.addItem(self.page_5, u"Classification")

        self.verticalLayout.addWidget(self.toolBox_3)

        self.bottomFrame = QFrame(self.centralwidget)
        self.bottomFrame.setObjectName(u"bottomFrame")
        self.bottomFrame.setMinimumSize(QSize(0, 30))
        self.bottomFrame.setFrameShape(QFrame.StyledPanel)
        self.bottomFrame.setFrameShadow(QFrame.Raised)
        self.label_11 = QLabel(self.bottomFrame)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(10, 5, 101, 16))
        self.displayINTRadioButton = QRadioButton(self.bottomFrame)
        self.displayINTRadioButton.setObjectName(u"displayINTRadioButton")
        self.displayINTRadioButton.setGeometry(QRect(145, 5, 89, 20))
        self.displayINTRadioButton.setChecked(True)
        self.displayPNGRadioButton = QRadioButton(self.bottomFrame)
        self.displayPNGRadioButton.setObjectName(u"displayPNGRadioButton")
        self.displayPNGRadioButton.setGeometry(QRect(250, 5, 89, 20))
        self.displayPDFRadioButton = QRadioButton(self.bottomFrame)
        self.displayPDFRadioButton.setObjectName(u"displayPDFRadioButton")
        self.displayPDFRadioButton.setGeometry(QRect(350, 5, 89, 20))
        self.displayCSVRadioButton = QRadioButton(self.bottomFrame)
        self.displayCSVRadioButton.setObjectName(u"displayCSVRadioButton")
        self.displayCSVRadioButton.setGeometry(QRect(445, 5, 89, 20))

        self.verticalLayout.addWidget(self.bottomFrame)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 733, 22))
        self.menuImport = QMenu(self.menubar)
        self.menuImport.setObjectName(u"menuImport")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuImport.menuAction())
        self.menuImport.addAction(self.actionLoad_CSV)
        self.menuImport.addAction(self.actionLoad_Multiple)
        self.menuImport.addSeparator()
        self.menuImport.addAction(self.actionLoad_Project)
        self.menuImport.addAction(self.actionSave_Project)
        self.menuImport.addSeparator()
        self.menuImport.addAction(self.actionExport_CSV)

        self.retranslateUi(MainWindow)

        self.toolBox.setCurrentIndex(0)
        self.toolBox_2.setCurrentIndex(0)
        self.toolBox_3.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"FeaturesScore", None))
        self.actionload.setText(QCoreApplication.translate("MainWindow", u"Load File", None))
        self.actionimport.setText(QCoreApplication.translate("MainWindow", u"Load Folder", None))
        self.actionsave.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.actionLoad_CSV.setText(QCoreApplication.translate("MainWindow", u"Load CSV", None))
        self.actionLoad_Project.setText(QCoreApplication.translate("MainWindow", u"Load Project", None))
        self.actionLoad_Multiple.setText(QCoreApplication.translate("MainWindow", u"Load Multiple", None))
        self.actionSave_Project.setText(QCoreApplication.translate("MainWindow", u"Save Project", None))
        self.actionExport_CSV.setText(QCoreApplication.translate("MainWindow", u"Export CSV", None))
        self.actionAggregate_Data.setText(QCoreApplication.translate("MainWindow", u"Aggregate Data", None))
        self.hideIgnoredFeaturesCheckBox.setText(QCoreApplication.translate("MainWindow", u"Hide Ignored", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2), QCoreApplication.translate("MainWindow", u"Features", None))
        self.excludeSelectedDescriptorButton.setText(QCoreApplication.translate("MainWindow", u"Exclude Selected", None))
        self.keepOnlySelectedDescriptorButton.setText(QCoreApplication.translate("MainWindow", u"Keep Only Selected", None))
        self.renameDescriptorButton.setText(QCoreApplication.translate("MainWindow", u"Rename Selected", None))
        self.resetDescriptorFilterButton.setText(QCoreApplication.translate("MainWindow", u"Reset Filter", None))
        self.ApllyDescriptorCorrelationButton.setText(QCoreApplication.translate("MainWindow", u"Correlation", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Distance Threshold:", None))
        self.showCorrelationCheckBox.setText(QCoreApplication.translate("MainWindow", u"Show", None))
        self.filterDescriptorWithCorrelationCheckBox.setText(QCoreApplication.translate("MainWindow", u"Filter", None))
        self.resetDescriptorCorelationButton.setText(QCoreApplication.translate("MainWindow", u"Reset", None))
        self.squareDistCorrelationRadioButton.setText(QCoreApplication.translate("MainWindow", u"Square", None))
        self.absDistCorrelationRadioButton.setText(QCoreApplication.translate("MainWindow", u"Abs", None))
        self.diffDistCorrelationRadioButton.setText(QCoreApplication.translate("MainWindow", u"Default", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page), QCoreApplication.translate("MainWindow", u"Descriptor Selection", None))
        self.filterGroupButtonRemove.setText(QCoreApplication.translate("MainWindow", u"Remove Selected", None))
        self.filterGroupButtonKeepOnly.setText(QCoreApplication.translate("MainWindow", u"Keep Only Selected", None))
        self.featureGateButton.setText(QCoreApplication.translate("MainWindow", u"Advanced Filter", None))
#if QT_CONFIG(tooltip)
        self.filterGroupButtonBack.setToolTip(QCoreApplication.translate("MainWindow", u"Cancels last filter", None))
#endif // QT_CONFIG(tooltip)
        self.filterGroupButtonBack.setText(QCoreApplication.translate("MainWindow", u"\u2190", None))
#if QT_CONFIG(tooltip)
        self.filterGroupButtonReset.setToolTip(QCoreApplication.translate("MainWindow", u"Reset all filters", None))
#endif // QT_CONFIG(tooltip)
        self.filterGroupButtonReset.setText(QCoreApplication.translate("MainWindow", u"\u21ba", None))
        self.aggregateDataButton.setText(QCoreApplication.translate("MainWindow", u"Aggregate Data", None))
        self.EditGroupIndexValues.setText(QCoreApplication.translate("MainWindow", u"Edit", None))
        self.countsButton.setText(QCoreApplication.translate("MainWindow", u"Count", None))
        self.toolBox_2.setItemText(self.toolBox_2.indexOf(self.page_3), QCoreApplication.translate("MainWindow", u"Data Groups", None))
        self.barPlotButton.setText(QCoreApplication.translate("MainWindow", u"Bar Plot", None))
        self.boxPlotButton.setText(QCoreApplication.translate("MainWindow", u"Box Plot", None))
        self.densityPlotButton.setText(QCoreApplication.translate("MainWindow", u"Density Plot", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Color:", None))
        self.plotColorComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"Custom", None))
        self.plotColorComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Default", None))
        self.plotColorComboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"Blue", None))
        self.plotColorComboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"Gray", None))

        self.label.setText(QCoreApplication.translate("MainWindow", u"Style:", None))
        self.stylePlotComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"Default", None))
        self.stylePlotComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Classic", None))

#if QT_CONFIG(tooltip)
        self.stylePlotComboBox.setToolTip(QCoreApplication.translate("MainWindow", u"Plot Style", None))
#endif // QT_CONFIG(tooltip)
        self.scatterPlotButton.setText(QCoreApplication.translate("MainWindow", u"Scatter Plot", None))
        self.pcaButton.setText(QCoreApplication.translate("MainWindow", u"PCA Plot", None))
        self.ldaButton.setText(QCoreApplication.translate("MainWindow", u"LDA Plot", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Points", None))
        self.maxPointLineEdit.setText(QCoreApplication.translate("MainWindow", u"1000", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Axis:", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Best Correlation(s)", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Mode", None))
        self.pcaModeComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"Auto (S+D)", None))
        self.pcaModeComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Sample", None))
        self.pcaModeComboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"Descriptor", None))
        self.pcaModeComboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"Cluster", None))
        self.pcaModeComboBox.setItemText(4, QCoreApplication.translate("MainWindow", u"Target", None))

        self.summaryPlottButton.setText(QCoreApplication.translate("MainWindow", u"Summary Plot", None))
        self.crossTestingButton.setText(QCoreApplication.translate("MainWindow", u"Cross-Testing", None))
        self.ssmdRadioButton.setText(QCoreApplication.translate("MainWindow", u"SSMD", None))
        self.pValueRadioButton.setText(QCoreApplication.translate("MainWindow", u"P-Values", None))
        self.uValueRadioButton.setText(QCoreApplication.translate("MainWindow", u"U-Values", None))
        self.maskSummaryCheckBox.setText(QCoreApplication.translate("MainWindow", u"Mask", None))
        self.correctionSummaryCheckBox.setText(QCoreApplication.translate("MainWindow", u"Correction", None))
        self.radarPlotButton.setText(QCoreApplication.translate("MainWindow", u"Radar Plot", None))
        self.replicateRadarButton.setText(QCoreApplication.translate("MainWindow", u"Replicate ", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"SSMD Axis Limit:", None))
        self.sortRadarCheckBox.setText(QCoreApplication.translate("MainWindow", u"Sort Radar by SSMD Values", None))
        self.automaticAnalysisButton.setText(QCoreApplication.translate("MainWindow", u"FeaturesScore Analysis", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Significativity Threshold:", None))
        self.significativityTresholdLineEdit.setText(QCoreApplication.translate("MainWindow", u"0.05", None))
        self.toolBox_3.setItemText(self.toolBox_3.indexOf(self.page_4), QCoreApplication.translate("MainWindow", u"Plots", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Bags: ", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u" Preprocessing:", None))

        __sortingEnabled = self.preprocChoiceList.isSortingEnabled()
        self.preprocChoiceList.setSortingEnabled(False)
        ___qlistwidgetitem = self.preprocChoiceList.item(0)
        ___qlistwidgetitem.setText(QCoreApplication.translate("MainWindow", u"toto", None));
        self.preprocChoiceList.setSortingEnabled(__sortingEnabled)

        self.modelComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"Default RF", None))

        self.label_19.setText(QCoreApplication.translate("MainWindow", u"Model Name", None))
        self.modelTypeComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"SVM", None))
        self.modelTypeComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Decision Tree", None))
        self.modelTypeComboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"Random Forest", None))
        self.modelTypeComboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"MLP", None))

#if QT_CONFIG(tooltip)
        self.addModelButton.setToolTip(QCoreApplication.translate("MainWindow", u"Add new model", None))
#endif // QT_CONFIG(tooltip)
        self.addModelButton.setText(QCoreApplication.translate("MainWindow", u"+", None))
        self.toogleEditModelButton.setText(QCoreApplication.translate("MainWindow", u"Edit", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.classifOptimButton.setText(QCoreApplication.translate("MainWindow", u"Optimize", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"iter:", None))
        self.crossValidateButton.setText(QCoreApplication.translate("MainWindow", u"Cross Validate", None))
        self.testClassifButton.setText(QCoreApplication.translate("MainWindow", u"Test", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Scores:", None))
        self.modelFitButton.setText(QCoreApplication.translate("MainWindow", u"Fit", None))
        self.modelPredictButton.setText(QCoreApplication.translate("MainWindow", u"Predict", None))
        self.toolBox_3.setItemText(self.toolBox_3.indexOf(self.page_5), QCoreApplication.translate("MainWindow", u"Classification", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Display options:", None))
        self.displayINTRadioButton.setText(QCoreApplication.translate("MainWindow", u"Interactive", None))
        self.displayPNGRadioButton.setText(QCoreApplication.translate("MainWindow", u"PNG", None))
        self.displayPDFRadioButton.setText(QCoreApplication.translate("MainWindow", u"PDF", None))
        self.displayCSVRadioButton.setText(QCoreApplication.translate("MainWindow", u"CSV", None))
        self.menuImport.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
    # retranslateUi

