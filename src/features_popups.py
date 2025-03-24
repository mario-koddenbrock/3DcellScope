from src.external import *
from src.features_stat import DESCRIPTOR_TYPE
class FolderImportPopup(QDialog):
    def __init__(self):
        super(FolderImportPopup,self).__init__()
        self.setupUi(self)
        self.browsButtonFolderRoot.clicked.connect(self.folderBrowse)

    def setupUi(self, Import_options):
        if not Import_options.objectName():
            Import_options.setObjectName(u"Import_options")
        Import_options.resize(400, 182)
        self.buttonBox = QDialogButtonBox(Import_options)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setGeometry(QRect(55, 155, 341, 32))
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        self.lineEditFileName = QLineEdit(Import_options)
        self.lineEditFileName.setObjectName(u"lineEditFileName")
        self.lineEditFileName.setGeometry(QRect(155, 45, 146, 21))
        self.spinBoxFolderDepth = QSpinBox(Import_options)
        self.spinBoxFolderDepth.setObjectName(u"spinBoxFolderDepth")
        self.spinBoxFolderDepth.setGeometry(QRect(106, 46, 34, 21))
        self.label = QLabel(Import_options)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(26, 46, 51, 16))
        self.widget = QWidget(Import_options)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(25, 15, 366, 26))
        self.horizontalLayout_2 = QHBoxLayout(self.widget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.lineEditFolderRoot = QLineEdit(self.widget)
        self.lineEditFolderRoot.setObjectName(u"lineEditFolderRoot")

        self.horizontalLayout_2.addWidget(self.lineEditFolderRoot)

        self.browsButtonFolderRoot = QPushButton(self.widget)
        self.browsButtonFolderRoot.setObjectName(u"browsButtonFolderRoot")

        self.horizontalLayout_2.addWidget(self.browsButtonFolderRoot)

        self.widget1 = QWidget(Import_options)
        self.widget1.setObjectName(u"widget1")
        self.widget1.setGeometry(QRect(25, 70, 361, 72))
        self.gridLayout = QGridLayout(self.widget1)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.label_3 = QLabel(self.widget1)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 0, 1, 1, 1)

        self.label_2 = QLabel(self.widget1)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)

        self.checkBoxAddDataframe = QCheckBox(self.widget1)
        self.checkBoxAddDataframe.setObjectName(u"checkBoxAddDataframe")

        self.gridLayout.addWidget(self.checkBoxAddDataframe, 1, 0, 1, 1)

        self.subFolderStringRange = QLineEdit(self.widget1)
        self.subFolderStringRange.setObjectName(u"subFolderStringRange")

        self.gridLayout.addWidget(self.subFolderStringRange, 1, 1, 1, 1)

        self.subFolderNamesStringSplit = QLineEdit(self.widget1)
        self.subFolderNamesStringSplit.setObjectName(u"subFolderNamesStringSplit")

        self.gridLayout.addWidget(self.subFolderNamesStringSplit, 1, 2, 1, 1)

        self.checkBoxAddFileName = QCheckBox(self.widget1)
        self.checkBoxAddFileName.setObjectName(u"checkBoxAddFileName")

        self.gridLayout.addWidget(self.checkBoxAddFileName, 2, 0, 1, 1)

        self.fileNameStringRange = QLineEdit(self.widget1)
        self.fileNameStringRange.setObjectName(u"fileNameStringRange")

        self.gridLayout.addWidget(self.fileNameStringRange, 2, 1, 1, 1)

        self.fileNameStringSplit = QLineEdit(self.widget1)
        self.fileNameStringSplit.setObjectName(u"fileNameStringSplit")

        self.gridLayout.addWidget(self.fileNameStringSplit, 2, 2, 1, 1)

        self.label_4 = QLabel(self.widget1)
        self.label_4.setObjectName(u"label_4")
        font = QFont()
        font.setBold(True)
        self.label_4.setFont(font)

        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)


        self.retranslateUi(Import_options)
        self.buttonBox.accepted.connect(Import_options.accept)
        self.buttonBox.rejected.connect(Import_options.reject)

        QMetaObject.connectSlotsByName(Import_options)
    # setupUi

    def retranslateUi(self, Import_options):
        Import_options.setWindowTitle(QCoreApplication.translate("Import_options", u"Import Options", None))
        self.lineEditFileName.setPlaceholderText(QCoreApplication.translate("Import_options", u"file name", None))
        self.label.setText(QCoreApplication.translate("Import_options", u"file depht", None))
        self.lineEditFolderRoot.setPlaceholderText(QCoreApplication.translate("Import_options", u"folder root", None))
        self.browsButtonFolderRoot.setText(QCoreApplication.translate("Import_options", u"Browse", None))
        self.label_3.setText(QCoreApplication.translate("Import_options", u"select_range", None))
        self.label_2.setText(QCoreApplication.translate("Import_options", u"split name:", None))
        self.checkBoxAddDataframe.setText(QCoreApplication.translate("Import_options", u"subfolder names", None))
        self.subFolderStringRange.setPlaceholderText(QCoreApplication.translate("Import_options", u"1:-1", None))
        self.checkBoxAddFileName.setText(QCoreApplication.translate("Import_options", u"file name", None))
        self.fileNameStringRange.setText(QCoreApplication.translate("Import_options", u"1:-1", None))
        self.label_4.setText(QCoreApplication.translate("Import_options", u"Add columns ", None))
    # retranslateUi
    def folderBrowse(self):
        f_path = Path(QFileDialog.getExistingDirectory(None,"Parent folder"))
        if len(str(f_path))>1:
            self.lineEditFolderRoot.setText(str(f_path))
    def returnOptions(self):
        return{
            "folder_root":self.lineEditFolderRoot.text()
            ,"depth": self.spinBoxFolderDepth.value()
            ,"file_name":self.lineEditFileName.text()
            ,"add_subfolder":self.checkBoxAddDataframe.isChecked()
            ,"subfolder_range": self.subFolderStringRange.text()
            ,"subfolder_split": self.subFolderNamesStringSplit.text()
            ,"add_fileName":self.checkBoxAddFileName.isChecked()
            ,"file_range":self.fileNameStringRange.text()
            ,"file_split":self.fileNameStringSplit.text()
        }
class AddOptions(QDialog):
    def __init__(self):
        super(AddOptions,self).__init__()
        self.setupUi(self)
    def setupUi(self, AddFilterOptions):
        if not AddFilterOptions.objectName():
            AddFilterOptions.setObjectName(u"AddFilterOptions")
        AddFilterOptions.resize(540, 101)
        sizePolicy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(AddFilterOptions.sizePolicy().hasHeightForWidth())
        AddFilterOptions.setSizePolicy(sizePolicy)
        AddFilterOptions.setSizeGripEnabled(True)
        self.horizontalLayout_2 = QHBoxLayout(AddFilterOptions)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setSizeConstraint(QLayout.SetDefaultConstraint)

        self.gridLayout.addLayout(self.verticalLayout, 1, 0, 1, 1)

        self.buttonBox = QDialogButtonBox(AddFilterOptions)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.gridLayout.addWidget(self.buttonBox, 2, 0, 1, 1)

        self.widget = QWidget(AddFilterOptions)
        self.widget.setObjectName(u"widget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy1)
        self.widget.setMinimumSize(QSize(0, 40))
        self.horizontalLayout = QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy2)
        self.label.setMinimumSize(QSize(0, 25))

        self.horizontalLayout.addWidget(self.label)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.pushButton = QPushButton(self.widget)
        self.pushButton.setObjectName(u"pushButton")
        sizePolicy2.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy2)
        self.pushButton.setMinimumSize(QSize(0, 25))
        self.pushButton.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout.addWidget(self.pushButton)


        self.gridLayout.addWidget(self.widget, 0, 0, 1, 1)


        self.horizontalLayout_2.addLayout(self.gridLayout)


        self.retranslateUi(AddFilterOptions)
        self.buttonBox.accepted.connect(AddFilterOptions.accept)
        self.buttonBox.rejected.connect(AddFilterOptions.reject)

        QMetaObject.connectSlotsByName(AddFilterOptions)
    # setupUi

    def retranslateUi(self, AddFilterOptions):
        AddFilterOptions.setWindowTitle(QCoreApplication.translate("AddFilterOptions", u"Dialog", None))
        self.label.setText(QCoreApplication.translate("AddFilterOptions", u"Advanced Filtering Option", None))
        self.pushButton.setText(QCoreApplication.translate("AddFilterOptions", u"Add", None))
    # retranslateUi

class AddAggregationOptions(AddOptions):
    def __init__(self, descriptors=[],aggregations=[]):
        super(AddAggregationOptions,self).__init__()
        self.descriptors = descriptors
        self.pushButton.clicked.connect(self.addAggregation)
        self.aggregation_widgets = []
        self.setWindowTitle("Aggregation Options")
        self.label.setText("Add Aggregation Options")
        for (feature,aggregator) in aggregations:
            self.addAggregation(feature=feature,aggregator=aggregator)

    def addAggregation(self,*args, feature='',aggregator='',):
        widge = QWidget(None)
        h_layout = QHBoxLayout(widge)
        feature_choice = QComboBox(widge)
        feature_choice.addItems(list(self.descriptors.keys()))
        idf = feature_choice.findText(feature)
        if idf>=0:
            feature_choice.setCurrentIndex(idf)
        opperator_choice = QComboBox(widge)
        opperator_choice.addItems(["mean"])
        ido = opperator_choice.findText(aggregator)
        if ido>=0:
            opperator_choice.setCurrentIndex(ido)
        
        button = QPushButton(parent=widge,text="X")
        h_layout.addWidget(QLabel(parent=widge,text="aggregate along"))
        h_layout.addWidget(feature_choice)
        h_layout.addWidget(QLabel(parent=widge,text="using:"))
        h_layout.addWidget(opperator_choice)
        h_layout.addWidget(button)
        widge.setLayout(h_layout)
        self.verticalLayout.addWidget(widge)
        index = len(self.aggregation_widgets)
        button.clicked.connect(lambda: self.removeAggregation(index))
        self.aggregation_widgets.append(widge)
    def removeAggregation(self,index=0):
        item = self.aggregation_widgets[index]
        item.hide()
        self.aggregation_widgets[index]=None
    def results(self):
        out=[]
        for item in self.aggregation_widgets:
            if item is None:
                continue
            childeren = item.children()
            feature = childeren[1].currentText()
            aggegator = childeren[2].currentText()
            out.append((feature,aggegator))
        return out
class AddFilterOptions(AddOptions):
    def __init__(self, descriptors=[]):
        super(AddFilterOptions,self).__init__()
        self.descriptors = descriptors
        self.pushButton.clicked.connect(self.addFilter)
        self.filter_widgets = []
    def addFilter(self):
        widge = QWidget(None)
        h_layout = QHBoxLayout(widge)
        feature_choice = QComboBox(widge)
        feature_choice.addItems(list(self.descriptors.keys()))
        opperator_choice = QComboBox(widge)
        opperator_choice.addItems(["==","!=",">","<"])
        
        value_choice = QLineEdit(widge)
        button = QPushButton(parent=widge,text="X")
        opperator_choice.setEnabled(False)
        value_choice.setEnabled(False)
        h_layout.addWidget(feature_choice)
        h_layout.addWidget(opperator_choice)
        h_layout.addWidget(value_choice)
        h_layout.addWidget(button)
        widge.setLayout(h_layout)
        self.verticalLayout.addWidget(widge)
        index = len(self.filter_widgets)
        feature_choice.currentIndexChanged.connect(lambda:self.feature_set(index))
        button.clicked.connect(lambda: self.removeFilter(index))
        self.filter_widgets.append(widge)
    def removeFilter(self,index=0):
        item = self.filter_widgets[index]
        item.hide()
        self.filter_widgets[index]=None
        # self.verticalLayout.removeItem(item)
        # item.deleteLater()
    def feature_set(self,index=0):
        item = self.filter_widgets[index]
        childeren = item.children()
        feature_name = childeren[1].currentText()
        desc = self.descriptors[feature_name]
        childeren[2].setEnabled(True)
        childeren[3].setEnabled(True)
        childeren[2].clear()
        childeren[2].addItems(["==","!="]+([">","<"] if desc.type.value ==1 else []))
    def results(self):
        out = []
        for item in self.filter_widgets:
            if item is None:
                continue
            childeren = item.children()
            feature = childeren[1].currentText()
            operator = childeren[2].currentText()
            value = childeren[3].text()
            if len(value)>0:
                out.append("""`%s` %s %s"""%(feature, operator, value))
        return out
