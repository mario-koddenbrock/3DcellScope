# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'features_popups_loadfolder.ui'
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
from PySide6.QtWidgets import (QAbstractButton, QApplication, QCheckBox, QDialog,
    QDialogButtonBox, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSpinBox,
    QWidget)

class Ui_Import_options(object):
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
        self.layoutWidget = QWidget(Import_options)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(25, 15, 366, 26))
        self.horizontalLayout_2 = QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.lineEditFolderRoot = QLineEdit(self.layoutWidget)
        self.lineEditFolderRoot.setObjectName(u"lineEditFolderRoot")

        self.horizontalLayout_2.addWidget(self.lineEditFolderRoot)

        self.browsButtonFolderRoot = QPushButton(self.layoutWidget)
        self.browsButtonFolderRoot.setObjectName(u"browsButtonFolderRoot")

        self.horizontalLayout_2.addWidget(self.browsButtonFolderRoot)

        self.layoutWidget1 = QWidget(Import_options)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.layoutWidget1.setGeometry(QRect(25, 70, 361, 72))
        self.gridLayout = QGridLayout(self.layoutWidget1)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.label_3 = QLabel(self.layoutWidget1)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 0, 1, 1, 1)

        self.label_2 = QLabel(self.layoutWidget1)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)

        self.checkBoxAddDataframe = QCheckBox(self.layoutWidget1)
        self.checkBoxAddDataframe.setObjectName(u"checkBoxAddDataframe")

        self.gridLayout.addWidget(self.checkBoxAddDataframe, 1, 0, 1, 1)

        self.subFolderStringRange = QLineEdit(self.layoutWidget1)
        self.subFolderStringRange.setObjectName(u"subFolderStringRange")

        self.gridLayout.addWidget(self.subFolderStringRange, 1, 1, 1, 1)

        self.subFolderNamesStringSplit = QLineEdit(self.layoutWidget1)
        self.subFolderNamesStringSplit.setObjectName(u"subFolderNamesStringSplit")

        self.gridLayout.addWidget(self.subFolderNamesStringSplit, 1, 2, 1, 1)

        self.checkBoxAddFileName = QCheckBox(self.layoutWidget1)
        self.checkBoxAddFileName.setObjectName(u"checkBoxAddFileName")

        self.gridLayout.addWidget(self.checkBoxAddFileName, 2, 0, 1, 1)

        self.fileNameStringRange = QLineEdit(self.layoutWidget1)
        self.fileNameStringRange.setObjectName(u"fileNameStringRange")

        self.gridLayout.addWidget(self.fileNameStringRange, 2, 1, 1, 1)

        self.fileNameStringSplit = QLineEdit(self.layoutWidget1)
        self.fileNameStringSplit.setObjectName(u"fileNameStringSplit")

        self.gridLayout.addWidget(self.fileNameStringSplit, 2, 2, 1, 1)

        self.label_4 = QLabel(self.layoutWidget1)
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

