# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'features_popups_filter.ui'
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
from PySide6.QtWidgets import (QAbstractButton, QApplication, QDialog, QDialogButtonBox,
    QGridLayout, QHBoxLayout, QLabel, QLayout,
    QPushButton, QSizePolicy, QSpacerItem, QVBoxLayout,
    QWidget)

class Ui_AddFilterOptions(object):
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

