# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'keyActivationWindow.ui'
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
        Dialog.resize(400, 112)
        icon = QIcon()
        icon.addFile(u"OSlogo.ico", QSize(), QIcon.Normal, QIcon.Off)
        Dialog.setWindowIcon(icon)
        self.keyLabel = QLabel(Dialog)
        self.keyLabel.setObjectName(u"keyLabel")
        self.keyLabel.setGeometry(QRect(30, 10, 81, 16))
        self.keyPushButton = QPushButton(Dialog)
        self.keyPushButton.setObjectName(u"keyPushButton")
        self.keyPushButton.setGeometry(QRect(307, 80, 75, 24))
        self.KeyLineEdit = QLineEdit(Dialog)
        self.KeyLineEdit.setObjectName(u"KeyLineEdit")
        self.KeyLineEdit.setGeometry(QRect(30, 40, 351, 22))

        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Enter a product key", None))
        self.keyLabel.setText(QCoreApplication.translate("Dialog", u"Product key : ", None))
        self.keyPushButton.setText(QCoreApplication.translate("Dialog", u"Next", None))
    # retranslateUi

