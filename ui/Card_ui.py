# Form implementation generated from reading ui file 'ui/card.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Card(object):
    def setupUi(self, Card):
        Card.setObjectName("Card")
        Card.resize(276, 167)
        Card.setStyleSheet("")
        self.verticalLayout = QtWidgets.QVBoxLayout(Card)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.CardFrame = QtWidgets.QFrame(parent=Card)
        self.CardFrame.setStyleSheet("")
        self.CardFrame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.CardFrame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.CardFrame.setObjectName("CardFrame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.CardFrame)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(parent=self.CardFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Nunito")
        font.setPointSize(24)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setStyleSheet("")
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(parent=self.CardFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Nunito")
        font.setPointSize(18)
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("")
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(parent=self.CardFrame)
        font = QtGui.QFont()
        font.setFamily("Nunito")
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        spacerItem = QtWidgets.QSpacerItem(20, 38, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.verticalLayout.addWidget(self.CardFrame)

        self.retranslateUi(Card)
        QtCore.QMetaObject.connectSlotsByName(Card)

    def retranslateUi(self, Card):
        _translate = QtCore.QCoreApplication.translate
        Card.setWindowTitle(_translate("Card", "Form"))
        self.label.setText(_translate("Card", "My Dataset"))
        self.label_2.setText(_translate("Card", "Dataset"))
        self.label_3.setText(_translate("Card", "path/to/dataset/"))
