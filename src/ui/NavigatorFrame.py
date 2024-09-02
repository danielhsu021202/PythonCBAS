# Form implementation generated from reading ui file 'ui/Navigator.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_NavigatorFrame(object):
    def setupUi(self, NavigatorFrame):
        NavigatorFrame.setObjectName("NavigatorFrame")
        NavigatorFrame.resize(704, 490)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(NavigatorFrame)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.TypeLabel = QtWidgets.QLabel(parent=NavigatorFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TypeLabel.sizePolicy().hasHeightForWidth())
        self.TypeLabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Nunito")
        font.setPointSize(32)
        font.setBold(True)
        self.TypeLabel.setFont(font)
        self.TypeLabel.setObjectName("TypeLabel")
        self.verticalLayout_3.addWidget(self.TypeLabel)
        self.widget = QtWidgets.QWidget(parent=NavigatorFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMaximumSize(QtCore.QSize(16777215, 50))
        self.widget.setObjectName("widget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.backButton = QtWidgets.QPushButton(parent=self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.backButton.sizePolicy().hasHeightForWidth())
        self.backButton.setSizePolicy(sizePolicy)
        self.backButton.setObjectName("backButton")
        self.horizontalLayout_2.addWidget(self.backButton)
        spacerItem = QtWidgets.QSpacerItem(465, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.fileViewerButton = QtWidgets.QPushButton(parent=self.widget)
        self.fileViewerButton.setObjectName("fileViewerButton")
        self.horizontalLayout_2.addWidget(self.fileViewerButton)
        self.verticalLayout_3.addWidget(self.widget)
        self.widget_2 = QtWidgets.QWidget(parent=NavigatorFrame)
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_3.addWidget(self.widget_2)
        self.pathLabel = QtWidgets.QLabel(parent=NavigatorFrame)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.pathLabel.setFont(font)
        self.pathLabel.setObjectName("pathLabel")
        self.verticalLayout_3.addWidget(self.pathLabel)
        self.scrollArea = QtWidgets.QScrollArea(parent=NavigatorFrame)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 678, 319))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setContentsMargins(12, 12, 12, 12)
        self.verticalLayout.setObjectName("verticalLayout")
        self.CardGrid = QtWidgets.QGridLayout()
        self.CardGrid.setObjectName("CardGrid")
        self.verticalLayout.addLayout(self.CardGrid)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_3.addWidget(self.scrollArea)

        self.retranslateUi(NavigatorFrame)
        QtCore.QMetaObject.connectSlotsByName(NavigatorFrame)

    def retranslateUi(self, NavigatorFrame):
        _translate = QtCore.QCoreApplication.translate
        NavigatorFrame.setWindowTitle(_translate("NavigatorFrame", "Frame"))
        self.TypeLabel.setText(_translate("NavigatorFrame", "TextLabel"))
        self.backButton.setText(_translate("NavigatorFrame", "Back"))
        self.fileViewerButton.setText(_translate("NavigatorFrame", "Open in FileViewer"))
        self.pathLabel.setText(_translate("NavigatorFrame", "TextLabel"))