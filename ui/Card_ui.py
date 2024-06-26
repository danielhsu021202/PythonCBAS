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
        Card.resize(545, 528)
        Card.setStyleSheet("")
        self.verticalLayout = QtWidgets.QVBoxLayout(Card)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.CardTypes = QtWidgets.QStackedWidget(parent=Card)
        self.CardTypes.setObjectName("CardTypes")
        self.NewItemCard = QtWidgets.QWidget()
        self.NewItemCard.setObjectName("NewItemCard")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.NewItemCard)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.newItemButton = QtWidgets.QPushButton(parent=self.NewItemCard)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.newItemButton.sizePolicy().hasHeightForWidth())
        self.newItemButton.setSizePolicy(sizePolicy)
        self.newItemButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("ui/../icons/add.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.newItemButton.setIcon(icon)
        self.newItemButton.setIconSize(QtCore.QSize(512, 512))
        self.newItemButton.setFlat(True)
        self.newItemButton.setObjectName("newItemButton")
        self.verticalLayout_4.addWidget(self.newItemButton)
        self.CardTypes.addWidget(self.NewItemCard)
        self.NormalCard = QtWidgets.QWidget()
        self.NormalCard.setObjectName("NormalCard")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.NormalCard)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.CardFrame = QtWidgets.QFrame(parent=self.NormalCard)
        self.CardFrame.setStyleSheet("")
        self.CardFrame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.CardFrame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.CardFrame.setObjectName("CardFrame")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.CardFrame)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.TypeLabel = QtWidgets.QLabel(parent=self.CardFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TypeLabel.sizePolicy().hasHeightForWidth())
        self.TypeLabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Nunito")
        font.setPointSize(18)
        font.setBold(True)
        self.TypeLabel.setFont(font)
        self.TypeLabel.setStyleSheet("")
        self.TypeLabel.setObjectName("TypeLabel")
        self.verticalLayout_3.addWidget(self.TypeLabel)
        self.TitleLabel = QtWidgets.QLabel(parent=self.CardFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TitleLabel.sizePolicy().hasHeightForWidth())
        self.TitleLabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Nunito")
        font.setPointSize(24)
        font.setBold(True)
        self.TitleLabel.setFont(font)
        self.TitleLabel.setStyleSheet("")
        self.TitleLabel.setObjectName("TitleLabel")
        self.verticalLayout_3.addWidget(self.TitleLabel)
        self.SubtitleLabel = QtWidgets.QLabel(parent=self.CardFrame)
        font = QtGui.QFont()
        font.setFamily("Nunito")
        self.SubtitleLabel.setFont(font)
        self.SubtitleLabel.setObjectName("SubtitleLabel")
        self.verticalLayout_3.addWidget(self.SubtitleLabel)
        spacerItem = QtWidgets.QSpacerItem(20, 138, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_3.addItem(spacerItem)
        self.verticalLayout_2.addWidget(self.CardFrame)
        self.CardTypes.addWidget(self.NormalCard)
        self.verticalLayout.addWidget(self.CardTypes)

        self.retranslateUi(Card)
        self.CardTypes.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(Card)

    def retranslateUi(self, Card):
        _translate = QtCore.QCoreApplication.translate
        Card.setWindowTitle(_translate("Card", "Form"))
        self.TypeLabel.setText(_translate("Card", "Dataset"))
        self.TitleLabel.setText(_translate("Card", "My Dataset"))
        self.SubtitleLabel.setText(_translate("Card", "path/to/dataset/"))
