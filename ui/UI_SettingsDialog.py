# Form implementation generated from reading ui file 'ui/SettingsDialog.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_SettingsDialog(object):
    def setupUi(self, SettingsDialog):
        SettingsDialog.setObjectName("SettingsDialog")
        SettingsDialog.resize(415, 601)
        self.verticalLayout = QtWidgets.QVBoxLayout(SettingsDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_3 = QtWidgets.QGroupBox(parent=SettingsDialog)
        self.groupBox_3.setObjectName("groupBox_3")
        self.formLayout = QtWidgets.QFormLayout(self.groupBox_3)
        self.formLayout.setObjectName("formLayout")
        self.label_4 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_4)
        self.label_5 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_5)
        self.descPlainTextEdit = QtWidgets.QPlainTextEdit(parent=self.groupBox_3)
        self.descPlainTextEdit.setObjectName("descPlainTextEdit")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.descPlainTextEdit)
        self.nameLineEdit = QtWidgets.QLineEdit(parent=self.groupBox_3)
        self.nameLineEdit.setObjectName("nameLineEdit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.nameLineEdit)
        self.verticalLayout.addWidget(self.groupBox_3)
        self.SettingsPages = QtWidgets.QStackedWidget(parent=SettingsDialog)
        self.SettingsPages.setObjectName("SettingsPages")
        self.CountsSettingsPage = QtWidgets.QWidget()
        self.CountsSettingsPage.setObjectName("CountsSettingsPage")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.CountsSettingsPage)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox = QtWidgets.QGroupBox(parent=self.CountsSettingsPage)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(parent=self.groupBox)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.criterionOrderLineEdit = QtWidgets.QLineEdit(parent=self.groupBox)
        self.criterionOrderLineEdit.setObjectName("criterionOrderLineEdit")
        self.horizontalLayout.addWidget(self.criterionOrderLineEdit)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.scanButton = QtWidgets.QPushButton(parent=self.groupBox)
        self.scanButton.setEnabled(False)
        self.scanButton.setObjectName("scanButton")
        self.verticalLayout_3.addWidget(self.scanButton)
        self.progressLabel = QtWidgets.QLabel(parent=self.groupBox)
        self.progressLabel.setObjectName("progressLabel")
        self.verticalLayout_3.addWidget(self.progressLabel)
        self.progressBar = QtWidgets.QProgressBar(parent=self.groupBox)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_3.addWidget(self.progressBar)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.criterionNumberLineEdit = QtWidgets.QLineEdit(parent=self.groupBox)
        self.criterionNumberLineEdit.setObjectName("criterionNumberLineEdit")
        self.horizontalLayout_3.addWidget(self.criterionNumberLineEdit)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.CountsSettingsPage)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_3 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.maxSeqLenLineEdit = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.maxSeqLenLineEdit.setObjectName("maxSeqLenLineEdit")
        self.horizontalLayout_2.addWidget(self.maxSeqLenLineEdit)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.criterionIncludeFailedCheck = QtWidgets.QCheckBox(parent=self.groupBox_2)
        self.criterionIncludeFailedCheck.setEnabled(False)
        self.criterionIncludeFailedCheck.setChecked(True)
        self.criterionIncludeFailedCheck.setObjectName("criterionIncludeFailedCheck")
        self.verticalLayout_4.addWidget(self.criterionIncludeFailedCheck)
        self.criterionAllowRedemptionCheck = QtWidgets.QCheckBox(parent=self.groupBox_2)
        self.criterionAllowRedemptionCheck.setEnabled(False)
        self.criterionAllowRedemptionCheck.setChecked(True)
        self.criterionAllowRedemptionCheck.setObjectName("criterionAllowRedemptionCheck")
        self.verticalLayout_4.addWidget(self.criterionAllowRedemptionCheck)
        self.straddleSessionsCheck = QtWidgets.QCheckBox(parent=self.groupBox_2)
        self.straddleSessionsCheck.setEnabled(False)
        self.straddleSessionsCheck.setObjectName("straddleSessionsCheck")
        self.verticalLayout_4.addWidget(self.straddleSessionsCheck)
        self.verticalLayout_2.addWidget(self.groupBox_2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.SettingsPages.addWidget(self.CountsSettingsPage)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.SettingsPages.addWidget(self.page_2)
        self.verticalLayout.addWidget(self.SettingsPages)
        spacerItem1 = QtWidgets.QSpacerItem(20, 99, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem2)
        self.cancelButton = QtWidgets.QPushButton(parent=SettingsDialog)
        self.cancelButton.setMinimumSize(QtCore.QSize(100, 0))
        self.cancelButton.setObjectName("cancelButton")
        self.horizontalLayout_4.addWidget(self.cancelButton)
        self.createButton = QtWidgets.QPushButton(parent=SettingsDialog)
        self.createButton.setMinimumSize(QtCore.QSize(100, 0))
        self.createButton.setObjectName("createButton")
        self.horizontalLayout_4.addWidget(self.createButton)
        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.retranslateUi(SettingsDialog)
        QtCore.QMetaObject.connectSlotsByName(SettingsDialog)

    def retranslateUi(self, SettingsDialog):
        _translate = QtCore.QCoreApplication.translate
        SettingsDialog.setWindowTitle(_translate("SettingsDialog", "Dialog"))
        self.groupBox_3.setTitle(_translate("SettingsDialog", "Basic Info"))
        self.label_4.setText(_translate("SettingsDialog", "Name:"))
        self.label_5.setText(_translate("SettingsDialog", "<html><head/><body><p>Description:</p></body></html>"))
        self.descPlainTextEdit.setPlaceholderText(_translate("SettingsDialog", "Description (optional)"))
        self.nameLineEdit.setPlaceholderText(_translate("SettingsDialog", "Name"))
        self.groupBox.setTitle(_translate("SettingsDialog", "Criterion"))
        self.label.setText(_translate("SettingsDialog", "Order:"))
        self.criterionOrderLineEdit.setPlaceholderText(_translate("SettingsDialog", "(integer)"))
        self.scanButton.setText(_translate("SettingsDialog", "Scan"))
        self.progressLabel.setText(_translate("SettingsDialog", "TextLabel"))
        self.label_2.setText(_translate("SettingsDialog", "Number:"))
        self.criterionNumberLineEdit.setPlaceholderText(_translate("SettingsDialog", "(integer or \"inf\")"))
        self.groupBox_2.setTitle(_translate("SettingsDialog", "Sequence Settings"))
        self.label_3.setText(_translate("SettingsDialog", "Max Sequence Length:"))
        self.maxSeqLenLineEdit.setPlaceholderText(_translate("SettingsDialog", "(integer)"))
        self.criterionIncludeFailedCheck.setText(_translate("SettingsDialog", "Include Failed"))
        self.criterionAllowRedemptionCheck.setText(_translate("SettingsDialog", "Allow Redemption"))
        self.straddleSessionsCheck.setText(_translate("SettingsDialog", "Straddle Sessions"))
        self.cancelButton.setText(_translate("SettingsDialog", "Cancel"))
        self.createButton.setText(_translate("SettingsDialog", "Run"))