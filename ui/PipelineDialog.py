# Form implementation generated from reading ui file 'ui/PipelineDialog.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_PipelineDialog(object):
    def setupUi(self, PipelineDialog):
        PipelineDialog.setObjectName("PipelineDialog")
        PipelineDialog.resize(284, 413)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PipelineDialog.sizePolicy().hasHeightForWidth())
        PipelineDialog.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(PipelineDialog)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_7 = QtWidgets.QLabel(parent=PipelineDialog)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_2.addWidget(self.label_7)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.typeSelector = QtWidgets.QComboBox(parent=PipelineDialog)
        self.typeSelector.setMinimumSize(QtCore.QSize(0, 25))
        self.typeSelector.setObjectName("typeSelector")
        self.typeSelector.addItem("")
        self.typeSelector.addItem("")
        self.typeSelector.addItem("")
        self.typeSelector.addItem("")
        self.typeSelector.addItem("")
        self.horizontalLayout_5.addWidget(self.typeSelector)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.line = QtWidgets.QFrame(parent=PipelineDialog)
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        self.pipelineStack = QtWidgets.QStackedWidget(parent=PipelineDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pipelineStack.sizePolicy().hasHeightForWidth())
        self.pipelineStack.setSizePolicy(sizePolicy)
        self.pipelineStack.setObjectName("pipelineStack")
        self.filterPage = QtWidgets.QWidget()
        self.filterPage.setObjectName("filterPage")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.filterPage)
        self.verticalLayout_3.setContentsMargins(0, -1, -1, -1)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.modeCombo = QtWidgets.QComboBox(parent=self.filterPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.modeCombo.sizePolicy().hasHeightForWidth())
        self.modeCombo.setSizePolicy(sizePolicy)
        self.modeCombo.setMinimumSize(QtCore.QSize(100, 25))
        self.modeCombo.setMaximumSize(QtCore.QSize(16777215, 25))
        self.modeCombo.setObjectName("modeCombo")
        self.modeCombo.addItem("")
        self.modeCombo.addItem("")
        self.verticalLayout_3.addWidget(self.modeCombo)
        self.label_6 = QtWidgets.QLabel(parent=self.filterPage)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_3.addWidget(self.label_6)
        self.axisCombo = QtWidgets.QComboBox(parent=self.filterPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.axisCombo.sizePolicy().hasHeightForWidth())
        self.axisCombo.setSizePolicy(sizePolicy)
        self.axisCombo.setMinimumSize(QtCore.QSize(100, 25))
        self.axisCombo.setMaximumSize(QtCore.QSize(16777215, 25))
        self.axisCombo.setObjectName("axisCombo")
        self.axisCombo.addItem("")
        self.axisCombo.addItem("")
        self.axisCombo.addItem("")
        self.verticalLayout_3.addWidget(self.axisCombo)
        self.indexEdit = QtWidgets.QLineEdit(parent=self.filterPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.indexEdit.sizePolicy().hasHeightForWidth())
        self.indexEdit.setSizePolicy(sizePolicy)
        self.indexEdit.setMinimumSize(QtCore.QSize(100, 25))
        self.indexEdit.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.indexEdit.setObjectName("indexEdit")
        self.verticalLayout_3.addWidget(self.indexEdit)
        self.label = QtWidgets.QLabel(parent=self.filterPage)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.operationCombo = QtWidgets.QComboBox(parent=self.filterPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.operationCombo.sizePolicy().hasHeightForWidth())
        self.operationCombo.setSizePolicy(sizePolicy)
        self.operationCombo.setMinimumSize(QtCore.QSize(100, 25))
        self.operationCombo.setMaximumSize(QtCore.QSize(16777215, 25))
        self.operationCombo.setObjectName("operationCombo")
        self.operationCombo.addItem("")
        self.operationCombo.addItem("")
        self.operationCombo.addItem("")
        self.operationCombo.addItem("")
        self.operationCombo.addItem("")
        self.verticalLayout_3.addWidget(self.operationCombo)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.valueEdit = QtWidgets.QLineEdit(parent=self.filterPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.valueEdit.sizePolicy().hasHeightForWidth())
        self.valueEdit.setSizePolicy(sizePolicy)
        self.valueEdit.setMinimumSize(QtCore.QSize(120, 25))
        self.valueEdit.setObjectName("valueEdit")
        self.verticalLayout.addWidget(self.valueEdit)
        self.rangeEditor = QtWidgets.QHBoxLayout()
        self.rangeEditor.setObjectName("rangeEditor")
        self.rangeFromEdit = QtWidgets.QLineEdit(parent=self.filterPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rangeFromEdit.sizePolicy().hasHeightForWidth())
        self.rangeFromEdit.setSizePolicy(sizePolicy)
        self.rangeFromEdit.setMinimumSize(QtCore.QSize(100, 25))
        self.rangeFromEdit.setObjectName("rangeFromEdit")
        self.rangeEditor.addWidget(self.rangeFromEdit)
        self.toLabel = QtWidgets.QLabel(parent=self.filterPage)
        self.toLabel.setObjectName("toLabel")
        self.rangeEditor.addWidget(self.toLabel)
        self.rangeToEdit = QtWidgets.QLineEdit(parent=self.filterPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rangeToEdit.sizePolicy().hasHeightForWidth())
        self.rangeToEdit.setSizePolicy(sizePolicy)
        self.rangeToEdit.setMinimumSize(QtCore.QSize(100, 25))
        self.rangeToEdit.setObjectName("rangeToEdit")
        self.rangeEditor.addWidget(self.rangeToEdit)
        self.verticalLayout.addLayout(self.rangeEditor)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        self.pipelineStack.addWidget(self.filterPage)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.page_2)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_4 = QtWidgets.QLabel(parent=self.page_2)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_6.addWidget(self.label_4)
        self.selectAxisCombo = QtWidgets.QComboBox(parent=self.page_2)
        self.selectAxisCombo.setObjectName("selectAxisCombo")
        self.selectAxisCombo.addItem("")
        self.selectAxisCombo.addItem("")
        self.verticalLayout_6.addWidget(self.selectAxisCombo)
        self.label_9 = QtWidgets.QLabel(parent=self.page_2)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_6.addWidget(self.label_9)
        self.selectIndexEdit = QtWidgets.QLineEdit(parent=self.page_2)
        self.selectIndexEdit.setObjectName("selectIndexEdit")
        self.verticalLayout_6.addWidget(self.selectIndexEdit)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_6.addItem(spacerItem1)
        self.pipelineStack.addWidget(self.page_2)
        self.renamePage = QtWidgets.QWidget()
        self.renamePage.setObjectName("renamePage")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.renamePage)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_2 = QtWidgets.QLabel(parent=self.renamePage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_4.addWidget(self.label_2)
        self.renameOriginalColumn = QtWidgets.QLineEdit(parent=self.renamePage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.renameOriginalColumn.sizePolicy().hasHeightForWidth())
        self.renameOriginalColumn.setSizePolicy(sizePolicy)
        self.renameOriginalColumn.setObjectName("renameOriginalColumn")
        self.verticalLayout_4.addWidget(self.renameOriginalColumn)
        self.label_3 = QtWidgets.QLabel(parent=self.renamePage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_4.addWidget(self.label_3)
        self.renameNewColumn = QtWidgets.QLineEdit(parent=self.renamePage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.renameNewColumn.sizePolicy().hasHeightForWidth())
        self.renameNewColumn.setSizePolicy(sizePolicy)
        self.renameNewColumn.setObjectName("renameNewColumn")
        self.verticalLayout_4.addWidget(self.renameNewColumn)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_4.addItem(spacerItem2)
        self.pipelineStack.addWidget(self.renamePage)
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.page)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_5 = QtWidgets.QLabel(parent=self.page)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_8.addWidget(self.label_5)
        self.groupByColumnEdit = QtWidgets.QLineEdit(parent=self.page)
        self.groupByColumnEdit.setObjectName("groupByColumnEdit")
        self.verticalLayout_8.addWidget(self.groupByColumnEdit)
        self.havingCheckBox = QtWidgets.QCheckBox(parent=self.page)
        self.havingCheckBox.setObjectName("havingCheckBox")
        self.verticalLayout_8.addWidget(self.havingCheckBox)
        self.havingLayout = QtWidgets.QWidget(parent=self.page)
        self.havingLayout.setEnabled(False)
        self.havingLayout.setObjectName("havingLayout")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.havingLayout)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.comboBox_2 = QtWidgets.QComboBox(parent=self.havingLayout)
        self.comboBox_2.setObjectName("comboBox_2")
        self.verticalLayout_7.addWidget(self.comboBox_2)
        self.lineEdit_2 = QtWidgets.QLineEdit(parent=self.havingLayout)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.verticalLayout_7.addWidget(self.lineEdit_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.lineEdit_3 = QtWidgets.QLineEdit(parent=self.havingLayout)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.horizontalLayout_3.addWidget(self.lineEdit_3)
        self.label_11 = QtWidgets.QLabel(parent=self.havingLayout)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_3.addWidget(self.label_11)
        self.lineEdit_4 = QtWidgets.QLineEdit(parent=self.havingLayout)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.horizontalLayout_3.addWidget(self.lineEdit_4)
        self.verticalLayout_7.addLayout(self.horizontalLayout_3)
        self.verticalLayout_8.addWidget(self.havingLayout)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_8.addItem(spacerItem3)
        self.pipelineStack.addWidget(self.page)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.page_3)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_8 = QtWidgets.QLabel(parent=self.page_3)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_5.addWidget(self.label_8)
        self.pipelineStack.addWidget(self.page_3)
        self.verticalLayout_2.addWidget(self.pipelineStack)
        self.errorLabel = QtWidgets.QLabel(parent=PipelineDialog)
        self.errorLabel.setObjectName("errorLabel")
        self.verticalLayout_2.addWidget(self.errorLabel)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.addStageButton = QtWidgets.QPushButton(parent=PipelineDialog)
        self.addStageButton.setObjectName("addStageButton")
        self.horizontalLayout_2.addWidget(self.addStageButton)
        self.cancelButton = QtWidgets.QPushButton(parent=PipelineDialog)
        self.cancelButton.setObjectName("cancelButton")
        self.horizontalLayout_2.addWidget(self.cancelButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.retranslateUi(PipelineDialog)
        self.pipelineStack.setCurrentIndex(0)
        self.havingCheckBox.toggled['bool'].connect(self.havingLayout.setEnabled) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(PipelineDialog)

    def retranslateUi(self, PipelineDialog):
        _translate = QtCore.QCoreApplication.translate
        PipelineDialog.setWindowTitle(_translate("PipelineDialog", "Add Pipeline Stage"))
        self.label_7.setText(_translate("PipelineDialog", "Stage Type:"))
        self.typeSelector.setItemText(0, _translate("PipelineDialog", "Filter"))
        self.typeSelector.setItemText(1, _translate("PipelineDialog", "Select"))
        self.typeSelector.setItemText(2, _translate("PipelineDialog", "Rename Column"))
        self.typeSelector.setItemText(3, _translate("PipelineDialog", "Group By"))
        self.typeSelector.setItemText(4, _translate("PipelineDialog", "Transpose"))
        self.modeCombo.setItemText(0, _translate("PipelineDialog", "Include"))
        self.modeCombo.setItemText(1, _translate("PipelineDialog", "Exclude"))
        self.label_6.setText(_translate("PipelineDialog", "from"))
        self.axisCombo.setItemText(0, _translate("PipelineDialog", "row numbers"))
        self.axisCombo.setItemText(1, _translate("PipelineDialog", "column numbers"))
        self.axisCombo.setItemText(2, _translate("PipelineDialog", "all cells"))
        self.indexEdit.setPlaceholderText(_translate("PipelineDialog", "row or column number(s)"))
        self.label.setText(_translate("PipelineDialog", "with values"))
        self.operationCombo.setItemText(0, _translate("PipelineDialog", "equal to"))
        self.operationCombo.setItemText(1, _translate("PipelineDialog", "less than"))
        self.operationCombo.setItemText(2, _translate("PipelineDialog", "greater than"))
        self.operationCombo.setItemText(3, _translate("PipelineDialog", "in"))
        self.operationCombo.setItemText(4, _translate("PipelineDialog", "between"))
        self.valueEdit.setPlaceholderText(_translate("PipelineDialog", "value(s)"))
        self.rangeFromEdit.setPlaceholderText(_translate("PipelineDialog", "start range"))
        self.toLabel.setText(_translate("PipelineDialog", "and"))
        self.rangeToEdit.setPlaceholderText(_translate("PipelineDialog", "end range"))
        self.label_4.setText(_translate("PipelineDialog", "Keep"))
        self.selectAxisCombo.setItemText(0, _translate("PipelineDialog", "rows"))
        self.selectAxisCombo.setItemText(1, _translate("PipelineDialog", "columns"))
        self.label_9.setText(_translate("PipelineDialog", "with name(s) or index(es)"))
        self.label_2.setText(_translate("PipelineDialog", "Rename Column"))
        self.renameOriginalColumn.setPlaceholderText(_translate("PipelineDialog", "original column (name or index)"))
        self.label_3.setText(_translate("PipelineDialog", "to"))
        self.renameNewColumn.setPlaceholderText(_translate("PipelineDialog", "new column name"))
        self.label_5.setText(_translate("PipelineDialog", "Group by columns"))
        self.havingCheckBox.setText(_translate("PipelineDialog", "Having values"))
        self.label_11.setText(_translate("PipelineDialog", "to"))
        self.label_8.setText(_translate("PipelineDialog", "Transpose"))
        self.errorLabel.setText(_translate("PipelineDialog", "TextLabel"))
        self.addStageButton.setText(_translate("PipelineDialog", "Add Stage"))
        self.cancelButton.setText(_translate("PipelineDialog", "Cancel"))