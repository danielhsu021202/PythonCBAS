# Form implementation generated from reading ui file 'ui/FileViewerWidget.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_FileViewer(object):
    def setupUi(self, FileViewer):
        FileViewer.setObjectName("FileViewer")
        FileViewer.resize(955, 699)
        self.gridLayout = QtWidgets.QGridLayout(FileViewer)
        self.gridLayout.setObjectName("gridLayout")
        self.mainSplitter = QtWidgets.QSplitter(parent=FileViewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mainSplitter.sizePolicy().hasHeightForWidth())
        self.mainSplitter.setSizePolicy(sizePolicy)
        self.mainSplitter.setMinimumSize(QtCore.QSize(635, 0))
        self.mainSplitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.mainSplitter.setObjectName("mainSplitter")
        self.splitter = QtWidgets.QSplitter(parent=self.mainSplitter)
        self.splitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.splitter.setObjectName("splitter")
        self.verticalLayoutWidget = QtWidgets.QWidget(parent=self.splitter)
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.fileTree = QtWidgets.QTreeWidget(parent=self.verticalLayoutWidget)
        self.fileTree.setObjectName("fileTree")
        self.fileTree.headerItem().setText(0, "Files")
        self.verticalLayout.addWidget(self.fileTree)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(parent=self.splitter)
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.settingsBox = QtWidgets.QGroupBox(parent=self.verticalLayoutWidget_2)
        self.settingsBox.setObjectName("settingsBox")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.settingsBox)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.refreshFileTreeButton = QtWidgets.QPushButton(parent=self.settingsBox)
        self.refreshFileTreeButton.setObjectName("refreshFileTreeButton")
        self.verticalLayout_4.addWidget(self.refreshFileTreeButton)
        self.verticalLayout_2.addWidget(self.settingsBox)
        self.filterBox = QtWidgets.QGroupBox(parent=self.verticalLayoutWidget_2)
        self.filterBox.setObjectName("filterBox")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.filterBox)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.rowFilterCheckbox = QtWidgets.QCheckBox(parent=self.filterBox)
        self.rowFilterCheckbox.setObjectName("rowFilterCheckbox")
        self.gridLayout_5.addWidget(self.rowFilterCheckbox, 0, 0, 1, 1)
        self.rowFilterSetting = QtWidgets.QComboBox(parent=self.filterBox)
        self.rowFilterSetting.setObjectName("rowFilterSetting")
        self.rowFilterSetting.addItem("")
        self.gridLayout_5.addWidget(self.rowFilterSetting, 0, 2, 1, 1)
        self.rowFilterNumber = QtWidgets.QLineEdit(parent=self.filterBox)
        self.rowFilterNumber.setObjectName("rowFilterNumber")
        self.gridLayout_5.addWidget(self.rowFilterNumber, 0, 3, 1, 1)
        self.columnFilterNumber = QtWidgets.QLineEdit(parent=self.filterBox)
        self.columnFilterNumber.setObjectName("columnFilterNumber")
        self.gridLayout_5.addWidget(self.columnFilterNumber, 1, 3, 1, 1)
        self.columnFilterCheckbox = QtWidgets.QCheckBox(parent=self.filterBox)
        self.columnFilterCheckbox.setObjectName("columnFilterCheckbox")
        self.gridLayout_5.addWidget(self.columnFilterCheckbox, 1, 0, 1, 1)
        self.applyFiltersButton = QtWidgets.QPushButton(parent=self.filterBox)
        self.applyFiltersButton.setObjectName("applyFiltersButton")
        self.gridLayout_5.addWidget(self.applyFiltersButton, 2, 3, 1, 1)
        self.columnFilterSetting = QtWidgets.QComboBox(parent=self.filterBox)
        self.columnFilterSetting.setObjectName("columnFilterSetting")
        self.columnFilterSetting.addItem("")
        self.gridLayout_5.addWidget(self.columnFilterSetting, 1, 2, 1, 1)
        self.rowFilterRowNumber = QtWidgets.QLineEdit(parent=self.filterBox)
        self.rowFilterRowNumber.setObjectName("rowFilterRowNumber")
        self.gridLayout_5.addWidget(self.rowFilterRowNumber, 0, 1, 1, 1)
        self.columnFilterColumnNumber = QtWidgets.QLineEdit(parent=self.filterBox)
        self.columnFilterColumnNumber.setObjectName("columnFilterColumnNumber")
        self.gridLayout_5.addWidget(self.columnFilterColumnNumber, 1, 1, 1, 1)
        self.verticalLayout_2.addWidget(self.filterBox)
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(parent=self.mainSplitter)
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.dataLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.dataLayout.setContentsMargins(0, 0, 0, 0)
        self.dataLayout.setObjectName("dataLayout")
        self.dataTable = QtWidgets.QTableWidget(parent=self.verticalLayoutWidget_4)
        self.dataTable.setObjectName("dataTable")
        self.dataTable.setColumnCount(0)
        self.dataTable.setRowCount(0)
        self.dataLayout.addWidget(self.dataTable)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pushButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_4)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_3.addWidget(self.pushButton)
        self.dataLayout.addLayout(self.horizontalLayout_3)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(parent=self.mainSplitter)
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label = QtWidgets.QLabel(parent=self.verticalLayoutWidget_3)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.actionsBox = QtWidgets.QGroupBox(parent=self.verticalLayoutWidget_3)
        self.actionsBox.setObjectName("actionsBox")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.actionsBox)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.countRowsButton = QtWidgets.QPushButton(parent=self.actionsBox)
        self.countRowsButton.setObjectName("countRowsButton")
        self.gridLayout_4.addWidget(self.countRowsButton, 0, 0, 1, 1)
        self.countColumnsButton = QtWidgets.QPushButton(parent=self.actionsBox)
        self.countColumnsButton.setObjectName("countColumnsButton")
        self.gridLayout_4.addWidget(self.countColumnsButton, 0, 1, 1, 1)
        self.verticalLayout_3.addWidget(self.actionsBox)
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.verticalLayoutWidget_3)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.infoTerminal = QtWidgets.QPlainTextEdit(parent=self.groupBox_2)
        self.infoTerminal.setObjectName("infoTerminal")
        self.gridLayout_2.addWidget(self.infoTerminal, 0, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.groupBox_2)
        self.groupBox = QtWidgets.QGroupBox(parent=self.verticalLayoutWidget_3)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.functionTerminal = QtWidgets.QPlainTextEdit(parent=self.groupBox)
        self.functionTerminal.setObjectName("functionTerminal")
        self.gridLayout_3.addWidget(self.functionTerminal, 0, 0, 1, 1)
        self.clearFunctionTerminalButton = QtWidgets.QPushButton(parent=self.groupBox)
        self.clearFunctionTerminalButton.setObjectName("clearFunctionTerminalButton")
        self.gridLayout_3.addWidget(self.clearFunctionTerminalButton, 1, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.groupBox)
        self.gridLayout.addWidget(self.mainSplitter, 1, 0, 1, 1)

        self.retranslateUi(FileViewer)
        self.rowFilterCheckbox.toggled['bool'].connect(self.rowFilterSetting.setEnabled) # type: ignore
        self.rowFilterCheckbox.toggled['bool'].connect(self.rowFilterNumber.setEnabled) # type: ignore
        self.columnFilterCheckbox.toggled['bool'].connect(self.columnFilterSetting.setEnabled) # type: ignore
        self.columnFilterCheckbox.toggled['bool'].connect(self.columnFilterNumber.setEnabled) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(FileViewer)

    def retranslateUi(self, FileViewer):
        _translate = QtCore.QCoreApplication.translate
        FileViewer.setWindowTitle(_translate("FileViewer", "Form"))
        self.fileTree.headerItem().setText(1, _translate("FileViewer", "Type"))
        self.settingsBox.setTitle(_translate("FileViewer", "Settings"))
        self.refreshFileTreeButton.setText(_translate("FileViewer", "Refresh"))
        self.filterBox.setTitle(_translate("FileViewer", "Filters"))
        self.rowFilterCheckbox.setText(_translate("FileViewer", "Row"))
        self.rowFilterSetting.setItemText(0, _translate("FileViewer", "="))
        self.columnFilterCheckbox.setText(_translate("FileViewer", "Column"))
        self.applyFiltersButton.setText(_translate("FileViewer", "Apply"))
        self.columnFilterSetting.setItemText(0, _translate("FileViewer", "="))
        self.pushButton.setText(_translate("FileViewer", "PushButton"))
        self.label.setText(_translate("FileViewer", "Info"))
        self.actionsBox.setTitle(_translate("FileViewer", "Actions"))
        self.countRowsButton.setText(_translate("FileViewer", "Sum Rows"))
        self.countColumnsButton.setText(_translate("FileViewer", "Sum Columns"))
        self.groupBox_2.setTitle(_translate("FileViewer", "Info Output"))
        self.groupBox.setTitle(_translate("FileViewer", "Function Output"))
        self.clearFunctionTerminalButton.setText(_translate("FileViewer", "Clear"))
