# Form implementation generated from reading ui file 'ui/mainwindow.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(853, 660)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.mainStack = QtWidgets.QStackedWidget(parent=self.centralwidget)
        self.mainStack.setObjectName("mainStack")
        self.ImportDataPage = QtWidgets.QWidget()
        self.ImportDataPage.setObjectName("ImportDataPage")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.ImportDataPage)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.mainStack.addWidget(self.ImportDataPage)
        self.FileViewerPage = QtWidgets.QWidget()
        self.FileViewerPage.setObjectName("FileViewerPage")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.FileViewerPage)
        self.verticalLayout.setObjectName("verticalLayout")
        self.mainStack.addWidget(self.FileViewerPage)
        self.NavigatorPage = QtWidgets.QWidget()
        self.NavigatorPage.setObjectName("NavigatorPage")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.NavigatorPage)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.mainStack.addWidget(self.NavigatorPage)
        self.gridLayout.addWidget(self.mainStack, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 853, 37))
        self.menubar.setObjectName("menubar")
        self.temporary_menu = QtWidgets.QMenu(parent=self.menubar)
        self.temporary_menu.setObjectName("temporary_menu")
        self.menuPage = QtWidgets.QMenu(parent=self.menubar)
        self.menuPage.setObjectName("menuPage")
        self.menuPreferences = QtWidgets.QMenu(parent=self.menubar)
        self.menuPreferences.setObjectName("menuPreferences")
        self.menuAppearance = QtWidgets.QMenu(parent=self.menuPreferences)
        self.menuAppearance.setObjectName("menuAppearance")
        self.menuTheme = QtWidgets.QMenu(parent=self.menuAppearance)
        self.menuTheme.setObjectName("menuTheme")
        self.menuImport = QtWidgets.QMenu(parent=self.menubar)
        self.menuImport.setObjectName("menuImport")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionGet_Sequences = QtGui.QAction(parent=MainWindow)
        self.actionGet_Sequences.setObjectName("actionGet_Sequences")
        self.actionDark_Theme = QtGui.QAction(parent=MainWindow)
        self.actionDark_Theme.setObjectName("actionDark_Theme")
        self.actionLight_Theme = QtGui.QAction(parent=MainWindow)
        self.actionLight_Theme.setObjectName("actionLight_Theme")
        self.actionAuto = QtGui.QAction(parent=MainWindow)
        self.actionAuto.setObjectName("actionAuto")
        self.actionImport_Data = QtGui.QAction(parent=MainWindow)
        self.actionImport_Data.setObjectName("actionImport_Data")
        self.actionFile_Viewer = QtGui.QAction(parent=MainWindow)
        self.actionFile_Viewer.setObjectName("actionFile_Viewer")
        self.actionGenerate_Groups = QtGui.QAction(parent=MainWindow)
        self.actionGenerate_Groups.setObjectName("actionGenerate_Groups")
        self.actionReset_Default = QtGui.QAction(parent=MainWindow)
        self.actionReset_Default.setObjectName("actionReset_Default")
        self.actionImport_Config = QtGui.QAction(parent=MainWindow)
        self.actionImport_Config.setObjectName("actionImport_Config")
        self.actionExport_Preferences = QtGui.QAction(parent=MainWindow)
        self.actionExport_Preferences.setObjectName("actionExport_Preferences")
        self.actionDarkTheme = QtGui.QAction(parent=MainWindow)
        self.actionDarkTheme.setCheckable(False)
        self.actionDarkTheme.setObjectName("actionDarkTheme")
        self.actionLightTheme = QtGui.QAction(parent=MainWindow)
        self.actionLightTheme.setCheckable(False)
        self.actionLightTheme.setObjectName("actionLightTheme")
        self.actionAutoTheme = QtGui.QAction(parent=MainWindow)
        self.actionAutoTheme.setCheckable(False)
        self.actionAutoTheme.setObjectName("actionAutoTheme")
        self.actionFont_Size = QtGui.QAction(parent=MainWindow)
        self.actionFont_Size.setObjectName("actionFont_Size")
        self.actionImport_Data_Dialog = QtGui.QAction(parent=MainWindow)
        self.actionImport_Data_Dialog.setObjectName("actionImport_Data_Dialog")
        self.actionCard = QtGui.QAction(parent=MainWindow)
        self.actionCard.setObjectName("actionCard")
        self.temporary_menu.addAction(self.actionGet_Sequences)
        self.temporary_menu.addAction(self.actionGenerate_Groups)
        self.temporary_menu.addAction(self.actionCard)
        self.menuPage.addSeparator()
        self.menuPage.addAction(self.actionImport_Data)
        self.menuPage.addAction(self.actionFile_Viewer)
        self.menuTheme.addAction(self.actionDarkTheme)
        self.menuTheme.addAction(self.actionLightTheme)
        self.menuTheme.addAction(self.actionAutoTheme)
        self.menuAppearance.addAction(self.menuTheme.menuAction())
        self.menuAppearance.addAction(self.actionFont_Size)
        self.menuPreferences.addAction(self.actionReset_Default)
        self.menuPreferences.addAction(self.actionImport_Config)
        self.menuPreferences.addAction(self.actionExport_Preferences)
        self.menuPreferences.addSeparator()
        self.menuPreferences.addAction(self.menuAppearance.menuAction())
        self.menuImport.addAction(self.actionImport_Data_Dialog)
        self.menubar.addAction(self.menuImport.menuAction())
        self.menubar.addAction(self.temporary_menu.menuAction())
        self.menubar.addAction(self.menuPage.menuAction())
        self.menubar.addAction(self.menuPreferences.menuAction())

        self.retranslateUi(MainWindow)
        self.mainStack.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.temporary_menu.setTitle(_translate("MainWindow", "Temporary Running"))
        self.menuPage.setTitle(_translate("MainWindow", "Page"))
        self.menuPreferences.setTitle(_translate("MainWindow", "Preferences"))
        self.menuAppearance.setTitle(_translate("MainWindow", "Appearance"))
        self.menuTheme.setTitle(_translate("MainWindow", "Theme"))
        self.menuImport.setTitle(_translate("MainWindow", "Import"))
        self.actionGet_Sequences.setText(_translate("MainWindow", "Get Sequences"))
        self.actionDark_Theme.setText(_translate("MainWindow", "Dark Theme"))
        self.actionLight_Theme.setText(_translate("MainWindow", "Light Theme"))
        self.actionAuto.setText(_translate("MainWindow", "Auto"))
        self.actionImport_Data.setText(_translate("MainWindow", "Import Data"))
        self.actionFile_Viewer.setText(_translate("MainWindow", "File Viewer"))
        self.actionGenerate_Groups.setText(_translate("MainWindow", "Generate Groups"))
        self.actionReset_Default.setText(_translate("MainWindow", "Reset Default"))
        self.actionImport_Config.setText(_translate("MainWindow", "Import Preferences"))
        self.actionExport_Preferences.setText(_translate("MainWindow", "Export Preferences"))
        self.actionDarkTheme.setText(_translate("MainWindow", "Dark Theme"))
        self.actionLightTheme.setText(_translate("MainWindow", "Light Theme"))
        self.actionAutoTheme.setText(_translate("MainWindow", "Auto"))
        self.actionFont_Size.setText(_translate("MainWindow", "Font Size"))
        self.actionImport_Data_Dialog.setText(_translate("MainWindow", "Import Data"))
        self.actionCard.setText(_translate("MainWindow", "Card"))
