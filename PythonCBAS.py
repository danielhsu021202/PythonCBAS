from sequences import SequencesProcessor
from resampler import Resampler
from statistical_analyzer import StatisticalAnalyzer
from settings import Settings

import sys
import os
import time
import numpy as np
import re
import argparse
import datetime

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QFileDialog, QMessageBox, QDialog, QTableWidgetItem, QTableWidget
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt

import qdarktheme

from ui.MainWindow import Ui_MainWindow
from ui.Lobby import Ui_Lobby

from FileViewer import FileViewer
from Navigator import Navigator
from ImportData import ImportData
from Card import Card
from settings import Project, Preferences



from utils import FileUtils
from files import CBASFile

import datetime



class Lobby(QDialog, Ui_Lobby):
    def __init__(self, preferences: Preferences):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("PythonCBAS")

        self.preferences = preferences
        self.setupRecentlyOpened()

        self.mainStack.setCurrentIndex(0)
        self.newProjectButton.clicked.connect(lambda: self.mainStack.setCurrentIndex(1))
        self.cancelButton.clicked.connect(self.reset)
        self.createProjectButton.clicked.connect(self.createProject)
        self.projectLocationButton.clicked.connect(self.getDirectory)
        self.openProjectButton.clicked.connect(self.getProject)

        # Table highlight whole row at same time
        self.recentlyOpenedTable.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        self.returnValue = None

    def run(self):
        """Run the dialog and return the project object."""
        self.exec()
        return self.returnValue
    
    def getDirectory(self):
        """Open a file dialog to get a directory"""
        dir = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.projectLocationField.setText(dir)

    def getProject(self):
        """Open a file dialog to get .json or .cbasproj file"""
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Project", filter="PythonCBAS Project (*.json *.cbasproj)")
        if filepath:
            self.loadProject(filepath)

    def reset(self):
        """Reset the create project dialog to its initial state."""
        self.mainStack.setCurrentIndex(0)
        self.projectNameField.clear()
        self.projectLocationField.clear()
        self.descriptionTextEdit.clear()

    def createProject(self):
        """Process the information in the create project dialog and create a project object."""
        filepath = os.path.join(self.projectLocationField.text(), self.projectNameField.text() + ".json")
        if os.path.exists(filepath):
            QMessageBox.warning(self, "Project Exists", "A project with that name already exists in the specified location.")
            return
        project = Project()
        datecreated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        project.createProject(self.projectNameField.text(), self.descriptionTextEdit.toPlainText(), 
                              datecreated, os.path.join(self.projectLocationField.text()), "beta")
        project.writeProject()
        self.preferences.addRecentlyOpened(project.getFilepath())
        self.returnValue = project
        self.close()

    def loadProject(self, filepath):
        """Load a project from a .json or .cbasproj file."""
        project = Project()
        project.readProject(filepath)
        self.preferences.addRecentlyOpened(project.getFilepath())
        self.returnValue = project
        self.close()

    def setupRecentlyOpened(self):
        self.recentlyOpenedTable.clear()

        # Initialize first column size
        self.recentlyOpenedTable.setColumnWidth(0, 190)

        # Set headers
        self.recentlyOpenedTable.setHorizontalHeaderLabels(["Project", "Last Modified"])

        recently_opened = list(self.preferences.getRecentlyOpened())
        recently_opened.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        for i, filepath in enumerate(recently_opened):
            self.recentlyOpenedTable.insertRow(i)
            project = Project()
            project.readProject(filepath)
            nameitem = QTableWidgetItem(project.getName())
            nameitem.setData(Qt.ItemDataRole.UserRole, filepath)
            self.recentlyOpenedTable.setItem(i, 0, nameitem)
            dateitem = QTableWidgetItem(project.getProjectDateModified())
            dateitem.setData(Qt.ItemDataRole.UserRole, filepath)
            self.recentlyOpenedTable.setItem(i, 1, dateitem)

        self.recentlyOpenedTable.cellDoubleClicked.connect(lambda: self.loadProject(self.recentlyOpenedTable.currentItem().data(Qt.ItemDataRole.UserRole)))


        
                              


class PythonCBAS(QMainWindow, Ui_MainWindow):
    def __init__(self, project: Project):
        super().__init__()
        self.setupUi(self)
        
        self.project = project

        self.setWindowTitle("PythonCBAS: " + project.getName())

        self.setUpMenuBar()

        self.navigator = Navigator(project_obj=self.project)
        self.NavigatorPage.layout().addWidget(self.navigator)

        # Set size
        self.resize(1010, 750)
        # Center on screen
        self.move(QApplication.primaryScreen().geometry().center() - self.frameGeometry().center())
        # self.showFullScreen()
        
        self.mainStack.setCurrentIndex(2)


    def importData(self):
        self.importDataDialog = ImportData()
        self.importDataDialog.exec()

    def showExampleCard(self):
        dialog = QDialog()
        card = Card(dialog)

        layout = QVBoxLayout()
        layout.addWidget(card)

        dialog.setLayout(layout)

        dialog.exec()

    def setUpMenuBar(self):
        self.menubar = self.menuBar()
        # self.menubar.setNativeMenuBar(False)  # For macOS
        self.actionDarkTheme.triggered.connect(lambda: qdarktheme.setup_theme("dark", additional_qss=qss))
        self.actionLightTheme.triggered.connect(lambda: qdarktheme.setup_theme("light"))
        self.actionAutoTheme.triggered.connect(lambda: qdarktheme.setup_theme("auto", additional_qss=qss))
        self.actionImport_Data.triggered.connect(lambda: self.mainStack.setCurrentIndex(0))
        self.actionFile_Viewer.triggered.connect(lambda: self.mainStack.setCurrentIndex(1))
        self.actionImport_Data_Dialog.triggered.connect(self.importData)
        self.actionCard.triggered.connect(self.showExampleCard)
        


if __name__ == "__main__":

    # Set current working directory to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    
    app = QApplication(sys.argv)

    # Load App Configurations
    appdatafolder = os.path.join("AppData")
    if not os.path.exists(appdatafolder):
        os.mkdir(appdatafolder)

    filepath = os.path.join(appdatafolder, "preferences.json")
    preferences = Preferences(os.path.join(appdatafolder, "preferences.json"))
    if not os.path.exists(filepath):
        preferences.writePreferences()
    else:
        preferences.readPreferences(filepath)



    # Load ui/styles.qss
    with open("ui/styles.qss", "r") as f:
        qss = f.read()

    qdarktheme.setup_theme("auto", additional_qss=qss)

    lobby = Lobby(preferences)
    project = lobby.run()
    if project is not None:
        mainWindow = PythonCBAS(project)
        mainWindow.show()
    else:
        sys.exit()
    
    sys.exit(app.exec())

        
