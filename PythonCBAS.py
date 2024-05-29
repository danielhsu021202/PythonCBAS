from settings import Settings

import sys
import os
import datetime

from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog, QTableWidgetItem, QTableWidget, QMenu
from PyQt6.QtCore import Qt

import qdarktheme

from ui.MainWindow import Ui_MainWindow
from ui.Lobby import Ui_Lobby

from Navigator import Navigator
from settings import Project, Preferences



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

        # Right click menu
        self.recentlyOpenedTable.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.recentlyOpenedTable.customContextMenuRequested.connect(self.recentlyOpenedMenu)

        self.returnValue = None

    def run(self):
        """Run the dialog and return the project object."""
        self.exec()
        return self.returnValue
    
    def getDirectory(self):
        """
        Open a file dialog to get a directory
        Default is the Documents folder
        """
        dir = QFileDialog.getExistingDirectory(self, "Select Directory", directory=Settings.getDocumentsFolder())
        self.projectLocationField.setText(dir)

    def getProject(self):
        """Open a file dialog to get .json or .cbasproj file"""
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Project", filter="PythonCBAS Project (*.json *.cbasproj)", directory=Settings.getDocumentsFolder())
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
        filepath = os.path.join(self.projectLocationField.text(), self.projectNameField.text() + ".cbasproj")  ## CHANGE FILE TYPE HERE
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
        self.recentlyOpenedTable.setRowCount(0)

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

    def recentlyOpenedMenu(self, pos):
        """Spawn a context menu when right clicking on the recently opened table."""
        menu = QMenu()
        openAction = menu.addAction("Open")
        removeAction = menu.addAction("Remove")
        action = menu.exec(self.recentlyOpenedTable.viewport().mapToGlobal(pos))
        if action == openAction:
            self.loadProject(self.recentlyOpenedTable.itemAt(pos).data(Qt.ItemDataRole.UserRole))
        elif action == removeAction:
            self.preferences.removeRecentlyOpened(self.recentlyOpenedTable.itemAt(pos).data(Qt.ItemDataRole.UserRole))
            self.setupRecentlyOpened()
                              


class PythonCBAS(QMainWindow, Ui_MainWindow):
    def __init__(self, project: Project, preferences, qss, close_callback):
        super().__init__()
        self.setupUi(self)

        self.preferences = preferences
        self.qss = qss
        self.close_callback = close_callback
        
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

    def setUpMenuBar(self):
        self.menubar = self.menuBar()
        # self.menubar.setNativeMenuBar(False)  # For macOS
        self.actionDarkTheme.triggered.connect(lambda: qdarktheme.setup_theme("dark", additional_qss=self.qss))
        self.actionLightTheme.triggered.connect(lambda: qdarktheme.setup_theme("light"))
        self.actionAutoTheme.triggered.connect(lambda: qdarktheme.setup_theme("auto", additional_qss=self.qss))
        self.actionImport_Data.triggered.connect(lambda: self.mainStack.setCurrentIndex(0))
        self.actionFile_Viewer.triggered.connect(lambda: self.mainStack.setCurrentIndex(1))
        self.actionExport_Preferences.triggered.connect(self.exportPreferences)
        self.actionImport_Preferences.triggered.connect(self.importPreferences)
        self.actionReset_Preferences.triggered.connect(self.resetPreferences)

    def exportPreferences(self):
        """Export the preferences to a file."""
        filepath, _ = QFileDialog.getSaveFileName(self, "Export Preferences", filter="JSON Files (*.json)", directory=Settings.getDocumentsFolder())
        if filepath:
            self.preferences.exportPreferences(filepath)

    def importPreferences(self):
        """Import preferences from a file."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Import Preferences", filter="JSON Files (*.json)", directory=Settings.getDocumentsFolder())
        if filepath:
            self.preferences.importPreferences(filepath)

    def resetPreferences(self):
        """Reset the preferences to default."""
        self.preferences.resetPreferences()


    def closeEvent(self, event):
        """When the window is closed, spawn the lobby."""
        self.close()
        self.close_callback()
        
        
class Application:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.setup_app()

    def setup_app(self):
        # Set current working directory to this file
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        
        # Load App Configurations
        appdatafolder = Settings.getAppDataFolder()
        if not os.path.exists(appdatafolder):
            os.mkdir(appdatafolder)

        preferences_filepath = os.path.join(appdatafolder, "preferences.json")
        self.preferences = Preferences(preferences_filepath)
        

        # Load ui/styles.qss
        with open("ui/styles.qss", "r") as f:
            self.qss = f.read()

        qdarktheme.setup_theme("auto", additional_qss=self.qss)

    def run(self):
        self.show_lobby()
        sys.exit(self.app.exec())

    def show_lobby(self):
        self.lobby = Lobby(self.preferences)
        project = self.lobby.run()
        if project is not None:
            self.show_main_window(project)
        else:
            sys.exit()

    def show_main_window(self, project):
        self.main_window = PythonCBAS(project, self.preferences, self.qss, self.show_lobby)
        self.main_window.show()

if __name__ == "__main__":
    app = Application()
    app.run()

        
