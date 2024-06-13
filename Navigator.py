from ui.NavigatorFrame import Ui_NavigatorFrame

from Card import Card
from ImportData import ImportData
from SettingsDialog import SettingsDialog
from settings import Project, DataSet, Counts, Resamples, Visualizations, prev_type
from FileViewer import FileViewer

from utils import StringUtils

from PyQt6.QtWidgets import QWidget, QMainWindow, QApplication, QMessageBox, QPushButton, QLabel, QHBoxLayout
from PyQt6 import QtCore

class Navigator(QWidget, Ui_NavigatorFrame):
    def __init__(self, project_obj: Project, parent=None):
        """
        """
        super(Navigator, self).__init__()
        self.setupUi(self)
        
        # Set up card grid to fill from top left, and going left to right
        self.CardGrid.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)

        self.backButton.clicked.connect(self.goBack)
        

        # self.project = project_obj
        self.obj = project_obj
        self.proj_obj = project_obj
        # self.prev_obj = None

        self.fileViewerButton.clicked.connect(lambda: self.spawnFileViewer(self.obj.getDir()))

        self.backButton.setDisabled(True)

        self.populateItems(self.obj, "dataset")


    def getProject(self):
        return self.proj_obj

    def goBack(self):
        if self.obj.getParent() is not None:
            self.populateItems(self.obj.getParent(), self.obj.getType())
        else:
            self.populateItems(self.obj, self.obj.getType())

    def clearCardGrid(self):
        for i in reversed(range(self.CardGrid.count())):
            widget = self.CardGrid.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()


    def populateItems(self, obj, mode: str):
        self.mode = mode
        self.obj = obj
        self.backButton.setDisabled(self.obj.getParent() is None)
        self.clearCardGrid()
        self.TypeLabel.setText(self.mode.upper() + " BROWSER")
        max_per_row = 3
        items = obj.getChildren()
        for i, item in enumerate(items):
            card = Card(item, parent=self)
            # Add cards left to right
            self.CardGrid.addWidget(card, i // max_per_row, i % max_per_row)
        
        # Add a card to the end to allow for adding new items
        card = Card(parent=self)
        card.setToolTip(f"Add new {self.mode}")
        self.CardGrid.addWidget(card, len(items) // max_per_row, len(items) % max_per_row)
        self.setAddItemButton(card)

        # Update the path label
        path = [obj.getName() for obj in self.obj.retracePath()]
        self.pathLabel.setText("  >  ".join(path))
        # self.buildPath(obj)

    def buildPath(self, obj):
        if self.pathWidget.layout() is not None:
            self.pathWidget.layout().deleteLater()
        path = [obj.getName() for obj in obj.retracePath()]
        layout = QHBoxLayout()
        for item in path:
            layout.addWidget(QPushButton(item))
        self.pathWidget.setLayout(layout)




    def setAddItemButton(self, card: Card):
        if self.mode == "dataset":
            card.newItemButton.clicked.connect(self.addDataset)
        elif self.mode == "counts":
            card.newItemButton.clicked.connect(self.addCounts)
        elif self.mode == "resamples":
            card.newItemButton.clicked.connect(self.addResamples)


    def addDataset(self):
        try:
            assert type(self.obj) == Project
            importData = ImportData(self.obj)
            dataset = importData.run()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during dataset import: {str(e)}")
            return

        try:
            if dataset is not None:
                self.obj.addDataset(dataset)
                self.obj.writeProject()
                self.populateItems(self.obj, "dataset")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during dataset creation: {str(e)}")
            return

    def addCounts(self):
        try:
            assert type(self.obj) == DataSet
            countsSettings = SettingsDialog("counts", proj_obj=self.proj_obj, parent_obj=self.obj, parent=self)
            counts = countsSettings.run()
            if counts is not None:
                self.obj.addCounts(counts)
                self.obj.writeProject()
                self.populateItems(self.obj, "counts")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during counts creation: {str(e)}")
            return

    def addResamples(self):
        try:
            assert type(self.obj) == Counts
            resamplesSettings = SettingsDialog("resamples", proj_obj=self.proj_obj, parent_obj=self.obj, parent=self)
            resamples = resamplesSettings.run()
            if resamples is not None:
                self.obj.addResamples(resamples)
                self.obj.writeProject()
                self.populateItems(self.obj, "resamples")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during resamples creation: {str(e)}")
            return


    def spawnFileViewer(self, directory):
        # Spawn an independent instance of the window
        window = QMainWindow(parent=self)
        viewer = FileViewer(directory)
        window.setCentralWidget(viewer)
        window.setWindowTitle(f"FileViewer: {self.obj.getName()} [{StringUtils.capitalizeFirstLetter(self.obj.getType())}]")
        # Set size
        window.resize(1500, 860)
        # Center on screen
        window.move(QApplication.primaryScreen().geometry().center() - self.frameGeometry().center())

        window.show()



    
