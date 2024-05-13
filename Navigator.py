from ui.NavigatorFrame import Ui_NavigatorFrame

from Card import Card
from ImportData import ImportData
from settings import Project

from PyQt6.QtWidgets import QWidget
from PyQt6 import QtCore

class Navigator(QWidget, Ui_NavigatorFrame):
    def __init__(self, project_obj: Project, parent=None):
        """
        mode:
            "dataset" for dataset
            "counts" for counts
            "resample" for resamples
            "pvalue" for pvalues
        """
        super(Navigator, self).__init__()
        self.setupUi(self)
        
        # Set up card grid to fill from top left, and going left to right
        self.CardGrid.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)

        

        self.project = project_obj

        self.populateItems(self.project.getChildren(), "dataset")

    
    def clearCardGrid(self):
        for i in reversed(range(self.CardGrid.count())):
            widget = self.CardGrid.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()


    def populateItems(self, items: dict, mode: str):
        self.mode = mode
        self.clearCardGrid()
        self.TypeLabel.setText(self.mode.upper() + " BROWSER")
        max_per_row = 3
        for i, item in enumerate(items.values()):
            card = Card(item, parent=self)
            # Add cards left to right
            self.CardGrid.addWidget(card, i // max_per_row, i % max_per_row)
        
        # Add a card to the end to allow for adding new items
        card = Card(parent=self)
        card.setToolTip(f"Add new {self.mode}")
        self.CardGrid.addWidget(card, len(items) // max_per_row, len(items) % max_per_row)
        self.setAddItemButton(card)

    def setAddItemButton(self, card: Card):
        if self.mode == "dataset":
            card.newItemButton.clicked.connect(self.addDataset)


    def addDataset(self):
        self.importData = ImportData()
        dataset = self.importData.run()
        if dataset is not None:
            self.project.addDataset(dataset.getDatasetID(), dataset)
            self.project.writeProject()
            self.populateItems(self.project.getChildren())

    
