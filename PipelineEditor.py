from ui.PipelineDialog import Ui_PipelineDialog

from PyQt6.QtWidgets import QDialog, QWidget
from PyQt6.QtCore import Qt

class PipelineDialog(QDialog, Ui_PipelineDialog):

    type_indices = {
        "FILTER": 0,
        "SELECT": 1,
        "RENAME": 2,
        "GROUP": 3,
        "TRANSFORM": 4
    }

    def __init__(self, parent=None):
        super(PipelineDialog, self).__init__(parent)
        self.setupUi(self)
        self.parent = parent

        self.errorLabel.hide()

        self.pipelineStack.setCurrentIndex(0)
        self.typeSelector.setCurrentIndex(0)
        self.filterToggleMode(0)

        self.typeSelector.currentIndexChanged.connect(self.changePage)


        self.filterPageInteractions()

        self.addStageButton.clicked.connect(self.addStage)
        self.cancelButton.clicked.connect(self.close)

    def changePage(self, idx):
        self.pipelineStack.setCurrentIndex(idx)

    def filterPageInteractions(self):
        self.axisCombo.currentIndexChanged.connect(lambda: self.indexEdit.show() if self.axisCombo.currentIndex() <= 1 else self.indexEdit.hide())
        self.operationCombo.currentIndexChanged.connect(self.filterToggleMode)
    
    def filterToggleMode(self, idx):
        if idx <= 3:
            self.valueEdit.show()
            self.rangeToEdit.hide()
            self.rangeFromEdit.hide()
            self.toLabel.hide()
        else:
            self.valueEdit.hide()
            self.rangeToEdit.show()
            self.rangeFromEdit.show()
            self.toLabel.show()

    def showError(self, message):
        self.errorLabel.setText(message)
        self.errorLabel.show()
    
    def addStage(self):
        if self.typeSelector.currentIndex() == self.getTypeIndex("FILTER"):
            # Filter Stage
            if self.indexEdit.isVisible():
                if self.indexEdit.text() == "":
                    self.showError(f"Error: {self.axisCombo.currentText()} field is empty.")
                    return
            if self.valueEdit.isVisible():
                if self.valueEdit.text() == "":
                    self.showError(f"Error: Value field is empty.")
                    return
            else:
                if not self.rangeFromEdit.text() or not self.rangeToEdit.text():
                    self.showError(f"Error: One of the value fields is empty.")
                    return
                
            stage = (self.typeSelector.currentText(), 
                        self.modeCombo.currentText(),
                        self.axisCombo.currentText(),
                        self.indexEdit.text() if self.indexEdit.isVisible() else "",
                        self.operationCombo.currentText(),
                        self.valueEdit.text() if self.valueEdit.isVisible() else (self.rangeFromEdit.text(), self.rangeToEdit.text()))
            self.parent.appendStageToList(stage)

        
        self.close()

    def getTypeIndex(self, type_str):
        return PipelineDialog.type_indices[type_str]
