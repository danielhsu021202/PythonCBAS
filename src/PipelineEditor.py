from ui.PipelineDialog import Ui_PipelineDialog

from PyQt6.QtWidgets import QDialog, QWidget
from PyQt6.QtCore import Qt

from utils import StringUtils

class PipelineDialog(QDialog, Ui_PipelineDialog):

    type_indices = {
        "FILTER": 0,
        "SELECT": 1,
        "RENAME": 2,
        "GROUP": 3,
        "TRANSFORM": 4
    }

    def __init__(self, col, parent=None):
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
        self.applyNowButton.clicked.connect(lambda: self.addStage(apply=True))
        self.cancelButton.clicked.connect(self.close)

        if col != "":
            print("Setting")
            self.indexEdit.setText(str(col))

    def changePage(self, idx):
        self.pipelineStack.setCurrentIndex(idx)

    def filterPageInteractions(self):
        self.axisCombo.currentIndexChanged.connect(lambda: self.indexEdit.show() if self.axisCombo.currentIndex() < 1 else self.indexEdit.hide())
        self.operationCombo.currentIndexChanged.connect(lambda: self.filterToggleMode(self.operationCombo.currentText()))
    
    def filterToggleMode(self, operation: str):
        if operation == "between":
            self.valueEdit.hide()
            self.rangeToEdit.show()
            self.rangeFromEdit.show()
            self.toLabel.show()
        else:
            self.valueEdit.show()
            self.rangeToEdit.hide()
            self.rangeFromEdit.hide()
            self.toLabel.hide()

    def showError(self, message):
        self.errorLabel.setText(message)
        self.errorLabel.setStyleSheet("color: red;")
        self.errorLabel.show()
    
    def addStage(self, apply=False):
        stage = None
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
        
        
        if apply:
            self.parent.applyStageNow(stage)
        else:
            self.parent.appendStageToList(stage)

        
        self.close()

    def getTypeIndex(self, type_str):
        return PipelineDialog.type_indices[type_str]
    



    
    def translateOperation(operation: str):
        """Translates the operation to a pandas query string."""
        if operation == "equal to":
            return "=="
        elif operation == "not equal to":
            return "!="
        elif operation == "greater than":
            return ">"
        elif operation == "less than":
            return "<"
        elif operation == "greater than or equal to":
            return ">="
        elif operation == "less than or equal to":
            return "<="
        elif operation == "between":
            return "between"
        elif operation == "in":
            return "in"
        else:
            return None
    
    def parseStage(stage: tuple, column_names: list, parent):
        """Parses the information in the filter stage, and returns a pandas query string."""
        # TODO: CLEAN THIS UP
        query = ""
        if stage[0] == "Filter":
            mode, axis, index, operation, value = stage[1:]
            operation_str = operation
            operation = PipelineDialog.translateOperation(operation)
            # (f"`{col}`") for col in column_names if not col.isnumeric()
            ticked_column_names = []
            for col in column_names:
                if not type(col) == int:
                    ticked_column_names.append(f"`{col}`")
                else:
                    ticked_column_names.append(str(col))
            if axis == "all cells":
                if operation == "between":
                    query = ' or '.join([f"({str(col)} >= {int(value[0])} and {str(col)} <= {int(value[1])})" for col in ticked_column_names])
                elif operation == "in":
                    values = StringUtils.parseComplexRange(value)
                    if values is None:
                        parent.showError("Filtering", "Error: Invalid range format for 'in' operation.")
                        return None
                    query = ' or '.join([f"({str(col)} in {values})" for col in ticked_column_names])
                else:
                    query = ' or '.join([f"({str(col)} {operation} {int(value)})" for col in ticked_column_names])
            elif axis == 'columns':
                # Determine whether the it's indices or column names
                indices = StringUtils.parseComplexRange(index)
                cols = []
                if indices:
                    for i in indices:
                        if i >= len(ticked_column_names):
                            parent.showError("Filtering", f"Error: Index {i} out of range.")
                            return None
                        cols.append(ticked_column_names[i])
                else:
                    try:
                        for col in index.split(","):
                            col = col.strip()  # Remove surrounding whitespace
                            if f"`{col}`" in ticked_column_names:
                                cols.append(f"`{col}`")
                            else:
                                parent.showError("Filtering", f"Error: Invalid column name: {col}.")
                                return None
                    except:
                        parent.showError("Filtering", "Error: Invalid column name format.")
                        return None
                if len(cols) == 0:
                    parent.showError("Filtering", "Error: No columns identified.")
                    return
                if operation == "between":
                    query = ' and '.join([f"({str(col)} >= {int(value[0])} and {str(col)} <= {int(value[1])})" for col in cols])
                elif operation == "in":
                    values = StringUtils.parseComplexRange(value)
                    if values is None:
                        parent.showError("Filtering", "Error: Invalid range format for 'in' operation.")
                        return None
                    query = ' and '.join([f"({str(col)} in {values})" for col in cols])
                else:
                    query = ' and '.join([f"({str(col)} {operation} {int(value)})" for col in cols])

            query = f"not ({query})" if mode == "Exclude" else query
            index_str = f"where {StringUtils.andSeparateStr(index, include_verb=True)}" if axis == "columns" else "where all columns are"
            value_str = ""
            if operation == "between":
                value_str = f"{value[0]} and {value[1]}"
            elif operation == "in":
                value_str = f"[{value}]"
            else:
                value_str = value
            message = f"Filtered to {mode} records {index_str} {operation_str} {value_str}."

            return query, message
            

