
from PyQt6.QtWidgets import QTableView, QWidget, QFileDialog, QTreeWidgetItem, QMenu, QTableWidgetItem, QAbstractItemView, QMessageBox
from PyQt6.QtGui import QIcon, QCursor
from PyQt6.QtCore import QAbstractTableModel, Qt

import os
import sys
import re
import pandas as pd
from statistics import median, mode

from ui.FileViewerWidget import Ui_FileViewer
from PipelineEditor import PipelineDialog
import os
from PyQt6.QtWidgets import QMenu
from PyQt6.QtCore import Qt

from utils import ListUtils

class PandasTableModel(QAbstractTableModel): 
    def __init__(self, df=pd.DataFrame(), parent=None): 
        super().__init__(parent)
        self._df = df

    def toDataFrame(self):
        return self._df

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if orientation == Qt.Orientation.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except IndexError:
                return None
        elif orientation == Qt.Orientation.Vertical:
            try:
                return self._df.index.tolist()[section]
            except IndexError:
                return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if not index.isValid():
            return None

        return str(self._df.iloc[index.row(), index.column()])

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt6 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide6 gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.at[row, col] = value
        return True

    def rowCount(self, parent=None): 
        return len(self._df.index)

    def columnCount(self, parent=None): 
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending=order == Qt.SortOrder.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()




class PandasTable(QTableView):
    def __init__(self, data, parent=None):
        super().__init__()
        self.df: pd.DataFrame = data
        self.parent = parent
        self.model = PandasTableModel(self.df)
        self.setModel(self.model)

        self.setAlternatingRowColors(True)

        self.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.horizontalHeader().customContextMenuRequested.connect(self.onHeaderContextMenuRequested)
        self.verticalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.verticalHeader().customContextMenuRequested.connect(self.onHeaderContextMenuRequested)

    def onHeaderContextMenuRequested(self, pos):
        menu = QMenu()
        sortAscendingAction = menu.addAction("Sort Ascending",)
        sortDescendingAction = menu.addAction("Sort Descending")
        sumAction = menu.addAction("Sum")
        averageAction = menu.addAction("Average")
        medianAction = menu.addAction("Median")
        modeAction = menu.addAction("Mode")
        rangeAction = menu.addAction("Range")
        filterAction = ""
        if self.sender() == self.horizontalHeader():
            filterAction = menu.addAction("Filter")
        action = menu.exec(self.mapToGlobal(pos))

        if action == sortAscendingAction:
            if self.sender() == self.horizontalHeader():
                self.sortColumn(self.columnAt(pos.x()), True)
            else:
                self.sortRow(self.rowAt(pos.y()), True)
        elif action == sortDescendingAction:
            if self.sender() == self.horizontalHeader():
                self.sortColumn(self.columnAt(pos.x()), False)
            else:
                self.sortRow(self.rowAt(pos.y()), False)
        elif action == sumAction:
            if self.sender() == self.horizontalHeader():
                col = self.columnAt(pos.x())
                self.writeToFuncTerminal(f"Sum of column {self.df.columns[col]}: {str(self.df[self.df.columns[col]].sum())}")
            else:
                row = self.rowAt(pos.y())
                self.writeToFuncTerminal(f"Sum of row {row}: {str(self.df.iloc[row].sum())}")
        elif action == averageAction:
            if self.sender() == self.horizontalHeader():
                col = self.columnAt(pos.x())
                self.writeToFuncTerminal(f"Average of column {self.df.columns[col]}: {str(self.df[self.df.columns[col]].mean())}")
            else:
                row = self.rowAt(pos.y())
                self.writeToFuncTerminal(f"Average of row {row}: {str(self.df.iloc[row].mean())}")
        elif action == medianAction:
            if self.sender() == self.horizontalHeader():
                col = self.columnAt(pos.x())
                selected_column = list(self.df[self.df.columns[col]])
                median_value = median(selected_column)
                self.writeToFuncTerminal(f"Median of column {self.df.columns[col]}: {str(median_value)}")
            else:
                row = self.rowAt(pos.y())
                selected_row = list(self.df.iloc[row])
                median_value = median(selected_row)
                self.writeToFuncTerminal(f"Median of row {row}: {str(median_value)}")
        elif action == modeAction:
            if self.sender() == self.horizontalHeader():
                col = self.columnAt(pos.x())
                selected_column = list(self.df[self.df.columns[col]])
                mode_value = mode(selected_column)
                self.writeToFuncTerminal(f"Mode of column {self.df.columns[col]}: {str(mode_value)}")
            else:
                row = self.rowAt(pos.y())
                selected_row = list(self.df.iloc[row])
                mode_value = mode(selected_row)
                self.writeToFuncTerminal(f"Mode of row {row}: {str(mode_value)}")
        elif action == rangeAction:
            if self.sender() == self.horizontalHeader():
                col = self.columnAt(pos.x())
                self.writeToFuncTerminal(f"Range of column {self.df.columns[col]}: [{str(self.df[self.df.columns[col]].min())}, {str(self.df[self.df.columns[col]].max())}]")
            else:
                row = self.rowAt(pos.y())
                self.writeToFuncTerminal(f"Range of row {row}: [{str(self.df.iloc[row].min())}, {str(self.df.iloc[row].max())}]")
        elif action == filterAction:
            # Summon the filter dialog, passing in the name of the column being called
            self.parent.addStage(col=self.df.columns[self.columnAt(pos.x())])


    def filterTable(self, query):
        self.df.query(query, inplace=True)
        self.updateTable()

    
    def sortColumn(self, column, order):
        self.df = self.df.sort_values(by=self.df.columns[column], axis=0, ascending=order)
        self.updateTable()

    def sortRow(self, row, order):
        self.df = self.df.sort_values(by=self.df.index[row], axis=1, ascending=order)
        self.updateTable()

    def writeToFuncTerminal(self, text):
        self.parent.functionTerminal.appendPlainText(text)

    def countRows(self):
        """Displays row count in the function terminal."""
        self.parent.functionTerminal.appendPlainText("Rows: " + str(len(self.df)) if self.df is not None else "No file selected.")

    def countColumns(self):
        """Displays column count in the function terminal."""
        self.parent.functionTerminal.appendPlainText("Columns: " + str(len(self.df.columns)) if self.df is not None else "No file selected.")

    def transpose(self):
        """Transposes the table."""
        self.df = self.df.transpose()
        self.updateTable(self.df)

    def countNaN(self):
        """Counts the number of NaN values in the table."""
        self.parent.functionTerminal.appendPlainText("NaNs: " + str(self.df.isnull().sum().sum()))


    
    def updateTable(self, df=None):
        if df is not None:
            self.df = df
        self.model = PandasTableModel(self.df)
        self.setModel(self.model)



        

class FileViewer(QWidget, Ui_FileViewer):

    supported_file_types = ['.cbas', '.txt', '.csv', '.pkl', '.npy']

    def __init__(self, parent=None):
        super(FileViewer, self).__init__(parent)
        self.setupUi(self)
        self.setDefaultSizes()

        self.directories = set([os.path.join("output"), os.path.join("metadata")])
        self.open_files = set()

        self.refreshFileTree()
        self.pd_table = None

        self.mapButtonActions()
        self.fileTree.itemDoubleClicked.connect(lambda item: self.fileTreeItemTriggered(item) if item.childCount() == 0 else None)

        self.setupFilterTable()




    def mapButtonActions(self):
        self.fileTreeImportButton.clicked.connect(self.importDirectory)
        self.refreshFileTreeButton.clicked.connect(self.refreshFileTree)
        self.countRowsButton.clicked.connect(lambda: self.tableAction("count_rows"))
        self.countColumnsButton.clicked.connect(lambda: self.tableAction("count_columns"))
        self.transposeButton.clicked.connect(lambda: self.tableAction("transpose"))
        self.countNaNButton.clicked.connect(lambda: self.tableAction("count_nan"))
        self.clearFunctionTerminalButton.clicked.connect(self.functionTerminal.clear)
        self.addFilterButton.clicked.connect(lambda: self.addStage(col=""))
        self.applyFiltersButton.clicked.connect(self.applyFilters)
        self.clearPipelineButton.clicked.connect(lambda: self.filterTable.setRowCount(0))
        self.fileTabs.tabCloseRequested.connect(self.closeFile)

    def setupFilterTable(self):
        # Context menu when a filter table row is right clicked
        self.filterTable.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.filterTable.customContextMenuRequested.connect(self.displayFilterContextMenu)

    def tableAction(self, action):
        if self.pd_table is None:
            self.functionTerminal.appendPlainText("No file selected.")
            return
        if action == "count_rows":
            self.pd_table.countRows()
        elif action == "count_columns":
            self.pd_table.countColumns()
        elif action == "transpose":
            self.pd_table.transpose()
        elif action == "count_nan":
            self.pd_table.countNaN()


    def displayFilterContextMenu(self, pos):
        """Display context menu if a row is right clicked"""
        index = self.filterTable.indexAt(pos)
        if index.isValid():
            menu = QMenu()
            editAction = menu.addAction("Edit")
            deleteAction = menu.addAction("Delete")
            action = menu.exec(QCursor.pos())
            if action == deleteAction:
                self.filterTable.removeRow(index.row())
            elif action == editAction:
                pass

        

    def importDirectory(self):
        """Opens a file dialog to import a directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.directories.add(directory)
            self.refreshFileTree()

    def refreshFileTree(self):
        """Refreshes the file tree view with the current directories."""
        self.fileTree.clear()
        for directory in ListUtils.naturalSort(self.directories, key=lambda d: os.path.dirname(d)):
            if os.path.isdir(directory):
                self.populateFileTree(directory, self.fileTree.invisibleRootItem())
            else:
                pass

        
    def populateFileTree(self, base_path, parent_item):
        """
        Recursively populates a QTreeView with the hierarchical structure of files and directories.
        """
        # Create a QTreeWidgetItem for the root directory
        root_item = QTreeWidgetItem(parent_item)
        root_item.setText(0, os.path.basename(base_path))  # Set the text for the root item
        root_item.setIcon(0, QIcon("icons/folder.png"))

        for item in ListUtils.naturalSort(os.listdir(base_path)):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                # Recursively populate the tree view with the contents of the directory
                self.populateFileTree(item_path, root_item)
            elif os.path.isfile(item_path) and item_path.endswith(tuple(self.supported_file_types)):
                # Create a QTreeWidgetItem for files
                file_item = QTreeWidgetItem(root_item)
                file_item.setText(0, item)  # Set the text for the item
                file_item.setData(0, Qt.ItemDataRole.UserRole, item_path)  # Set the data for the item

    def fileType(self, filepath):
        """Returns the file type of the file at the given path."""
        filename = os.path.basename(filepath)
        if filename == "animals.txt":
            return "animals"
        elif filename.startswith("allSeqAllAn"):
            return "allSeqAllAn"
        
    def getColumnNames(self, filepath, df):
        """Returns the column names of the file at the given path."""
        if self.fileType(filepath) == "animals":
            return ["Animal Number", "Cohort Number", "Animal Key", "Genotype", "Sex", "Lesion", "Implant"]
        elif self.fileType(filepath) == "allSeqAllAn":
            return ["Animal Number", "Trial Number", "Sequence Number"]
        else: return [str(i) for i in range(len(df.columns))]

    def fileTreeItemTriggered(self, item):
        """Called when a file is double clicked in the file tree. Opens the file in the table view."""
        filepath = item.data(0, Qt.ItemDataRole.UserRole)
        if filepath not in self.open_files:
            self.openFile(filepath)
        else:
            # Switch to the tab whose filepath matches the selected file
            for i in range(self.fileTabs.count()):
                if self.fileTabs.widget(i).property("filepath") == filepath:
                    self.fileTabs.setCurrentIndex(i)
                    self.pd_table = self.fileTabs.currentWidget()
                    break
        self.actionsBox.setEnabled(len(self.open_files) > 0)

    def openFile(self, filepath):
        """Opens the file at the given path and displays it in the table view."""
        if filepath.endswith(".txt") or filepath.endswith(".csv"):
            # Check if the file is tabular
            try:
                df = pd.read_csv(filepath, header=None)
            except:
                self.showError("Opening file", "File is not tabular in nature.")
                return
            #TODO: Get rid of self.df
            df.columns = self.getColumnNames(filepath, df)
            self.pd_table = PandasTable(df, self)
            # New tab
            self.open_files.add(filepath)
            self.fileTabs.addTab(self.pd_table, os.path.basename(filepath))
            self.pd_table.setProperty("filepath", filepath)
            self.fileTabs.setCurrentWidget(self.pd_table)

    def currentFile(self):
        """Returns the filepath of the currently open file."""
        if len(self.open_files) == 0:
            return None
        return self.fileTabs.currentWidget().property("filepath")

    def closeFile(self, index):
        """Closes the file at the given index."""
        print(self.open_files)
        self.open_files.remove(self.fileTabs.widget(index).property("filepath"))
        self.fileTabs.removeTab(index)
        self.actionsBox.setEnabled(len(self.open_files) > 0)

    def addStage(self, col=""):
        # Spawn a new PipelineDialog
        dialog = PipelineDialog(col, self)
        dialog.show()

    def appendStageToList(self, stage):
        idx = self.filterTable.rowCount()
        self.filterTable.insertRow(idx)
        self.filterTable.setItem(idx, 0, QTableWidgetItem(str(stage)))
        self.filterTable.item(idx, 0).setData(Qt.ItemDataRole.UserRole, stage)

        

    def applyStageNow(self, stage_tuple):
        query, message = PipelineDialog.parseStage(stage_tuple, self.getColumnNames(self.currentFile()), self)
        if query is None:
            return
        self.pd_table.filterTable(query)
        self.functionTerminal.appendPlainText(message)

    def applyFilters(self):
        if self.currentFile() is None:
            self.showError("Applying filters", "No file selected.")
            return
        queries_and_messages = []
        for i in range(self.filterTable.rowCount()):
            query, message = PipelineDialog.parseStage(self.filterTable.item(i, 0).data(Qt.ItemDataRole.UserRole), self.getColumnNames(self.currentFile()), self)
            if query is None:
                return
            queries_and_messages.append(query)
        # Apply the filters
        for query, message in queries_and_messages:
            self.pd_table.filterTable(query)
            self.functionTerminal.appendPlainText(message)


    def setDefaultSizes(self):
        # Set the default sizes of the splitter
        self.mainSplitter.setSizes([300, 800, 200])

    def showError(self, action, msg):
        """Spawn an error dialog"""
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Icon.Warning)
        error_dialog.setText(msg)
        error_dialog.setWindowTitle(f"Error during {action}")
        error_dialog.exec()


    
                
        



