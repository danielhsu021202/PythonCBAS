
from PyQt6.QtWidgets import QTableView, QWidget, QFileDialog, QTreeWidgetItem, QMenu
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QAbstractTableModel, Qt

import os
import sys
import re
import pandas as pd

from ui.FileViewerWidget import Ui_FileViewer
import os
from PyQt6.QtWidgets import QMenu
from PyQt6.QtCore import Qt

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

        self.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.horizontalHeader().customContextMenuRequested.connect(self.onHeaderContextMenuRequested)
        self.verticalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.verticalHeader().customContextMenuRequested.connect(self.onHeaderContextMenuRequested)

    def onHeaderContextMenuRequested(self, pos):
        menu = QMenu()
        sortAscendingAction = menu.addAction("Sort Ascending",)
        sortDescendingAction = menu.addAction("Sort Descending")
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

    
    def sortColumn(self, column, order):
        self.df = self.df.sort_values(by=self.df.columns[column], axis=0, ascending=order)
        self.updateTable()

    def sortRow(self, row, order):
        self.df = self.df.sort_values(by=self.df.index[row], axis=1, ascending=order)
        self.updateTable()

        

            




    def updateTable(self):
        self.model = PandasTableModel(self.df)
        self.setModel(self.model)



        

class FileViewer(QWidget, Ui_FileViewer):
    def __init__(self, parent=None):
        super(FileViewer, self).__init__(parent)
        self.setupUi(self)
        self.setDefaultSizes()

        self.directories = set()
        self.directories.add(os.path.join("output"))
        self.refreshFileTree()
        self.df = None

        self.mapButtonActions()
        self.fileTree.itemDoubleClicked.connect(self.openFile)

    def mapButtonActions(self):
        self.fileTreeImportButton.clicked.connect(self.importDirectory)
        self.refreshFileTreeButton.clicked.connect(self.refreshFileTree)
        self.countRowsButton.clicked.connect(self.countRows)
        self.countColumnsButton.clicked.connect(self.countColumns)
        self.clearFunctionTerminalButton.clicked.connect(self.functionTerminal.clear)

    def displayHeaderContextMenu(self, pos):
        """Displays the context menu for the header.
            Axis 0 is row, 1 is column."""
        print("Requested")
        menu = QMenu()
        menu.addAction("Sort Ascending",)
        menu.addAction("Sort Descending")
        menu.exec(self.dataTable.mapToGlobal(pos))
        

    def importDirectory(self):
        """Opens a file dialog to import a directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.directories.add(directory)
            self.refreshFileTree()

    def refreshFileTree(self):
        """Refreshes the file tree view with the current directories."""
        self.fileTree.clear()
        for directory in self.directories:
            if os.path.isdir(directory):
                self.populateFileTree(directory, self.fileTree.invisibleRootItem())
            else:
                pass

    def countRows(self):
        """Displays row count in the function terminal."""
        self.functionTerminal.appendPlainText("Rows: " + str(len(self.df)) if self.df is not None else "No file selected.")

    def countColumns(self):
        """Displays column count in the function terminal."""
        self.functionTerminal.appendPlainText("Columns: " + str(len(self.df.columns)) if self.df is not None else "No file selected.")

    def naturalSort(self, l):
        """Sorts a list of strings in natural order."""
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)
        
    def populateFileTree(self, base_path, parent_item):
        """
        Recursively populates a QTreeView with the hierarchical structure of files and directories.
        """
        # Create a QTreeWidgetItem for the root directory
        root_item = QTreeWidgetItem(parent_item)
        root_item.setText(0, os.path.basename(base_path))  # Set the text for the root item
        root_item.setIcon(0, QIcon("icons/folder.png"))

        for item in self.naturalSort(os.listdir(base_path)):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                # Recursively populate the tree view with the contents of the directory
                self.populateFileTree(item_path, root_item)
            elif os.path.isfile(item_path):
                # Create a QTreeWidgetItem for files
                file_item = QTreeWidgetItem(root_item)
                file_item.setText(0, item)  # Set the text for the item
                file_item.setData(0, Qt.ItemDataRole.UserRole, item_path)  # Set the data for the item

    def openFile(self, item):
        """Opens the file and displays it in the table view, replacing the current widget."""
        if item.childCount() == 0:
            self.df = pd.read_csv(item.data(0, Qt.ItemDataRole.UserRole), header=None)
            self.df.columns = [str(i) for i in range(len(self.df.columns))]
            pd_table = PandasTable(self.df, self)
            self.dataTableLayout.replaceWidget(self.dataTableLayout.itemAt(0).widget(), pd_table)
            self.fileNameLabel.setText(os.path.basename(item.data(0, Qt.ItemDataRole.UserRole)))


    def setDefaultSizes(self):
        # Set the default sizes of the splitter
        self.mainSplitter.setSizes([300, 800, 200])

    
                
        



