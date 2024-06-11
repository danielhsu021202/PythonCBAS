from PyQt6.QtWidgets import QApplication, QVBoxLayout, QTextEdit, QListWidgetItem, QFileDialog, QTableWidget, QLabel, QTableWidgetItem, QMessageBox, QDialog, QMenu, QPushButton, QLineEdit, QLabel
from PyQt6.QtCore import Qt

import os
import sys
import shutil
import numpy as np

from utils import ListUtils, FileUtils, StringUtils
from settings import Settings, DataSet

from ui.ImportDataDialog import Ui_ImportDataDialog

from files import CBASFile

class ColumnSelectTable(QTableWidget):
    def __init__(self, file, delimiter, mode, parent=None):
        super().__init__(parent)
        self.editable = False

        self.file = file
        self.mat = FileUtils.getMatrix(file, delimiter=delimiter, limit_rows=100, dtype=str)

        # Disable editing
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        

        # Populate the table with the matrix, with column headers as indices
        self.setColumnCount(self.mat.shape[1])
        self.setRowCount(self.mat.shape[0])
        self.setHorizontalHeaderLabels([str(i) for i in range(self.mat.shape[1])])
        self.setVerticalHeaderLabels([str(i) for i in range(self.mat.shape[0])])

        self.columns = {i: None for i in range(self.columnCount())}

        for i in range(self.mat.shape[0]):
            for j in range(self.mat.shape[1]):
                self.setItem(i, j, QTableWidgetItem(str(self.mat[i, j])))

        # Context Menu
        self.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        if mode == 'info':
            self.horizontalHeader().customContextMenuRequested.connect(self.onColumnRightClickedInfo)
        elif mode == 'data':
            self.horizontalHeader().customContextMenuRequested.connect(self.onColumnRightClickedData)



    
    def onColumnRightClickedInfo(self, pos):
        """Display a context menu when a column is right clicked. For labeling subject info files."""
        menu = QMenu(self)
        labelAction = menu.addAction("Label Column")
        setCovariate = menu.addAction("Set Covariate")
        menu.addSeparator()
        resetAction = menu.addAction("Reset Column")

        action = menu.exec(self.mapToGlobal(pos))
        column_idx = self.columnAt(pos.x())
        if action == labelAction:
            self.labelColumn(column_idx)
        elif action == setCovariate:
           # If the covariate column already exists, reset that one first
            for idx, label in self.columns.items():
                if label == "Covariate":
                    self.resetColumn(idx)
                    break
            self.setHorizontalHeaderItem(column_idx, QTableWidgetItem("Covariate"))
            self.columns[column_idx] = "Covariate"
        elif action == resetAction:
            self.resetColumn(column_idx)

    
    def onColumnRightClickedData(self, pos):
        """Display a context menu when a column is right clicked. For labeling subject data files."""
        column_labels = self.columns.values()
        column_idx = self.columnAt(pos.x())
        menu = QMenu(self)
        if "Session" not in column_labels:        menu.addAction("Session")
        if "Contingency" not in column_labels:    menu.addAction("Contingency")
        if "Choice" not in column_labels:         menu.addAction("Choice")
        if "Modifier" not in column_labels:       menu.addAction("Modifier")
        menu.addSeparator()
        resetAction = menu.addAction("Reset Column")

        action = menu.exec(self.mapToGlobal(pos))
        if action is None: return
        if action != resetAction:
            self.columns[column_idx] = action.text()
            self.setHorizontalHeaderItem(column_idx, QTableWidgetItem(self.columns[column_idx]))
        else:
            self.resetColumn(column_idx)

    def resetColumn(self, column_idx):
        """Reset the selected column."""
        self.columns[column_idx] = None
        self.setHorizontalHeaderItem(column_idx, QTableWidgetItem(str(column_idx)))


    def labelColumn(self, column_idx):
        """Label the selected column."""
        # Spawn a simple two button dialog
        column_label = self.labelColumnDialog(column_idx)
        if column_label and not column_label.isspace():
            if column_label == "Covariate":
                QMessageBox.warning(self, "Warning", "The label 'Covariate' is reserved for covariates. Please choose another label.")
                return
            if column_label in self.columns.values():
                QMessageBox.warning(self, "Warning", f"A column labled '{column_label}' already exists.")
                return
            self.setHorizontalHeaderItem(column_idx, QTableWidgetItem(column_label))
            self.columns[column_idx] = column_label


    def labelColumnDialog(self, column_idx):
        """Dialog to input a label for the column."""
        # Spawn a simple two button dialog
        dialog = QDialog()
        dialog.setWindowTitle(f"Label Column {column_idx}")
        dialog.resize(200, 110)
        
        layout = QVBoxLayout()

        label = QLabel(f"Label column {column_idx}:")

        line_edit = QLineEdit()
        line_edit.setPlaceholderText("Column Label")

        done = QPushButton("Done")
        done.clicked.connect(lambda: dialog.close())
        
        layout.addWidget(label)
        layout.addWidget(line_edit)
        layout.addWidget(done)
        
        dialog.setLayout(layout)
        dialog.exec()
        return str(line_edit.text())



        

class ImportData(QDialog, Ui_ImportDataDialog):
    def __init__(self, proj_obj):
        super().__init__()
        self.setupUi(self)
        self.proj_obj = proj_obj

        # Instance Variables
        self.source_directory_path = None
        self.dataset_dir = None
        self.an_info_name = None
        self.delimiter = ','  # Force the delimiter to be a comma by default
        self.cohorts = []
        self.hasModifier = None

        # Column Select Tables
        self.anInfoTable = None
        self.anDataTable = None

        # Column Names
        self.anInfoColumns = []
        self.anDataColumns = []

        # File paths
        self.cohort_file = None
        self.all_animal_file = None
        self.all_paths_file = None

        # Next button functionality
        self.nextButton.clicked.connect(self.nextPage)

        # Set back button to only be enabled when the current page is not the first page
        self.backButton.setDisabled(True)
        self.Pages.currentChanged.connect(lambda: self.backButton.setDisabled(self.Pages.currentIndex() == 0))
        self.backButton.clicked.connect(self.prevPage)

        # Initial Hides
        self.importDirectoryProgressBar.hide()
        self.importDirectoryProgressBar.setValue(0)
        self.importingStatusLabel.hide()
        self.importResultLabel.hide()
        self.directoryLabel.hide()
        self.selectDelimiterGroupBox.hide()
        self.destLocationGroupBox.hide()
        
        # Start with the first page!
        self.setDatasetAttributes()

        self.returnValue = None

    def run(self):
        self.exec()
        return self.returnValue

    def prevPage(self):
        """Go back a page."""
        page = self.Pages.currentIndex() - 1
        self.Pages.setCurrentIndex(page)
        self.nextButton.setText("Next" if page < self.Pages.count() - 1 else "Import")
        self.nextButton.setDisabled(False)

    def nextPage(self):
        """Go to the next page."""
        page = self.Pages.currentIndex()
        if page == 0:
            self.validateDatasetAttributes()
        elif page == 1:
            self.validateDirectoryInfo()
        elif page == 2:
            self.validateAnInfo()
        elif page == 3:
            self.validateAnimalInfoColumns()
        elif page == 4:
            self.validateAnimalFileColumns()

        
    def setDatasetAttributes(self):
        """
        PAGE 0
        Set the attributes of the dataset.
        """
        self.Pages.setCurrentIndex(0)
        self.nextButton.setDisabled(True)
        self.datasetNameLineEdit.textChanged.connect(lambda: self.nextButton.setDisabled(self.datasetNameLineEdit.text() == ""))
        self.chooseDestPathButton.clicked.connect(lambda: self.chooseDirectory(self.destPathLineEdit))

    def validateDatasetAttributes(self):
        """
        DATA VALIDATION FUNCTION
        """
        # Check if the name already exists
        names = [item.getName() for item in self.proj_obj.getChildren()]
        if self.datasetNameLineEdit.text() in names:
            QMessageBox.critical(self, "Error", "A dataset with that name already exists.")
            return
        self.importDirectoryPage()
    
    def importDirectoryPage(self):
        """
        PAGE 1
        Import data from a directory.
        """
        self.Pages.setCurrentIndex(1)
        self.nextButton.setDisabled(True)
        self.viewCohortsBox.setDisabled(self.source_directory_path is None)
        self.importDirectoryButton.setDisabled(self.directoryPathLineEdit.text() == "")
        self.directoryPathLineEdit.textChanged.connect(lambda: self.importDirectoryButton.setDisabled(self.directoryPathLineEdit.text() == ""))
        self.importDirectoryButton.clicked.connect(self.importDirectory)
        self.chooseDirectoryButton.clicked.connect(lambda: self.chooseDirectory(self.directoryPathLineEdit))
        self.chooseDirectoryButton.mouseDoubleClickEvent = lambda event: None
        self.nextButton.setDisabled(len(self.cohorts) == 0)


    def chooseDirectory(self, line_edit):
        """
        HELPER FUNCTION
        Open a file dialog and select a directory to import data from.
        """
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", directory=Settings.getDocumentsFolder())
        if directory:
            line_edit.setText(directory)

    def importDirectory(self):
        """
        INTERACTIONS FUNCTION
        Import data from the selected directory.
        """
        self.cohorts = []
        self.animalListWidget.clear()
        
        if not os.path.exists(self.directoryPathLineEdit.text()):
            QMessageBox.critical(self, "Error", "The selected directory does not exist.")
            return
        
        directory = self.directoryPathLineEdit.text()
        self.source_directory_path = directory
        
        # Cap the length at 50 characters
        self.directoryLabel.setText(f"Directory: {directory if len(directory) <= 50 else '...' + directory[-47:]}")
        self.directoryLabel.show()

        # self.importDirectoryProgressBar.show()
        # self.importingStatusLabel.show()
        self.importingStatusLabel.setText(f"Scanning {os.path.basename(directory)}...")

        self.viewCohortsBox.setDisabled(False)

        self.cohortListWidget.clear()
        if self.cohortFoldersCheckBox.isChecked():
            self.cohorts = []
            for item in ListUtils.naturalSort(os.listdir(directory)):
                # Add directory paths to the list
                if os.path.isdir(os.path.join(directory, item)):
                    self.cohorts.append(os.path.join(directory, item))
        else:
            # The list is just the directory name
            self.cohorts = [directory]
        
        # Remove hidden files
        self.cohorts = [item for item in self.cohorts if not os.path.basename(item).startswith('.')]

        for item in self.cohorts:
            cohort_path = os.path.join(directory, item) if self.cohortFoldersCheckBox.isChecked() else directory
            
            list_item = QListWidgetItem(os.path.basename(item))
            list_item.setData(Qt.ItemDataRole.UserRole, cohort_path)
            self.cohortListWidget.addItem(list_item)

        self.cohortListWidget.itemDoubleClicked.connect(self.importAnimals)
        self.importResultLabel.setText(f"Found {len(self.cohorts)} cohorts.")
        self.importResultLabel.show()

        self.animalListWidget.itemDoubleClicked.connect(self.previewAnimalFile)



        # delimiter_chosen = lambda: (self.radio_comma or self.radio_tab or self.radio_space or (self.radio_other and self.otherDelimiterLineEdit.text()))
        language_filled = lambda: self.numChoicesLineEdit.text() and self.numContingenciesLineEdit.text()

        def updateNextButton():
            self.nextButton.setDisabled((len(self.cohorts) == 0) or not language_filled())
            # self.nextButton.setDisabled((len(self.cohorts) == 0) or not delimiter_chosen())
        
        self.numChoicesLineEdit.textChanged.connect(updateNextButton)
        self.numContingenciesLineEdit.textChanged.connect(updateNextButton)
        updateNextButton()
        # self.radio_comma.toggled.connect(updateNextButton)
        # self.radio_tab.toggled.connect(updateNextButton)
        # self.radio_space.toggled.connect(updateNextButton)
        # self.radio_other.toggled.connect(updateNextButton)
        # self.otherDelimiterLineEdit.textChanged.connect(updateNextButton)

    def previewAnimalFile(self):
        """
        Preview the selected animal file by printing its contents in a new window.
        """
        file = self.animalListWidget.currentItem().data(Qt.ItemDataRole.UserRole)
        with open(file, 'r') as f:
            contents = f.read()
        
        # Spawn a dialog with a text field to display the contents
        dialog = QDialog()
        dialog.setWindowTitle(f"Previewing {os.path.basename(file)}")
        dialog.resize(300, 400)

        text_edit = QTextEdit()
        text_edit.setText(contents)
        text_edit.setReadOnly(True)

        # Make it so that if a line is too long, scroll bars appear
        text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        text_edit.setAcceptRichText(False)

        row_label = QLabel(f"Rows: {len(contents.splitlines())}")

        layout = QVBoxLayout()
        
        layout.addWidget(text_edit)
        layout.addWidget(row_label)
        

        dialog.setLayout(layout)

        dialog.exec()


    def importAnimals(self):
        """
        HELPER FUNCTION
        Handle double clicks on the cohort list view.
        """
        # For all the files in the cohort, add them to the animal list view
        path = self.cohortListWidget.currentItem().data(Qt.ItemDataRole.UserRole)
        self.animalListWidget.clear()
        for item in ListUtils.naturalSort(os.listdir(path)):
            animal_path = os.path.join(path, item)
            if os.path.isfile(animal_path) and not os.path.basename(animal_path).startswith('.'):
                list_item = QListWidgetItem(item)
                list_item.setData(Qt.ItemDataRole.UserRole, animal_path)
                self.animalListWidget.addItem(list_item)
        self.animalFilesLabel.setText(f"Animal Files ({len(self.animalListWidget)} files)")

    def validateDirectoryInfo(self):
        """
        DATA VALIDATION FUNCTION
        """
        # Ensure the line edits are integers
        try:
            num_choices = int(self.numChoicesLineEdit.text())
            num_contingencies = int(self.numContingenciesLineEdit.text())
        except:
            QMessageBox.critical(self, "Error", "The number of choices and contingencies must be integers.")
            return
        
        if num_choices <= 0 or num_contingencies < 0:
            QMessageBox.critical(self, "Error", "The number of choices must be greater than 0 and the number of contingencies must be non-negative.")
            return
        
        self.selectAnInfo()


    def selectAnInfo(self):
        """
        PAGE 2
        Select the animal info file file.
        """
        # Set the delimiter depending on the radio button selected
        # if self.radio_comma.isChecked():
        #     self.delimiter = ','
        # elif self.radio_tab.isChecked():
        #     self.delimiter = '\t'
        # elif self.radio_space.isChecked():
        #     self.delimiter = ' '
        # elif self.radio_other.isChecked():
        #     self.delimiter = self.otherDelimiterLineEdit.text()

        self.Pages.setCurrentIndex(2)
        self.anInfoListWidget.clear()
        self.nextButton.setDisabled(True)
        self.anInfoListWidget.itemDoubleClicked.connect(self.validateAnInfo)

        cohort = self.cohorts[0]

        self.sampleCohortLabel.setText(f"Showing cohort: {os.path.basename(cohort)}")

        for item in ListUtils.naturalSort(os.listdir(cohort)):
            file = os.path.join(cohort, item)
            if os.path.isfile(file) and not os.path.basename(file).startswith('.'):
                list_item = QListWidgetItem(item)
                list_item.setData(Qt.ItemDataRole.UserRole, os.path.basename(file))
                self.anInfoListWidget.addItem(list_item)
        self.anInfoListWidget.itemClicked.connect(lambda: self.nextButton.setDisabled(self.anInfoListWidget.currentItem() is None))

    def validateAnInfo(self):
        """
        DATA VALIDATION FUNCTION
        Ensures the following conditions are met:
            - The selected file is present in all cohorts
            - Verify that the number of columns is the same for all cohorts
            - Verify that the number of rows is the same as the number of non-aninfo files
            - Verify that the file is numerical
        """
        self.an_info_name = self.anInfoListWidget.currentItem().data(Qt.ItemDataRole.UserRole)
        
        # Verify that the file is present in all cohorts
        missing_files = []
        for cohort in self.cohorts:
            if not os.path.exists(os.path.join(cohort, self.an_info_name)):
                missing_files.append(os.path.basename(cohort))
        if missing_files:
            # Show a warning
            QMessageBox.warning(self, "Warning", f"The file {self.an_info_name} is missing from the following cohorts:\n{', '.join(missing_files)}")

        target_num_cols = -1
        for cohort in self.cohorts:
            try:
                mat = FileUtils.getMatrix(os.path.join(cohort, self.an_info_name), delimiter=self.delimiter)
            except:
                QMessageBox.critical(self, "Error", f"Unable to parse the file {self.an_info_name}.")
                return

            # Check if the matrix is numerical
            try:
                mat = mat.astype(float)
            except:
                QMessageBox.critical(self, "Error", f"The file {self.an_info_name} is not numerical.")
                return
            
            # Check if the number of columns is the same for all cohorts
            if target_num_cols == -1:
                target_num_cols = mat.shape[1]
            if mat.shape[1] != target_num_cols:
                QMessageBox.critical(self, "Error", "The number of columns in the selected file is not the same for all cohorts.")
                return
            
            # Count the number of non-aninfo files
            num_files = len([name for name in os.listdir(cohort) if os.path.isfile(os.path.join(cohort, name)) and not name.startswith('.') and name != self.an_info_name])
            if num_files != mat.shape[0]:
                QMessageBox.critical(self, "Error", f"""The number of rows in the selected file is not the same as the number of non-aninfo files in {os.path.basename(cohort)}.
                                     \nExpected {num_files}, got {mat.shape[0]}.""")
                return
        
        # If the above code runs without any issues, we can proceed to the next step
        self.labelAnimalInfoColumns()

    def clearContainer(self, container: QVBoxLayout):
        """
        HELPER FUNCTION
        Clear a container of widgets.
        """
        for i in range(container.count()):
            widget = container.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
            
    def labelAnimalInfoColumns(self):
        """
        PAGE 3
        Label the columns of the animal info file.
        """
        self.Pages.setCurrentIndex(3)

        # Get the first animal info file of the first cohort as a sample
        cohort = self.cohorts[0]
        an_info_file = os.path.join(cohort, self.an_info_name)

        self.clearContainer(self.anInfoTableWidgetContainer)
        self.anInfoTable = ColumnSelectTable(an_info_file, self.delimiter, mode='info')
        self.anInfoTableWidgetContainer.addWidget(self.anInfoTable)
        self.sampleAnimalLabel_info.setText(f"Showing sample animal info file: {os.path.basename(cohort)}/{self.an_info_name}")

    def validateAnimalInfoColumns(self):
        """
        DATA VALIDATION FUNCTION
        """
        # Check if the columns have been labeled
        table_vals = self.anInfoTable.columns.values()
        if None in table_vals:
            QMessageBox.warning(self, "Warning", "Not all columns have been labeled. If not labeled, the column will be referred to by its index.")

        if "Covariate" in table_vals:
            QMessageBox.information(self, "Covariate Selected", "A covariate column has been identified. This means this dataset can use the Correlational CBAS.")


        # Append the column names to the list. Use the index if the column is not labeled
        # self.anInfoColumns = {label: i for i, label in self.anInfoTable.columns.items()}

        for i, label in enumerate(self.anInfoTable.columns.values()):
            if label is None:
                self.anInfoColumns.append(f"<column {i}>")
            else:
                self.anInfoColumns.append(label)
        
        # If all the above checks pass, we can proceed to the next step
        self.labelAnimalFileColumns()
    
    def labelAnimalFileColumns(self):
        """
        PAGE 4
        Label the columns of the animal file.
        """
        self.Pages.setCurrentIndex(4)
        self.nextButton.setText("Import")

        # Get the first animal file of the first cohort as a sample
        cohort = self.cohorts[0]
        animal_file = os.path.join(cohort, [name for name in os.listdir(cohort) if os.path.isfile(os.path.join(cohort, name)) and not name.startswith('.') and name != self.an_info_name][0])

        self.clearContainer(self.anDataTableWidgetContainer)
        self.anDataTable = ColumnSelectTable(animal_file, self.delimiter, mode='data')
        self.anDataTableWidgetContainer.addWidget(self.anDataTable)

        self.sampleAnimalLabel_data.setText(f"Showing sample animal file: {os.path.basename(cohort)}/{os.path.basename(animal_file)}")

    def validateAnimalFileColumns(self):
        """
        DATA VALIDATION FUNCTION
        """
        # Check if the columns have been labeled
        table_vals = self.anDataTable.columns.values()
        if "Choice" not in table_vals:
            QMessageBox.critical(self, "Error", "The 'Choice' column is missing and is required.")
            return
        missing = [col for col in ["Session", "Contingency", "Modifier"] if col not in table_vals]
        if missing:
            QMessageBox.warning(self, "Warning", f"The columns {StringUtils.andSeparateList(missing)} are not specified. These columns are optional.\nIf not included, 'Session' and 'Contingency' will be considered 0 for all subjects.")
        
        # Append the column names to the list. Use the index if the column is not labeled
        for label in self.anDataTable.columns.values():
            if label is None:
                self.anDataColumns.append(None)
            else:
                self.anDataColumns.append(label)
        # self.anDataColumns = {label: i for i, label in self.anDataTable.columns.items()}

        self.hasModifier = "Modifier" in self.anDataColumns
        self.processFiles()
        self.createDataset()

    def processFiles(self):
        folder = self.proj_obj.getDir()  # The project directory
        self.dataset_dir = os.path.join(folder, self.datasetNameLineEdit.text())  # The dataset directory
        dest = os.path.join(self.dataset_dir, "Data")  # The data directory
        source = self.source_directory_path  # The source directory
        # Remove the data directory if it exists and recreate it
        if os.path.exists(self.dataset_dir):
            shutil.rmtree(self.dataset_dir)
        os.mkdir(self.dataset_dir)
        os.mkdir(dest)
        

        cohortNumbers = {}  # Dictionary to store the cohort names and their corresponding assigned numbers

        if self.cohortFoldersCheckBox.isChecked():
            cohorts = ListUtils.naturalSort([os.path.join(source, folder) for folder in os.listdir(source) if os.path.isdir(os.path.join(source, folder))])
        else:
            cohorts = [source]
        for i, cohort_folder in enumerate(cohorts):
            cohort_name = os.path.basename(cohort_folder)
            cohortNumbers[cohort_name] = i
            os.mkdir(os.path.join(dest, cohort_name))
            files = [file for file in os.listdir(cohort_folder) if FileUtils.validFile(os.path.join(cohort_folder, file))]
            for file in files:
                self.processFile(os.path.join(cohort_folder, file), os.path.join(dest, cohort_name))

        self.cohort_file = os.path.join(self.dataset_dir, "cohorts.txt")
        self.writeCohortFile(self.cohort_file, cohortNumbers)
        self.processAnimalMetadata(cohortNumbers, dest)
                    
                
    def processFile(self, filepath, dest, qlabel: QLabel=None):
        if qlabel is not None: qlabel.setText(f"Processing {os.path.basename(filepath)}")
        if os.path.basename(filepath) == self.an_info_name:
            # Copy the file over
            shutil.copy(filepath, dest)
            return
        possibleColumns = Settings.getColumnOrder()
        columnMap = {}
        orig_mat = FileUtils.getMatrix(filepath, delimiter=self.delimiter)
        new_mat = np.zeros((orig_mat.shape[0], 0))
        for i, column in enumerate(possibleColumns):
            if column in self.anDataColumns:
                idx = self.anDataColumns.index(column)
                # Copy that column over
                new_mat = np.hstack((new_mat, orig_mat[:, idx].reshape(-1, 1)))
            else:
                if column == "Modifier": continue  # Modifier is not required
                new_mat = np.hstack((new_mat, np.zeros((orig_mat.shape[0], 1))))
            columnMap[column] = i
        # Write the new matrix to a file
        filename = os.path.basename(filepath)
        new_file = os.path.join(dest, filename)
        FileUtils.writeMatrix(new_file, new_mat, delimiter=self.delimiter)

    def processAnimalMetadata(self, cohort_dict: dict, data_dir):
        animal_num = 0

        all_paths = []
        # self.animal_file = os.path.join(self.dataset_dir, "animals.txt")
        animal_matrix = []
        for cohort_name, cohort_num in cohort_dict.items():
            cohort_folder = os.path.join(data_dir, cohort_name)
            info_file = os.path.join(cohort_folder, self.an_info_name)
            animal_files = ListUtils.naturalSort([name for name in os.listdir(cohort_folder) if os.path.isfile(os.path.join(cohort_folder, name)) 
                                                        and name != self.an_info_name
                                                        and not name.startswith('.')])
            all_paths += [os.path.join(cohort_folder, file) for file in animal_files]
            animal_info_matrix = FileUtils.getMatrix(info_file, delimiter=self.delimiter, dtype=(
                float if "Covariate" in self.anInfoColumns else int
            ))
            # Get rid of hidden files
            animal_files = [file for file in animal_files if not file.startswith('.')]
            assert len(animal_files) == len(animal_info_matrix)
            for i, animal_file in enumerate(animal_files):
                animal_info = animal_info_matrix[i]
                row = [animal_num, cohort_num]
                row.extend(animal_info)
                animal_matrix.append(row)
                animal_num += 1
        
        animal_matrix_file = CBASFile("all_animals", np.array(animal_matrix), col_headers=["Animal No.", "Cohort No."] + self.anInfoColumns)
        animal_matrix_file.saveFile(self.dataset_dir)
        self.all_animals_file_path = os.path.join(self.dataset_dir, "all_animals.cbas")

        self.all_paths_file = os.path.join(self.dataset_dir, "all_paths.pkl")
        FileUtils.pickleObj(all_paths, self.all_paths_file)
        self.num_animals = len(all_paths)

        
    def createDataset(self):
        anDataColumns = Settings.buildDataColumnsDict(0, 1, 2, 3 if self.hasModifier else None)
        dataset = DataSet()
        dataset.createDataset(self.datasetNameLineEdit.text(),
                              self.datasetDescriptionTextEdit.toPlainText(),
                              self.dataset_dir,
                              self.an_info_name,
                              self.anInfoColumns,
                              anDataColumns,
                              "Covariate" in self.anInfoColumns,
                              int(self.numChoicesLineEdit.text()),
                              self.hasModifier,
                              int(self.numContingenciesLineEdit.text()),
                              self.num_animals,
                              self.cohort_file_path,
                              self.all_animals_file_path,
                              self.all_paths_file,       
        )
        dataset.setParent(self.proj_obj)
        self.returnValue = dataset
        self.close()

    def writeCohortFile(self, file, cohort_dict: dict):
        cohort_matrix = []
        for cohort, i in cohort_dict.items():
            cohort_matrix.append([cohort, i])
        cohort_matrix = np.array(cohort_matrix)
        cohort_file = CBASFile("cohorts", cohort_matrix, col_headers=["Cohort", "Cohort No."])
        cohort_file.saveFile(self.dataset_dir)
        self.cohort_file_path = os.path.join(self.dataset_dir, "cohorts.cbas")
        



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImportData()
    window.show()
    sys.exit(app.exec())
