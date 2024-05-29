from ui.UI_SettingsDialog import Ui_SettingsDialog
from ui.GroupSelectorDialog import Ui_GroupSelectorDialog

from settings import Settings, Project, DataSet, Counts, Resamples

from sequences import SequencesProcessor
from resampler import Resampler
from statistical_analyzer import StatisticalAnalyzer
from visualizations import Visualizer

from ui.ProgressDialog import Ui_ProgressDialog

from utils import TimeUtils, ReturnContainer
from files import CBASFile

from PyQt6.QtWidgets import QApplication, QDialog, QWidget, QMessageBox, QLabel, QProgressBar, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, QMenu, QListWidgetItem, QTableWidgetItem
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRegularExpression
from PyQt6 import QtGui


import time
import numpy as np
import pandas as pd

class ProgressDialog(QDialog, Ui_ProgressDialog):
    """
    Spawns a QDialog with a that is a UI displaying the progress for a running process, expressed as a QThread object.
   
    Worker thread and the running function as well as metadata for the process is passed in to describe what process should be run.
    
    Each worker is required to emit a start, running, and end signal to indicate the start, progress, and end of the process:
        start_signal: pyqtSignal()
        running_signal: pyqtSignal(tuple), tuple is (value, label) pair
        end_signal: pyqtSignal(object), object is the return value of the process, which is specified by the retrieval function
            and gotten automatically.

    The end signal receives the return value by calling the retrieval function, which is used to retrieve the return value from the worker thread.
    This is specified by the worker to describe what the return value is.
    """
    def __init__(self, title: str, progress_max: int, 
                 worker: QThread, worker_function, retrieval_function, 
                 start_signal: pyqtSignal, running_signal: pyqtSignal, end_signal: pyqtSignal, 
                 parent=None):
        super(ProgressDialog, self).__init__(parent)
        self.setupUi(self)

        self.progress_max = progress_max
        self.returnValue = None
        self.worker = worker
        self.display_percent = False
        
        # UI Setup
        self.resize(300, 200)

        self.setWindowTitle(title)

        self.cancelButton.clicked.connect(self.cancel)

        self.timeElapsed.setText("Time elapsed: 0.00 seconds")

        self.running_thread = QThread()
        self.worker.moveToThread(self.running_thread)
        self.running_thread.started.connect(worker_function)
        end_signal.connect(self.running_thread.quit)
        self.running_thread.finished.connect(self.running_thread.deleteLater)
        start_signal.connect(self.start)
        running_signal.connect(self.updateProgress)

        self.running_thread.finished.connect(lambda: self.end(retrieval_function()))

        # Timer setup
        self.timer = QTimer()
        self.startTime = None
        self.timer.timeout.connect(self.updateTimeElapsed)
        self.timer.start(1000)  # Update every second

    def displayPercentage(self):
        """Switch the progress bar to display percentage instead of value."""
        self.display_percent = True

    def displayProportion(self):
        """Switch the progress bar to display the proportion instead of percentage."""
        self.display_percent = False

    def start(self):
        """
        Called when the worker thread emits a start signal.
        Reset the progress bar and start the timer.
        """
        self.progressBar.setValue(0)
        self.startTime = time.time()  # Record the start time

    def updateProgress(self, progress: tuple):
        """
        Called when the worker thread emits a running signal.
        Update the progress bar with the given value and label.
        """
        value, progress_label = progress
        percentage = value / self.progress_max * 100
        self.progressBar.setValue(int(percentage))
        if self.display_percent:
            self.progressLabel.setText(f"{percentage:.2f}%")
        else:
            self.progressBar.setFormat(f"{value} / {self.progress_max}")
        self.progressLabel.setText(progress_label)

    def updateTimeElapsed(self):
        if self.startTime is not None:
            elapsed_time = time.time() - self.startTime
            self.timeElapsed.setText(f"Time elapsed: {TimeUtils.format_time(elapsed_time)}")

    def end(self, return_value):
        """
        Called when the worker thread emits an end signal.
        Update the progress bar to 100% and close the dialog.
        """
        self.timer.stop()  # Stop the timer when the operation is finished
        self.returnValue = return_value
        self.close()

    def run(self):
        """Run the dialog and return the return value from the worker thread."""
        self.running_thread.start()
        self.exec()
        return self.returnValue
    
    def cancel(self):
        """Cancel the running process."""
        try:
            self.running_thread.quit()
            self.running_thread.wait()
        except Exception as e:
            pass
        self.close()
        self.returnValue = None
        return self.returnValue


class GroupSelectorDialog(QDialog, Ui_GroupSelectorDialog):
    def __init__(self, aninfocolumns: list, all_animals_matrix_path, parent=None):
        super(GroupSelectorDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Groups Selector")

        self.returnValue = None

        self.aninfocolumns = aninfocolumns
        self.all_animals_matrix_path = all_animals_matrix_path

        self.group1available = self.aninfocolumns
        self.group2available = self.aninfocolumns

        self.addGroup1Button.clicked.connect(lambda: self.add(1))
        self.addGroup2Button.clicked.connect(lambda: self.add(2))

        self.assignButton.clicked.connect(self.assign)

        # Right click to spawn qmenu at mouse position
        self.group1ListWidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.group1ListWidget.customContextMenuRequested.connect(lambda pos: self.spawnMenu(self.group1ListWidget.mapToGlobal(pos), 1))
        self.group2ListWidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.group2ListWidget.customContextMenuRequested.connect(lambda pos: self.spawnMenu(self.group2ListWidget.mapToGlobal(pos), 2))


    def add(self, group: int):
        # Create a horizontal layout with a dropdown, a line edit, and a button
        widget = QWidget()
        layout = QHBoxLayout()
        dropdown = QComboBox()
        dropdown.addItems(self.group1available if group == 1 else self.group2available)
        lineEdit = QLineEdit(placeholderText="(integer)")
        lineEdit.setValidator(QtGui.QIntValidator())
        lineEdit.setMaximumWidth(100)

        layout.addWidget(dropdown)
        layout.addWidget(lineEdit)
        layout.setContentsMargins(0, 0, 0, 0)

        widget.setLayout(layout)
        widget.setContentsMargins(6, 3, 6, 3)
        

        item = QListWidgetItem()
        item.setSizeHint(widget.sizeHint())

        if group == 1: 
            self.group1ListWidget.addItem(item)
            self.group1ListWidget.setItemWidget(item, widget)
        elif group == 2: 
            self.group2ListWidget.addItem(item)
            self.group2ListWidget.setItemWidget(item, widget)

    def spawnMenu(self, pos, group: int):
        menu = QMenu()
        delete_action = menu.addAction("Delete")
        action = menu.exec(pos)
        if action == delete_action:
            self.delete(group)

    def delete(self, group: int):
        if group == 1:
            for item in self.group1ListWidget.selectedItems():
                self.group1ListWidget.takeItem(self.group1ListWidget.row(item))
        elif group == 2:
            for item in self.group2ListWidget.selectedItems():
                self.group2ListWidget.takeItem(self.group2ListWidget.row(item))

    def assign(self):
        # Generate dictionaries mapping from the content of the combo box to the line edit
        group1 = {}
        group2 = {}
        for i in range(self.group1ListWidget.count()):
            item = self.group1ListWidget.item(i)
            widget = self.group1ListWidget.itemWidget(item)
            key = widget.layout().itemAt(0).widget().currentText()
            value = widget.layout().itemAt(1).widget().text()
            if value == "":
                QMessageBox.critical(self, "Input Error", f"Group value for {key} in group 1 must be filled.")
                return
            group1[key] = int(value)
        for i in range(self.group2ListWidget.count()):
            item = self.group2ListWidget.item(i)
            widget = self.group2ListWidget.itemWidget(item)
            key = widget.layout().itemAt(0).widget().currentText()
            value = widget.layout().itemAt(1).widget().text()
            if value == "":
                QMessageBox.critical(self, "Input Error", f"Attribute value for {key} in group 2 must be filled.")
                return
            group2[key] = int(value)
        filters = [group1, group2]
        print(filters)
        orig_groups, all_animals = Resampler.assignGroups(self.all_animals_matrix_path, self.aninfocolumns, filters)
        self.returnValue = (filters, orig_groups, all_animals)
        if len(all_animals) != len(set(all_animals)):
            overlap = set(orig_groups[0]) & set(orig_groups[1])
            QMessageBox.critical(self, "Input Error", f"""Groups must be mutually exclusive.\n
                                                        Overlapping Subjects ({len(overlap)}): {overlap}""")
            return
        self.close()

    def run(self):
        self.exec()
        return self.returnValue



class SettingsDialog(QDialog, Ui_SettingsDialog):
    def __init__(self, setting_type: str, proj_obj: Project, parent_obj=None, parent=None):
        super(SettingsDialog, self).__init__()
        self.setupUi(self)
        self.parent = parent
        self.type = setting_type

        self.proj_obj = proj_obj
        self.parent_obj = parent_obj

        if self.type == "counts":
            assert type(self.parent_obj) == DataSet
            self.setupCountsSettings()
        elif self.type == "resamples":
            assert type(self.parent_obj) == Counts
            self.setupResampleSettings()

        self.scanned_criterion_matrix = None
            
        self.returnValue = None


    ### COUNTS ###

    def setupCountsSettings(self):
        """Set up the settings for Counts."""
        self.SettingsPages.setCurrentIndex(0)
        self.setWindowTitle("Counts Settings")

        self.criterionOrderLineEdit.textChanged.connect(lambda: self.scanButton.setDisabled(self.criterionOrderLineEdit.text() == ""))

        self.scanButton.clicked.connect(self.scanCriterionOrder)
        self.visualizeButton.clicked.connect(self.visualizeCriterionOrder)
        self.createButton.clicked.connect(self.runCounts)

        self.criterionOrderLineEdit.setValidator(QtGui.QIntValidator(0, 2**31-1))
        self.maxSeqLenLineEdit.setValidator(QtGui.QIntValidator(1, 2**31-1))
        # Regular Expression to take either 'inf' or a nonnegative number
        pattern = r'^(inf|\d+(\.\d*)?)$'
        regExp = QRegularExpression(pattern)
        self.criterionNumberLineEdit.setValidator(QtGui.QRegularExpressionValidator(regExp))

        if not self.parent_obj.getHasModifier():
            self.criterionOrderLineEdit.setText(str(0))
            self.criterionOrderLineEdit.setDisabled(True)
            self.criterionOrderLineEdit.setToolTip("This dataset has no modifier, so only criterion order 0 can be used.")

    def scanCriterionOrder(self):
        """
        Scan the criterion order.
        """
        assert type(self.parent_obj) == DataSet

        order = int(self.criterionOrderLineEdit.text())

        inf_criterion = Settings.buildCriterionDict(order, float('inf'), self.criterionIncludeFailedCheck.isChecked(), self.criterionAllowRedemptionCheck.isChecked())
        counts_language = Settings.buildCountsLanguageDict(order, bool(self.straddleSessionsCheck.isChecked()))  # Can use order as the max sequence length

        processor = SequencesProcessor(None,  # Name is none because we're not saving this, just scanning.
                                        self.parent_obj.getDir(), 
                                       self.parent_obj.getAnDataColumns(), 
                                       self.parent_obj.getLanguage(), 
                                       inf_criterion, 
                                       counts_language, 
                                       self.parent_obj.getNumAnimals())
        self.scanned_criterion_matrix = ProgressDialog(f"Scanning Criterion Order {order}", self.parent_obj.getNumAnimals(),
                                          processor, processor.scanCriterionOrder, processor.getCriterionMatrix,
                                          processor.start_processing_signal, processor.processing_progress_signal, processor.scan_complete_signal, self).run()
        
        self.visualizeButton.setDisabled(self.scanned_criterion_matrix is None)

    def visualizeCriterionOrder(self):
        # Generate a scatter plot of the criterion matrix and open a dialog to display it
        
        criterion_matrix = pd.DataFrame(self.scanned_criterion_matrix).fillna(0)

        visualizer = Visualizer("Criterion", parent=self)
        visualizer.createScatterPlot(criterion_matrix, "Maximum # of Perfect Performance Sequences per Subject", "Subject #", "Number of Perfect Performance Sequences")
        visualizer.showWindow()
    


    def runCounts(self):
        assert type(self.parent_obj) == DataSet

        # Field Validation
        
        
        if self.nameLineEdit.text() == "":
            QMessageBox.critical(self, "Input Error", "Name not filled.")
            return
        else:
            # Prevent duplicate names
            names = [item.getName() for item in self.parent_obj.getChildren()]
            if self.nameLineEdit.text() in names:
                QMessageBox.critical(self, "Error", "A Counts with that name already exists.")
                return
        if (self.criterionOrderLineEdit.text() == ""):
            QMessageBox.critical(self, "Input Error", "Criterion order not filled.")
            return
        if (self.maxSeqLenLineEdit.text() == ""):
            QMessageBox.critical(self, "Input Error", "Max sequence length not filled.")
            return
        number_text = self.criterionNumberLineEdit.text()
        if (number_text == ""):
            QMessageBox.critical(self, "Input Error", "Criterion number not filled.")
            return
        elif (number_text != "inf") and (not number_text.isnumeric()):
            QMessageBox.critical(self, "Input Error", "Criterion number must be numeric or 'inf'.")
            return

        # Create the Counts object
        number = float('inf') if self.criterionNumberLineEdit.text() == "inf" else int(self.criterionNumberLineEdit.text())
        order = int(self.criterionOrderLineEdit.text())
        max_seq_len = int(self.maxSeqLenLineEdit.text())

        if order > max_seq_len:
            QMessageBox.critical(self, "Input Error", "Criterion order cannot exceed max sequence length.")
            return

        criterion = Settings.buildCriterionDict(order, number, self.criterionIncludeFailedCheck.isChecked(), self.criterionAllowRedemptionCheck.isChecked())
        counts_language = Settings.buildCountsLanguageDict(max_seq_len, bool(self.straddleSessionsCheck.isChecked()))

        processor = SequencesProcessor(self.nameLineEdit.text(),
                                        self.parent_obj.getDir(),
                                        self.parent_obj.getAnDataColumns(),
                                        self.parent_obj.getLanguage(),
                                        criterion,
                                        counts_language,
                                        self.parent_obj.getNumAnimals())
        
        total = self.parent_obj.getNumAnimals() + self.parent_obj.getNumContingencies() * max_seq_len
        progress_dialog = ProgressDialog(f"Calculating Sequence Counts", total,
                                            processor, processor.runSequenceCounts, processor.getCountsDir,
                                            processor.start_processing_signal, processor.processing_progress_signal, processor.seq_cnts_complete_signal, self)
        progress_dialog.displayPercentage()
        counts_dir = progress_dialog.run()

        self.createCounts(counts_dir, criterion, counts_language)
        
        
    def createCounts(self, counts_dir, criterion, counts_language):
        """Create the Counts object and return it."""
        counts = Counts()
        counts.createCounts(self.nameLineEdit.text(),
                            counts_dir,
                            self.descPlainTextEdit.toPlainText(),
                            criterion,
                            counts_language)
        counts.setParent(self.parent_obj)
        self.returnValue = counts
        self.close()


    
    ### RESAMPLES ###

    def setupResampleSettings(self):
        """Set up the settings for Resamples."""
        self.SettingsPages.setCurrentIndex(1)
        self.setWindowTitle("Resample Settings")

        self.group1TableWidget.setColumnWidth(0, 200)
        self.group2TableWidget.setColumnWidth(0, 200)

        # Validators
        self.customSeedLineEdit.setValidator(QtGui.QIntValidator())
        self.numResamplesLineEdit.setValidator(QtGui.QIntValidator(1, 2**31-1))
        # Takes comma-separated nonnegative integers
        pattern = r'^(\d+,)*\d+$'
        regExp = QRegularExpression(pattern)
        self.contingenciesLineEdit.setValidator(QtGui.QRegularExpressionValidator(regExp))
        self.alphaLineEdit.setValidator(QtGui.QDoubleValidator(0, 1.0, 2))
        self.gammaLineEdit.setValidator(QtGui.QDoubleValidator(0, 1.0, 2))

        # Intial States
        self.customSeedLineEdit.setDisabled(True)

        self.selectGroupsButton.clicked.connect(self.selectGroups)
        self.useAllContingenciesCheck.stateChanged.connect(self.useAllContingenciesToggled)

        if not self.parent_obj.getParent().correlationalPossible():
            self.useCorrelationalCheckBox.setDisabled(True)
            self.useCorrelationalCheckBox.setToolTip("Correlational resamples are not possible for this dataset.")

        # Actions
        self.createButton.clicked.connect(self.runResamples)
        self.fdpRadio.toggled.connect(self.errorCorrectionSettingToggled)
        self.fwerRadio.toggled.connect(self.errorCorrectionSettingToggled)

    def useAllContingenciesToggled(self):
        if self.useAllContingenciesCheck.isChecked():
            self.contingenciesLineEdit.setDisabled(True)
            all_conts_str = ",".join([str(i) for i in range(self.parent_obj.getParent().getNumContingencies())])
            self.contingenciesLineEdit.setText(all_conts_str)
        else:
            self.contingenciesLineEdit.setDisabled(False)

    def errorCorrectionSettingToggled(self):
        if self.fdpRadio.isChecked():
            self.alphaLineEdit.setText("0.5")
            self.gammaLineEdit.setText("0.05")
        else:
            self.alphaLineEdit.setText("0.05")
            

    def selectGroups(self):
        assert type(self.parent_obj) == Counts
        dataset = self.parent_obj.getParent()
        assert type(dataset) == DataSet
        group_selector = GroupSelectorDialog(dataset.getAnInfoColumnNames(), dataset.getAllAnimalsFile(), parent=self)
        filters, orig_groups, all_animals = group_selector.run()
        # Groups must be mutually exclusive
        
        self.group1TableWidget.setRowCount(0)
        self.group2TableWidget.setRowCount(0)
        for attribute, value in filters[0].items():
            self.group1TableWidget.insertRow(self.group1TableWidget.rowCount())
            row = self.group1TableWidget.rowCount() - 1
            self.group1TableWidget.setItem(row, 0, QTableWidgetItem(str(attribute))) 
            self.group1TableWidget.setItem(row, 1, QTableWidgetItem(str(value)))
        for attribute, value in filters[1].items():
            self.group2TableWidget.insertRow(self.group2TableWidget.rowCount())
            row = self.group2TableWidget.rowCount() - 1
            self.group2TableWidget.setItem(row, 0, QTableWidgetItem(str(attribute))) 
            self.group2TableWidget.setItem(row, 1, QTableWidgetItem(str(value)))
        self.group1CountLabel.setText(f"{len(orig_groups[0])} Animals") 
        self.group2CountLabel.setText(f"{len(orig_groups[1])} Animals")
        self.totalAnimalsLabel.setText(f"Total: {len(all_animals)} Animals")

        self.orig_groups = orig_groups
        self.all_animals = all_animals      

    def runResamples(self):
        assert type(self.parent_obj) == Counts

        # Field Validation
        if self.nameLineEdit.text() == "":
            QMessageBox.critical(self, "Input Error", "Name not filled.")
            return
        if self.numResamplesLineEdit.text() == "":
            QMessageBox.critical(self, "Input Error", "Number of resamples not filled.")
            return
        if self.contingenciesLineEdit.text() == "":
            QMessageBox.critical(self, "Input Error", "Contingencies not filled.")
            return
        if self.alphaLineEdit.text() == "":
            QMessageBox.critical(self, "Input Error", "Alpha not filled.")
            return
        if float(self.alphaLineEdit.text()) > 1.0:
            QMessageBox.critical(self, "Input Error", "Alpha must be between 0 and 1.")
            return 
        if self.gammaLineEdit.text() == "":
            QMessageBox.critical(self, "Input Error", "Gamma not filled.")
            return
        if float(self.gammaLineEdit.text()) > 1.0:
            QMessageBox.critical(self, "Input Error", "Gamma must be between 0 and 1.")
            return

        # Create the Resamples object
        conts = [int(cont) for cont in self.contingenciesLineEdit.text().split(",")]
        num_resamples = int(self.numResamplesLineEdit.text())
        contingencies = [int(cont) for cont in self.contingenciesLineEdit.text().split(",")]
        alpha = float(self.alphaLineEdit.text())
        gamma = float(self.gammaLineEdit.text())
        seed = int(self.customSeedLineEdit.text()) if self.customSeedLineEdit.text() != "" else None

        resampler = Resampler(self.nameLineEdit.text(), self.parent_obj.getDir(), 
                              self.parent_obj.getMaxSequenceLength(), conts,
                              custom_seed=seed)
        
        running_process = None
        
        if self.useCorrelationalCheckBox.isChecked():
            animal_matrix_file = self.parent_obj.getParent().getAllAnimalsFile()
            # animal_matrix = np.loadtxt(animal_matrix_file, delimiter=",", dtype=float)
            animal_matrix = CBASFile.loadFile(animal_matrix_file).getData()
            covariate_col = self.parent_obj.getParent().getAnInfoColumnNames().index("Covariate") + 2  # +2 because of the animal number and sequence number columns
            resampler.setCovariates(animal_matrix[:, covariate_col].astype(float))
            running_process = lambda: resampler.generateResampledMatrix(correlational=True, num_resamples=num_resamples)
        else:
            if len(self.orig_groups[0]) == 0 or len(self.orig_groups[1]) == 0:
                QMessageBox.critical(self, "Input Error", "Each group must have at least one animal.")
                return
            resampler.setGroups(self.orig_groups, self.all_animals)
            running_process = lambda: resampler.generateResampledMatrix(correlational=False, num_resamples=num_resamples)

        # We seem to be having an issue with the multithreading using QThread interfering with multiprocessing, so for now run without the progress dialog
        running_process()
        resampled_matrix = resampler.getResampledMatrix()

        # progress_dialog = ProgressDialog("Resampling", num_resamples,
        #                                 resampler, running_process, resampler.getResampledMatrix,
        #                                 resampler.resample_start_signal, resampler.resample_progress_signal, resampler.resample_end_signal, self)
        # progress_dialog.displayPercentage()
        # resampled_matrix = progress_dialog.run()
        
        # p_values = stats_analyzer.getPValuesFullParallel(k=2, alpha=0.05)
        # print(p_values)
        # print("P-values calculated. Time taken: ", format_time(time.time() - section_start))
        # sys.exit()


        stats_analyzer = StatisticalAnalyzer(resampled_matrix)
        stats_analyzer.setParams(alpha, gamma)


        resample_dir = resampler.getDir()

        seq_num_index = SequencesProcessor.buildSeqNumIndex(resampler.getAllSeqCntsMatrix(), conts, self.parent_obj.getMaxSequenceLength(),
                                                            resample_dir)
        

        worker_function = None
        if self.fwerRadio.isChecked():
            # FWER Control only
            worker_function = stats_analyzer.FWERControl
        else:
            # FDP Control
            worker_function = stats_analyzer.fdpControl

        progress_dialog = ProgressDialog("Calculating p-values", 100,
                                         stats_analyzer, worker_function, stats_analyzer.getPValueResults,
                                            stats_analyzer.start_signal, stats_analyzer.progress_signal, stats_analyzer.end_signal, self)
        progress_dialog.displayPercentage()
        p_values, k = progress_dialog.run()
                                         
        
        



        # p_values, k = stats_analyzer.fdpControl(alpha=0.5, gamma=0.05, abbreviated=False)
        
        print(f"Stopped at k={k}")
        print(f"Number of significant sequences: {len(p_values)}")
        
        counts_dir = self.parent_obj.getDir()
        p_val_mat = []
        for p_val, seq_num, positively_correlated in p_values:
            seq, cont, length, local_num = SequencesProcessor.getSequence(seq_num, seq_num_index, counts_dir)
            p_val_mat.append([p_val, seq, cont, length, local_num, positively_correlated])
        p_val_mat = np.array(p_val_mat, dtype=object)
        p_val_file = CBASFile("significant_sequences", p_val_mat, col_headers=['P-Value', 'Sequence', 'Contingency', 'Length', 'Local Seq. No.', 'Positively Correlated'])
        p_val_file.saveFile(resample_dir)

        self.createResamples(resample_dir, seed, num_resamples, contingencies, 
                             self.orig_groups if not self.useCorrelationalCheckBox.isChecked() else None, 
                             alpha, gamma)
        
        


    def createResamples(self, directory, custom_seed, num_resamples, contingencies, groups, alpha, gamma):
        """Create the Resamples object and return it."""
        resamples = Resamples()
        resamples.createResamples(self.nameLineEdit.text(),
                                  self.descPlainTextEdit.toPlainText(),
                                  directory,
                                  self.useCorrelationalCheckBox.isChecked(),
                                  custom_seed,
                                  num_resamples,
                                  contingencies,
                                  groups,
                                  self.fdpRadio.isChecked(),
                                  alpha,
                                  gamma)
        resamples.setParent(self.parent_obj)
        self.returnValue = resamples
        self.close()



    def run(self):
        self.exec()
        return self.returnValue
    


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    # Spawn a group selector dialog
    dialog = GroupSelectorDialog(["Genotype", "Sex", "Lesion", "Implant"])
    dialog.run()
    
    sys.exit(app.exec())
    
    
