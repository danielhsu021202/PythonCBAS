from ui.UI_SettingsDialog import Ui_SettingsDialog

from settings import Settings, Project, DataSet, Counts

from sequences import SequencesProcessor

from utils import TimeUtils, ReturnContainer

from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox, QLabel, QProgressBar, QVBoxLayout
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6 import QtGui

import time

class ProgressDialog(QDialog):
    def __init__(self, title: str, progress_max: int, 
                 worker: QThread, worker_function, retrieval_function, 
                 start_signal: pyqtSignal, running_signal: pyqtSignal, end_signal: pyqtSignal, 
                 parent=None):
        super(ProgressDialog, self).__init__(parent)

        self.progress_max = progress_max
        self.returnValue = None
        self.worker = worker
        self.display_percent = False
        
        # UI Setup
        self.resize(300, 150)

        self.setWindowTitle(title)
        self.progressLabel = QLabel()
        self.progressBar = QProgressBar()
        self.timeElapsed = QLabel()

        self.timeElapsed.setText("Time elapsed: 0.00 seconds")

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.progressLabel)
        self.layout.addWidget(self.progressBar)
        self.layout.addWidget(self.timeElapsed)

        self.setLayout(self.layout)
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
        self.display_percent = True

    def displayProportion(self):
        self.display_percent = False

    def start(self):
        self.progressBar.setValue(0)
        self.startTime = time.time()  # Record the start time

    def updateProgress(self, progress: tuple):

        # progress is (value, name) pair
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
        self.timer.stop()  # Stop the timer when the operation is finished
        self.returnValue = return_value
        self.close()

    def run(self):
        self.running_thread.start()
        self.exec()
        return self.returnValue





class SettingsDialog(QDialog, Ui_SettingsDialog):
    def __init__(self, setting_type: str, proj_obj: Project, parent_obj=None):
        super(SettingsDialog, self).__init__()
        self.setupUi(self)
        self.type = setting_type

        self.proj_obj = proj_obj
        self.parent_obj = parent_obj

        if self.type == "counts":
            assert type(self.parent_obj) == DataSet
            self.setupCountsSettings()
        elif self.type == "resamples":
            assert type(self.parent_obj) == Counts
            self.setupResampleSettings()
            
        self.returnValue = None


    ### COUNTS ###

    def setupCountsSettings(self):
        """Set up the settings for Counts."""
        self.SettingsPages.setCurrentIndex(0)
        self.setWindowTitle("Counts Settings")
        self.progressBar.hide()
        self.progressLabel.hide()

        self.criterionOrderLineEdit.textChanged.connect(lambda: self.scanButton.setDisabled(self.criterionOrderLineEdit.text() == ""))

        self.scanButton.clicked.connect(self.scanCriterionOrder)
        self.createButton.clicked.connect(self.runCounts)

        if not self.parent_obj.getHasModifier():
            self.criterionOrderLineEdit.setText(str(0))
            self.criterionOrderLineEdit.setDisabled(True)
            self.criterionOrderLineEdit.setToolTip("This dataset has no modifier, so only criterion order 0 can be used.")

    def scanCriterionOrder(self):
        """
        TODO: Histogram
        Scan the criterion order.
        """
        assert type(self.parent_obj) == DataSet

        order = int(self.criterionOrderLineEdit.text())

        if order < 0:
            QMessageBox.critical(self, "Input Error", "Criterion order must be non-negative.")
            return

        inf_criterion = Settings.buildCriterionDict(order, float('inf'), self.criterionIncludeFailedCheck.isChecked(), self.criterionAllowRedemptionCheck.isChecked())
        counts_language = Settings.buildCountsLanguageDict(order, bool(self.straddleSessionsCheck.isChecked()))  # Can use order as the max sequence length

        # self.running_thread = QThread()
        processor = SequencesProcessor(None,  # Name is none because we're not saving this, just scanning.
                                        self.parent_obj.getDir(), 
                                       self.parent_obj.getAnDataColumnNames(), 
                                       self.parent_obj.getLanguage(), 
                                       inf_criterion, 
                                       counts_language, 
                                       self.parent_obj.getNumAnimals())
        criterion_matrix = ProgressDialog(f"Scanning Criterion Order {order}", self.parent_obj.getNumAnimals(),
                                          processor, processor.scanCriterionOrder, processor.getCriterionMatrix,
                                          processor.start_processing_signal, processor.processing_progress_signal, processor.scan_complete_signal, self).run()
    


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
        if (self.criterionOrderLineEdit.text() == "") or (not self.criterionOrderLineEdit.text().isnumeric()):
            QMessageBox.critical(self, "Input Error", "Criterion order not filled or must be numeric.")
            return
        if (self.maxSeqLenLineEdit.text() == "") or (not self.maxSeqLenLineEdit.text().isnumeric()):
            QMessageBox.critical(self, "Input Error", "Max sequence length not filled or must be numeric.")
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
        criterion = Settings.buildCriterionDict(order, number, self.criterionIncludeFailedCheck.isChecked(), self.criterionAllowRedemptionCheck.isChecked())
        counts_language = Settings.buildCountsLanguageDict(max_seq_len, bool(self.straddleSessionsCheck.isChecked()))

        processor = SequencesProcessor(self.nameLineEdit.text(),
                                        self.parent_obj.getDir(),
                                        self.parent_obj.getAnDataColumnNames(),
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
        self.createButton.clicked.connect(self.createResamples)

    def createResamples(self):
        pass



    def run(self):
        self.exec()
        return self.returnValue
    
    
