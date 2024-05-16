from ui.UI_SettingsDialog import Ui_SettingsDialog

from settings import Settings, Project, DataSet, Counts

from sequences import SequencesProcessor

from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox
from PyQt6.QtCore import Qt, QThread

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
            
        self.returnValue = None


    ### COUNTS ###

    def setupCountsSettings(self):
        """Set up the settings for Counts."""
        self.SettingsPages.setCurrentIndex(0)
        self.setWindowTitle("Counts Settings")
        self.progressBar.hide()
        self.useNumberButton.hide()
        self.progressLabel.hide()
        # self.minLabel.hide()
        self.maxLabel.hide()
        scanButtonControl = lambda: self.scanButton.setDisabled((self.criterionOrderLineEdit.text() == "") or (self.maxSeqLenLineEdit.text() == ""))
        self.criterionOrderLineEdit.textChanged.connect(scanButtonControl)
        self.maxSeqLenLineEdit.textChanged.connect(scanButtonControl)
        self.scanButton.clicked.connect(self.scanCriterionOrder)

        self.createButton.clicked.connect(self.createCounts)

    def scanCriterionOrder(self):
        """
        TODO: Histogram
        Scan the criterion order.
        """
        assert type(self.parent_obj) == DataSet
        # self.minLabel.show()
        

        order = int(self.criterionOrderLineEdit.text())
        inf_criterion = Settings.buildCriterionDict(order, float('inf'), self.criterionIncludeFailedCheck.isChecked(), self.criterionAllowRedemptionCheck.isChecked())
        counts_language = Settings.buildCountsLanguageDict(int(self.maxSeqLenLineEdit.text()), bool(self.straddleSessionsCheck.isChecked()))

        self.running_thread = QThread()
        processor = SequencesProcessor(self.parent_obj.getDir(), 
                                       self.parent_obj.getAnDataColumnNames(), 
                                       self.parent_obj.getLanguage(), 
                                       inf_criterion, 
                                       counts_language, 
                                       self.parent_obj.getNumAnimals())
        processor.moveToThread(self.running_thread)
        processor.start_scan_signal.connect(self.startScan)
        self.running_thread.started.connect(processor.scanCriterionOrder)
        self.running_thread.finished.connect(self.running_thread.deleteLater)

        processor.scan_progress_signal.connect(self.updateScanProgress)
        
        processor.scan_complete_signal.connect(self.running_thread.quit)
        processor.scan_complete_signal.connect(self.running_thread.deleteLater)
        processor.scan_complete_signal.connect(self.scanComplete)
        self.running_thread.start()

        # max_trials = processor.scanCriterionOrder()
        # self.maxLabel.setText(f"Max trials: {max_trials}")

        self.maxLabel.show()
        self.useNumberButton.show()
        # self.useNumberButton.clicked.connect(lambda: self.criterionNumberLineEdit.setText(str(max_trials)))
    
    def startScan(self):
        self.progressBar.show()
        self.progressLabel.show()
        self.progressBar.setValue(0)
    
    def updateScanProgress(self, progress: tuple):
        animal_num, name = progress
        self.progressBar.setValue(int(animal_num / self.parent_obj.getNumAnimals() * 100))
        self.progressBar.setFormat(f"{animal_num} / {self.parent_obj.getNumAnimals()}")
        self.progressLabel.setText(f"Scanning subject: {name}")

    def scanComplete(self, mat):
        self.progressBar.hide()
        self.progressLabel.hide()
        self.maxLabel.show()
        self.scanButton.setDisabled(True)
        print(mat)




    def runCounts(self, criterion, counts_language):
        assert type(self.parent_obj) == DataSet
        processor = SequencesProcessor(self.parent_obj.getDir(),
                                        self.parent_obj.getAnDataColumnNames(),
                                        self.parent_obj.getLanguage(),
                                        criterion,
                                        counts_language,
                                        self.parent_obj.getNumAnimals())
        

    def createCounts(self):
        """Create the Counts object and return it."""
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
        criterion = Settings.buildCriterionDict(int(self.criterionOrderLineEdit.text()), number, self.criterionIncludeFailedCheck.isChecked(), self.criterionAllowRedemptionCheck.isChecked())
        counts_language = Settings.buildCountsLanguageDict(int(self.maxSeqLenLineEdit.text()), bool(self.straddleSessionsCheck.isChecked()))

        counts = Counts()
        counts.createCounts(self.nameLineEdit.text(),
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
    
    
