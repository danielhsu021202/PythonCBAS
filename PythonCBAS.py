from sequences import SequencesProcessor
from resampler import Resampler
from statistical_analyzer import StatisticalAnalyzer
from files import FileManager
from settings import Settings

import sys
import os
import time
import numpy as np
import re
import argparse
import datetime

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QFileDialog, QMessageBox, QDialog
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt

import qdarktheme

from ui.MainWindow import Ui_MainWindow
from ui.Lobby import Ui_Lobby

from FileViewer import FileViewer
from ImportData import ImportData
from Card import Card
from settings import Project



from utils import FileUtils
from files import CBASFile

import datetime

def format_time(seconds):
    duration = datetime.timedelta(seconds=seconds)
    if duration < datetime.timedelta(milliseconds=1):
        return "{:.2f} microseconds".format(duration.microseconds)
    elif duration < datetime.timedelta(seconds=1):
        return "{:.2f} milliseconds".format(duration.microseconds / 1000)
    elif duration < datetime.timedelta(minutes=1):
        return "{:.2f} seconds".format(duration.total_seconds())
    elif duration < datetime.timedelta(hours=1):
        return "{:.2f} minutes".format(duration.total_seconds() / 60)
    elif duration < datetime.timedelta(days=1):
        return "{:.2f} hours".format(duration.total_seconds() / 3600)
    else:
        return "{:.2f} days".format(duration.total_seconds() / 86400)


def startCBASTerminal(num_samples):
    divider = "=" * 65

    start = time.time()

    section_start = time.time()
    print("Starting PythonCBAS Engine...")
    print(divider)
    print("Retrieving settings...")
    settings = Settings()
    # settings.setCriterion({'ORDER': 0, 'NUMBER': float('inf'), 'INCLUDE_FAILED': True, 'ALLOW_REDEMPTION': True})
    settings.setCriterion({'ORDER': 4, 'NUMBER': 100, 'INCLUDE_FAILED': True, 'ALLOW_REDEMPTION': True})
    CRITERION = settings.getCriterion()
    print(f'''CRITERION:  Order {CRITERION['ORDER']}, 
            Number {CRITERION['NUMBER']}, 
            {'Include' if CRITERION['INCLUDE_FAILED'] else 'Exclude'} Failed, 
            {'' if CRITERION['ALLOW_REDEMPTION'] else "Don't "}Allow Redemption''')
    conts = [int(cont) for cont in args.contingencies.split(",")] if args.contingencies != "all" else "all"
    print("Settings retrieved. Time taken: ", format_time(time.time() - section_start))

    print(divider)
    section_start = time.time()
    print("Setting up files...")
    fileManager = FileManager(settings.getFiles())
    fileManager.setupFiles()
    print("Files set up. Time taken: ", format_time(time.time() - section_start))

    print(divider)
    section_start = time.time()
    print("Processing sequences and calculating criterion...")
    sequencesProcessor = SequencesProcessor(settings)
    sequencesProcessor.processAllAnimals()
    print("Sequences and criterion processed. Time taken: ", format_time(time.time() - section_start))

    print(divider)
    section_start = time.time()
    print("Generating sequence and criterion files...")
    sequencesProcessor.generateSequenceFiles()
    all_seq_cnts = sequencesProcessor.exportAllSeqCnts(conts)
    print("Sequence and criterion files generated. Time taken: ", format_time(time.time() - section_start))

    sequencesProcessor.dumpMemory()

    print(divider)
    section_start = time.time()
    print("Grouping animals...")
    
    
    resampler = Resampler(settings, conts=conts)
    resampler.setAllSeqCntsMatrix(all_seq_cnts)
    resampler.assignGroups([{"GENOTYPE": 0, "LESION": 0}, {"GENOTYPE": 1, "LESION": 0}])
    print("Groups assigned:")
    for i, group in enumerate(resampler.orig_groups):
        print(f"Group {i + 1}: {len(group)} animals")
    print("Time taken: ", format_time(time.time() - section_start))
    section_start = time.time()
    print("Calculating sequence rates...")
    seq_rates_matrix = resampler.getSequenceRatesVerbose(resampler.orig_groups)
    print("Sequence rates calculated. Time taken: ", format_time(time.time() - section_start))
    section_start = time.time()
    print("Writing sequence rates to file...")
    resampler.writeSequenceRatesFile(seq_rates_matrix)
    print("Sequence rates written to file. Time taken: ", format_time(time.time() - section_start))

    print(divider)
    section_start = time.time()
    print(f"Resampling {num_samples} times, considering contingencies {conts}...")
    resampled_matrix = resampler.generateResampledMatrixParallel(num_resamples=num_samples)
    print(f"Resampling finished. Time taken: ", format_time(time.time() - section_start))
    section_start = time.time()
    print("Writing resampled matrix to file...")
    cont_str = "all" if conts == "all" else "_".join([str(cont) for cont in conts])
    resampler.writeResampledMatrix(resampled_matrix, filename=f'resampled_mat_{num_samples}_samples_cont_{cont_str}')
    print("Resampled matrix written to file. Time taken: ", format_time(time.time() - section_start))



    print(divider)
    section_start = time.time()
    print("Calculating p-values...")
    stats_analyzer = StatisticalAnalyzer(resampled_matrix)
    # p_values = stats_analyzer.getPValuesFullParallel(k=2, alpha=0.05)
    # print(p_values)
    # print("P-values calculated. Time taken: ", format_time(time.time() - section_start))
    # sys.exit()

    p_values, k = stats_analyzer.fdpControl(alpha=0.5, gamma=0.05, abbreviated=False)
    
    print(f"Stopped at k={k}")
    print("P-values calculated. Time taken: ", format_time(time.time() - section_start))

    section_start = time.time()
    print("Writing significant sequences to file...")

    p_val_mat = []
    for p_val, seq_num in p_values:
        seq, cont, length, local_num = sequencesProcessor.getSequence(seq_num)
        p_val_mat.append([p_val, seq, cont, length, local_num])
    p_val_mat = np.array(p_val_mat, dtype=object)
    p_val_file = CBASFile("significant_sequences", p_val_mat)
    p_val_file.saveFile(settings.getFiles()['OUTPUT'])

    print(f"Significant sequences written to file. {len(p_val_mat)} sequences found. \nTime taken: ", format_time(time.time() - section_start))

    print(divider)
    print(f"Total Time: {format_time(time.time() - start)}")
    sys.exit()


class Lobby(QDialog, Ui_Lobby):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("PythonCBAS")

        self.mainStack.setCurrentIndex(0)
        self.newProjectButton.clicked.connect(lambda: self.mainStack.setCurrentIndex(1))
        self.cancelButton.clicked.connect(self.reset)
        self.createProjectButton.clicked.connect(self.createProject)
        self.projectLocationButton.clicked.connect(self.getDirectory)
        self.openProjectButton.clicked.connect(self.getProject)

        self.returnValue = None

    def run(self):
        self.exec()
        return self.returnValue
    
    def getDirectory(self):
        dir = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.projectLocationField.setText(dir)

    def getProject(self):
        """Open a file dialog to get .json or .cbasproj file"""
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Project", filter="PythonCBAS Project (*.json *.cbasproj)")
        if filepath:
            self.loadProject(filepath)

    def reset(self):
        self.mainStack.setCurrentIndex(0)
        self.projectNameField.clear()
        self.projectLocationField.clear()
        self.descriptionTextEdit.clear()

    def createProject(self):
        filepath = os.path.join(self.projectLocationField.text(), self.projectNameField.text() + ".json")
        if os.path.exists(filepath):
            QMessageBox.warning(self, "Project Exists", "A project with that name already exists in the specified location.")
            return
        project = Project()
        datecreated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        project.createProject(self.projectNameField.text(), self.descriptionTextEdit.toPlainText(), 
                              datecreated, self.projectLocationField.text(), "beta")
        project.writeProject()
        self.returnValue = project
        self.close()

    def loadProject(self, filepath):
        project = Project()
        project.readProject(filepath)
        self.returnValue = project
        self.close()
        
                              


class PythonCBAS(QMainWindow, Ui_MainWindow):
    def __init__(self, project: Project):
        super().__init__()
        self.setupUi(self)
        
        
        self.project = project

        self.setWindowTitle(project.getProjectName())

        self.setUpMenuBar()



        # Add the File Viewer Frame to the main staci
        self.fileViewer = FileViewer()
        self.FileViewerPage.layout().addWidget(self.fileViewer)

        # Set size
        self.resize(1920, 1080)
        # self.showFullScreen()
        
        self.mainStack.setCurrentIndex(1)


    def importData(self):
        self.importDataDialog = ImportData()
        self.importDataDialog.exec()

    def showExampleCard(self):
        dialog = QDialog()
        card = Card(dialog)

        layout = QVBoxLayout()
        layout.addWidget(card)

        dialog.setLayout(layout)

        dialog.exec()

    def setUpMenuBar(self):
        self.menubar = self.menuBar()
        # self.menubar.setNativeMenuBar(False)  # For macOS
        self.actionGet_Sequences.triggered.connect(self.runCBAS)
        self.actionDarkTheme.triggered.connect(lambda: qdarktheme.setup_theme("dark", additional_qss=qss))
        self.actionLightTheme.triggered.connect(lambda: qdarktheme.setup_theme("light"))
        self.actionAutoTheme.triggered.connect(lambda: qdarktheme.setup_theme("auto", additional_qss=qss))
        self.actionImport_Data.triggered.connect(lambda: self.mainStack.setCurrentIndex(0))
        self.actionFile_Viewer.triggered.connect(lambda: self.mainStack.setCurrentIndex(1))
        self.actionImport_Data_Dialog.triggered.connect(self.importData)
        self.actionCard.triggered.connect(self.showExampleCard)

    def runCBAS(self):
        startCBASTerminal()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PythonCBAS")
    parser.add_argument("-s", "--sequence", help="Run sequencing in terminal mode", action="store_true")
    parser.add_argument("-n", "--num_samples", help="Number of samples to resample", type=int, default=0)
    parser.add_argument("-c", "--contingencies", help="List of contingencies to include in the resampling, comma-separated", type=str, default="all")
    args = parser.parse_args()

    if args.sequence:
        if args.sequence:
            startCBASTerminal(args.num_samples)
        sys.exit()
    else:
        app = QApplication(sys.argv)

        # Load App Configurations


        # Load ui/styles.qss
        with open("ui/styles.qss", "r") as f:
            qss = f.read()

        

        qdarktheme.setup_theme("auto", additional_qss=qss)


        
        
        lobby = Lobby()
        project = lobby.run()
        if project is not None:
            mainWindow = PythonCBAS(project)
            mainWindow.show()
        else:
            sys.exit()

        sys.exit(app.exec())
