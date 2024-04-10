from sequences import SequencesProcessor
from resampler import Resampler
from files import FileManager
from settings import Settings

import sys
import os
import time
import numpy as np
import re
import argparse

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTreeWidgetItemIterator
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt

import qdarktheme

from ui.MainWindow import Ui_MainWindow

from FileViewer import FileViewer

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
    all_seq_cnts = sequencesProcessor.exportAllSeqCnts()
    print("Sequence and criterion files generated. Time taken: ", format_time(time.time() - section_start))

    sequencesProcessor = None  # Clear memory

    print(divider)
    section_start = time.time()
    print("Grouping animals...")
    conts = [int(cont) for cont in args.contingencies.split(",")] if args.contingencies != "all" else "all"
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
    resampler.writeResampledMatrix(resampled_matrix, filename='resampled_mat_1000_samples_cont_1')
    print("Resampled matrix written to file. Time taken: ", format_time(time.time() - section_start))
    


    

    

    print(divider)
    print(f"Total Time: {format_time(time.time() - start)}")


class PythonCBAS(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("PythonCBAS")
        

        self.setUpMenuBar()



        # Add the File Viewer Frame to the main staci
        self.fileViewer = FileViewer()
        self.FileViewerPage.layout().addWidget(self.fileViewer)

        # Set size
        self.resize(1920, 1080)
        # self.showFullScreen()
        
        self.mainStack.setCurrentIndex(1)


    

    def setUpMenuBar(self):
        self.menubar = self.menuBar()
        self.menubar.setNativeMenuBar(False)  # For macOS
        self.actionGet_Sequences.triggered.connect(self.runCBAS)
        self.actionDarkTheme.triggered.connect(lambda: qdarktheme.setup_theme("dark", additional_qss=qss))
        self.actionLightTheme.triggered.connect(lambda: qdarktheme.setup_theme("light"))
        self.actionAutoTheme.triggered.connect(lambda: qdarktheme.setup_theme("auto", additional_qss=qss))
        self.actionImport_Data.triggered.connect(lambda: self.mainStack.setCurrentIndex(0))
        self.actionFile_Viewer.triggered.connect(lambda: self.mainStack.setCurrentIndex(1))

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

        # Load ui/styles.qss
        with open("ui/styles.qss", "r") as f:
            qss = f.read()

        

        qdarktheme.setup_theme("auto", additional_qss=qss)


        
        
        window = PythonCBAS()
        window.show()
        sys.exit(app.exec())
