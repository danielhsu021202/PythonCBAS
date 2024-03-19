from sequences import SequencesProcessor
from files import FileManager
from settings import Settings

import sys
import os
import time
import numpy as np
import re

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


def startCBASTerminal():
    divider = "=" * 65

    start = time.time()

    section_start = time.time()
    print("Starting PythonCBAS Engine...")
    print(divider)
    print("Retrieving settings...")
    settings = Settings()
    # settings.setCriterion({'ORDER': 0, 'NUMBER': float('inf'), 'INCLUDE_FAILED': True, 'ALLOW_REDEMPTION': True})
    settings.setCriterion({'ORDER': 0, 'NUMBER': float('inf'), 'INCLUDE_FAILED': True, 'ALLOW_REDEMPTION': True})
    FILES = settings.getFiles()
    ANIMAL_FILE_FORMAT = settings.getAnimalFileFormat()
    LANGUAGE = settings.getLanguage()
    CRITERION = settings.getCriterion()
    CONSTANTS = settings.getConstants()
    print(f'''CRITERION:  Order {CRITERION['ORDER']}, 
            Number {CRITERION['NUMBER']}, 
            {'Include' if CRITERION['INCLUDE_FAILED'] else 'Exclude'} Failed, 
            {'' if CRITERION['ALLOW_REDEMPTION'] else "Don't "}Allow Redemption''')
    print("Settings retrieved. Time taken: ", format_time(time.time() - section_start))

    print(divider)
    section_start = time.time()
    print("Setting up files...")
    fileManager = FileManager(FILES)
    fileManager.setupFiles()
    print("Files set up. Time taken: ", format_time(time.time() - section_start))

    print(divider)
    section_start = time.time()
    print("Processing sequences and calculating criterion...")
    sequencesProcessor = SequencesProcessor(FILES, ANIMAL_FILE_FORMAT, LANGUAGE, CRITERION, CONSTANTS)
    sequencesProcessor.processAllAnimals()
    print("Sequences and criterion processed. Time taken: ", format_time(time.time() - section_start))

    print(divider)
    section_start = time.time()
    print("Generating sequence and criterion files...")
    sequencesProcessor.generateSequenceFiles()
    print("Sequence and criterion files generated. Time taken: ", format_time(time.time() - section_start))

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
        self.resize(1600, 900)


    

    def setUpMenuBar(self):
        self.menubar = self.menuBar()
        self.menubar.setNativeMenuBar(False)  # For macOS
        self.actionGet_Sequences.triggered.connect(self.runCBAS)
        self.actionDark_Theme.triggered.connect(lambda: qdarktheme.setup_theme("dark"))
        self.actionLight_Theme.triggered.connect(lambda: qdarktheme.setup_theme("light"))
        self.actionAuto.triggered.connect(lambda: qdarktheme.setup_theme("auto"))
        self.actionImport_Data.triggered.connect(lambda: self.mainStack.setCurrentIndex(0))
        self.actionFile_Viewer.triggered.connect(lambda: self.mainStack.setCurrentIndex(1))

    def runCBAS(self):
        startCBASTerminal()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load ui/styles.qss
    with open("ui/styles.qss", "r") as f:
        qss = f.read()

    

    qdarktheme.setup_theme("auto", additional_qss=qss)


    
    
    window = PythonCBAS()
    window.show()
    sys.exit(app.exec())
