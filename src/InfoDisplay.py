from PyQt6.QtWidgets import QWidget

import sys

from ui.UI_InfoDisplay import Ui_InfoDisplay
from settings import Project, DataSet, Counts, Resamples, Visualizations
from utils import FileUtils, StringUtils, TimeUtils
from datetime import datetime

class InfoDisplay(QWidget, Ui_InfoDisplay):
    
    def __init__(self, obj, filepath: str):
        super(InfoDisplay, self).__init__()
        self.setupUi(self)

        if filepath is not None:
            self.filepath = filepath
            if filepath.endswith(".cbas"):
                self.mode = "cbasfile"
            else:
                self.mode = "file"
            

        else:
            self.mode = "obj"

        self.obj = obj
        self.obj_type = type(obj)


        self.setBasicInformation()

    def setBasicInformation(self):
        # Set window title
        self.titleLabel.setText(self.obj.getName())
        self.descriptionTextBrowser.setText(self.obj.getDescription())
        if self.obj.getDateModified():
            # Reformat to "Monday, January 1, 2021 at 12:00 AM"
            dateModifiedFormatted = TimeUtils.reformat(self.obj.getDateModified(), "blunt", "readable")
        else:
            dateModifiedFormatted = "Unknown"
        self.subtitleLabel.setText(f"Modified: {dateModifiedFormatted}")
        self.sizeLabel.setText(str(StringUtils.formatSize(FileUtils.sizeOfFolder(self.obj.getDir()))))

        # General
        self.general_kindLabel.setText(self.obj.getType().title())
        self.general_sizeLabel.setText(f"{FileUtils.sizeOfFolder(self.obj.getDir(), comma=True)} bytes ({StringUtils.formatSize(FileUtils.sizeOfFolder(self.obj.getDir()))} on disk)")
        self.general_whereLabel.setText(self.obj.getDir())
        self.general_createdLabel.setText(TimeUtils.reformat(self.obj.getDateCreated(), "blunt", "readable"))
        self.general_modifiedLabel.setText(dateModifiedFormatted)
        


