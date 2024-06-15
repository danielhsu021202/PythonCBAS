from PyQt6.QtWidgets import QWidget

import sys

from ui.UI_InfoDisplay import Ui_InfoDisplay
from settings import Project, DataSet, Counts, Resamples, Visualizations

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
        self.titleLabel.setText(self.obj.getName())
        self.descriptionTextBrowser.setText(self.obj.getDescription())
        


