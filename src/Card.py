from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QDialog, QWidget, QApplication, QVBoxLayout, QMenu, QMessageBox, QInputDialog, QMainWindow

from ui.Card_ui import Ui_Card
from settings import DataSet, Counts, Resamples, next_type

from utils import StringUtils

class Card(QWidget, Ui_Card):
    def __init__(self, obj=None, parent=None):
        super(Card, self).__init__()
        self.setupUi(self)

        self.obj = obj
        self.navigator = parent

        if self.obj:
            self.type = self.obj.getType() if self.obj is not None else None
            title, subtitle = self.obj.getCardInfo()
            self.TitleLabel.setText(title)
            self.TitleLabel.setWordWrap(True)
            self.TypeLabel.setText(StringUtils.capitalizeFirstLetter(self.type))
            self.SubtitleLabel.setText(subtitle)
        

        if self.obj is None:
            self.CardTypes.setCurrentIndex(0)
        else:
            self.CardTypes.setCurrentIndex(1)
            if next_type[self.type] is not None:
                self.mouseDoubleClickEvent = lambda _: self.navigator.populateItems(self.obj, next_type[self.type])

            # On right click, spawn menu
            self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            self.customContextMenuRequested.connect(self.menuActions)

        # Set stylesheet
        self.setStyleSheet("""
                            #CardFrame {
                                border: 3px solid rgb(75, 75, 75);
                                border-radius: 16px;
                                background-color: rgb(105, 105, 105);
                            }
                            #CardFrame:hover {
                                background-color: rgb(125, 125, 125);
                            }
                            #CardFrame:pressed {
                                background-color: rgb(85, 85, 85);
                            }
                            #newItemButton {
                                border: 3px solid rgb(75, 75, 75);
                                border-radius: 16px;
                                background-color: rgb(105, 105, 105);
                            }
                            #newItemButton:hover {
                                background-color: rgb(125, 125, 125);
                            }
                            #newItemButton:pressed {
                                  background-color: rgb(85, 85, 85);
                            }
                            #TitleLabel {
                                color: rgb(255, 255, 255);
                           }
                            #TypeLabel {
                                color: rgb(255, 255, 255);
                            }
                            #SubtitleLabel {
                                color: rgb(255, 255, 255);
                            }
                            
                           """)
        
        # Set size
        self.setFixedSize(300, 200)

    def menuActions(self):
        menu = QMenu(self)
        # Type Specific Actions
        if type(self.obj) == DataSet:
            pass
        elif type(self.obj) == Counts:
            pass
        elif type(self.obj) == Resamples:
            pass
        
        menu.addSeparator()

        # General Actions
        getinfo_action = menu.addAction("Get Info")
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")
        fileviewer_action = menu.addAction("Open in FileViewer")


        # Execute at mouse position
        action = menu.exec(QtGui.QCursor.pos())

        # Actions
        if action == getinfo_action:
            self.navigator.spawnInfoDisplay(self.obj, None)
        elif action == rename_action:
            # Prompt for new name
            new_name, ok = QInputDialog.getText(self, 'Rename', 'Enter a new name:', text=self.obj.getName())
            if ok:
                self.obj.setName(new_name)
                self.obj.renameDir(new_name)
                self.commitChanges()
            
        elif action == delete_action:
            # Prompt for deletion
            reply = QMessageBox.question(self, 'Delete', f"Are you sure you want to delete {self.obj.getName()} and all the files associated with it?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.obj.getParent().deleteChild(self.obj)
                self.commitChanges()
        elif action == fileviewer_action:
            self.navigator.spawnFileViewer(self.obj.getDir())

    def commitChanges(self):
        self.navigator.getProject().writeProject()
        self.navigator.populateItems(self.navigator.obj, self.navigator.mode)

    

            


        






if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    # spawn a card in a dialog
    dialog = QDialog()

    layout = QVBoxLayout()
    layout.addWidget(Card())
    layout.addWidget(Card(DataSet()))

    dialog.setLayout(layout)

    dialog.show()
    sys.exit(app.exec())
