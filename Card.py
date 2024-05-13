from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QDialog, QWidget, QApplication, QVBoxLayout

from ui.Card_ui import Ui_Card
from settings import DataSet, next_type

from utils import StringUtils

class Card(QWidget, Ui_Card):
    def __init__(self, item=None, parent=None):
        super(Card, self).__init__()
        self.setupUi(self)

        self.item = item
        self.navigator = parent

        if self.item:
            self.type = self.item.getType() if self.item is not None else None
            title, subtitle = self.item.getCardInfo()
            self.TitleLabel.setText(title)
            self.TypeLabel.setText(StringUtils.capitalizeFirstLetter(self.type))
            self.SubtitleLabel.setText(subtitle)
        

        if self.item is None:
            self.CardTypes.setCurrentIndex(0)
        else:
            self.CardTypes.setCurrentIndex(1)
            if next_type[self.type]:
                self.mouseDoubleClickEvent = lambda _: self.navigator.populateItems(self.item.getChildren(), next_type[self.type])

        # Set stylesheet
        self.setStyleSheet("""
                            #CardFrame {
                                border: 4px solid rgb(75, 75, 75);
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
                                border: 4px solid rgb(75, 75, 75);
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
