from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QDialog, QWidget, QApplication, QVBoxLayout

from ui.Card_ui import Ui_Card

class Card(QWidget, Ui_Card):
    def __init__(self, parent=None):
        super(Card, self).__init__()
        self.setupUi(self)


        # Set stylesheet
        self.setStyleSheet("""
                            #CardFrame {
                                border: 4px solid rgb(75, 75, 75);
                                border-radius: 16px;
                                background-color: rgb(105, 105, 105);
                           }
                           """)
        
        # Set size
        self.setFixedSize(250, 150)





if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    # spawn a card in a dialog
    dialog = QDialog()
    card = Card(dialog)

    layout = QVBoxLayout()
    layout.addWidget(card)

    dialog.setLayout(layout)

    dialog.show()
    sys.exit(app.exec())
