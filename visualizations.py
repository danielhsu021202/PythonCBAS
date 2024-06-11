import os
from settings import Settings
qtwebengine_dictionaries_path = os.path.join(Settings.getAppDataFolder(), "qtwebengine_dictionaries")
if not os.path.exists(qtwebengine_dictionaries_path):
    os.mkdir(qtwebengine_dictionaries_path)
os.environ["QTWEBENGINE_DICTIONARIES_PATH"] = qtwebengine_dictionaries_path
from PyQt6.QtWebEngineWidgets import QWebEngineView
import sys

from ui.UI_VisualizerWindow import Ui_VisualizerWindow

from PyQt6.QtWidgets import QApplication, QMainWindow, QDialog, QVBoxLayout, QFileDialog, QMessageBox

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

import pandas as pd

from utils import WebUtils

class VisualizerWindow(QDialog, Ui_VisualizerWindow):
    def __init__(self, name, visualizations, parent=None):
        super(VisualizerWindow, self).__init__()
        self.setupUi(self)

        self.parent = parent
        self.setWindowTitle(name)
        self.visualizations = visualizations

        self.web_view = QWebEngineView()
        self.webEngineWidget.setLayout(QVBoxLayout())
        self.webEngineWidget.layout().setContentsMargins(0, 0, 0, 0)
        self.webEngineWidget.layout().addWidget(self.web_view)

        


        self.html = WebUtils.htmlStartPlotly()

        for fig in self.visualizations:
            self.html += pio.to_html(fig, include_plotlyjs=WebUtils.plotly_latest_js_url, full_html=False)

        self.html += WebUtils.htmlEnd()

        self.web_view.setHtml(self.html)

        self.htmlButton.clicked.connect(self.toHTML)

        self.resize(1200, 900)

    def toHTML(self):
        file, _ = QFileDialog.getSaveFileName(self.parent, "Save HTML", filter="HTML Files (*.html)")
        if file:
            with open(file, 'w') as f:
                f.write(self.html)



class Visualizer:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.visualizations = []


    def createScatterPlot(self, matrix, title="", xaxis_title="", yaxis_title=""):
        # Scatter plot of the list
        # x is index, y is max of that row
        df = pd.DataFrame(matrix)
        # fig = px.scatter(df, x=df.index, y=df.max(axis=1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df.max(axis=1), mode='markers'))
        fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
        
        # Display the plot in pyqt

        self.visualizations.append(fig)

    
        

    def showWindow(self):
        if not WebUtils.check_internet_connection():
            QMessageBox.critical(self.parent, "No Internet Connection", "Currently, internet connection is required for generating and viewing plots.")
            return
        VisualizerWindow(self.name, self.visualizations, self.parent).exec()
        







