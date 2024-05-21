import os
qtwebengine_dictionaries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtwebengine_dictionaries")
if not os.path.exists(qtwebengine_dictionaries_path):
    os.mkdir(qtwebengine_dictionaries_path)
os.environ["QTWEBENGINE_DICTIONARIES_PATH"] = qtwebengine_dictionaries_path
from PyQt6.QtWebEngineWidgets import QWebEngineView

from PyQt6.QtWidgets import QApplication, QMainWindow, QDialog, QVBoxLayout

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

import pandas as pd

from utils import WebUtils


class Visualizer:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.visualizations = []
        self.web_view = QWebEngineView()


    def createScatterPlot(self, matrix):
        # Scatter plot of the list
        # x is index, y is max of that row
        df = pd.DataFrame(matrix)
        # fig = px.scatter(df, x=df.index, y=df.max(axis=1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df.max(axis=1), mode='markers'))
        fig.update_layout(title="Scatter Plot", xaxis_title="Index", yaxis_title="Max Value")
        
        # Display the plot in pyqt

        self.visualizations.append(fig)

    
        

    def showWindow(self):
        dialog = QDialog(parent=self.parent)
        layout = QVBoxLayout()
        fig = self.visualizations[0]
        self.web_view.setHtml(pio.to_html(fig, include_plotlyjs='cdn'))
        layout.addWidget(self.web_view)
        dialog.setLayout(layout)
        dialog.resize(800, 600)
        dialog.show()
        







