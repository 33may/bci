from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore

class ChannelWindow(QtWidgets.QDialog):
    """Popup window to inspect a single channel at larger scale."""
    def __init__(self, parent, ch_name: str, t: np.ndarray, y: np.ndarray):
        super().__init__(parent)
        self.setWindowTitle(f"Channel: {ch_name}")
        self.resize(600, 300)
        layout = QtWidgets.QVBoxLayout(self)

        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self.plot)

        self.curve = self.plot.plot(t, y, pen=pg.mkPen(200))
        self.cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((50, 200, 50), width=2))
        self.plot.addItem(self.cursor)

        self._t = t
        self._y = y

    def update_time_cursor(self, t_float: float):
        self.cursor.setX(t_float)

    def update_series(self, t: np.ndarray, y: np.ndarray):
        self._t, self._y = t, y
        self.curve.setData(t, y)