from __future__ import annotations
from PySide6 import QtWidgets, QtCore

class TimeSlider(QtWidgets.QSlider):
    """Horizontal time slider bound to [0, n_times-1]."""
    timeChanged = QtCore.Signal(int)

    def __init__(self, n_times: int, sfreq: float, parent=None):
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self.setRange(0, max(0, n_times - 1))
        self.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.setTickInterval(int(max(1, sfreq)))
        self.setFixedHeight(30)
        self.valueChanged.connect(self._emit_time)

    def _emit_time(self, v: int):
        self.timeChanged.emit(int(v))

    def set_time(self, idx: int):
        self.blockSignals(True)
        self.setValue(int(idx))
        self.blockSignals(False)