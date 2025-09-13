from __future__ import annotations
from PySide6 import QtWidgets

class BaseView(QtWidgets.QWidget):
    """Common base for views. Exposes update hooks."""
    def on_time_index_changed(self, t_idx: int):
        """Called when global time index changes."""
        pass