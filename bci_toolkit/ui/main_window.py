from __future__ import annotations
from PySide6 import QtWidgets, QtCore
from PySide6.QtGui import QAction
from ..views.signal_view import SignalView
from ..views.topomap_view import TopomapView
from .widgets import TimeSlider


class MainWindow(QtWidgets.QMainWindow):
    """Main application window with docks and a global time slider."""
    def __init__(self, data_model):
        super().__init__()
        self.setWindowTitle("BCI Toolkit")
        self.dm = data_model

        # Central widget: just a container for slider at bottom
        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        # Docks
        self.signal_view = SignalView(self, self.dm, window_sec=10.0, min_ch_px=30)
        self.topomap_view = TopomapView(self, self.dm, grid_res=120, lim=0.11)

        dock_sig = QtWidgets.QDockWidget("Signals", self)
        dock_sig.setWidget(self.signal_view)
        dock_sig.setObjectName("dock_signals")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock_sig)

        dock_topo = QtWidgets.QDockWidget("Topomap", self)
        dock_topo.setWidget(self.topomap_view)
        dock_topo.setObjectName("dock_topomap")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, dock_topo)

        dock_sig = QtWidgets.QDockWidget("Signals", self)
        dock_sig.setWidget(self.signal_view)
        dock_sig.setObjectName("dock_signals")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock_sig)

        dock_topo = QtWidgets.QDockWidget("Topomap", self)
        dock_topo.setWidget(self.topomap_view)
        dock_topo.setObjectName("dock_topomap")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, dock_topo)

        # ↓ вместо tabify делаем настоящий сплит: Topomap сверху, Signals снизу
        self.splitDockWidget(dock_topo, dock_sig, QtCore.Qt.Orientation.Vertical)
        self.setTabPosition(QtCore.Qt.AllDockWidgetAreas, QtWidgets.QTabWidget.TabPosition.North)

        # Global time slider
        self.slider = TimeSlider(self.dm.n_times, self.dm.sfreq)
        vbox.addWidget(self.slider)

        # Wiring: slider <-> views
        self.slider.timeChanged.connect(self.on_time_slider)
        self.signal_view.timeChanged.connect(self.on_view_time_change)

        # Initial state
        self.on_time_slider(0)

        # Menu for future extensibility
        self._build_menu()

    def _build_menu(self):
        bar = self.menuBar()
        view_menu = bar.addMenu("&View")

        act_stack_v = QAction("Stack Topomap ↑ Signals", self)
        def _stack_v():
            # Re-stack vertically (topomap above signals)
            self.removeDockWidget(self.findChild(QtWidgets.QDockWidget, "dock_signals"))
            self.removeDockWidget(self.findChild(QtWidgets.QDockWidget, "dock_topomap"))
            dock_topo = QtWidgets.QDockWidget("Topomap", self)
            dock_topo.setObjectName("dock_topomap")
            dock_topo.setWidget(self.topomap_view)
            dock_sig = QtWidgets.QDockWidget("Signals", self)
            dock_sig.setObjectName("dock_signals")
            dock_sig.setWidget(self.signal_view)
            self.addDockWidget(QtCore.Qt.TopDockWidgetArea, dock_topo)
            self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock_sig)
            self.splitDockWidget(dock_topo, dock_sig, QtCore.Qt.Vertical)
        act_stack_v.triggered.connect(_stack_v)
        view_menu.addAction(act_stack_v)

        act_side = QAction("Side-by-side (Left | Right)", self)
        def _side():
            # Put views side by side (left/right)
            self.removeDockWidget(self.findChild(QtWidgets.QDockWidget, "dock_signals"))
            self.removeDockWidget(self.findChild(QtWidgets.QDockWidget, "dock_topomap"))
            dock_topo = QtWidgets.QDockWidget("Topomap", self)
            dock_topo.setObjectName("dock_topomap")
            dock_topo.setWidget(self.topomap_view)
            dock_sig = QtWidgets.QDockWidget("Signals", self)
            dock_sig.setObjectName("dock_signals")
            dock_sig.setWidget(self.signal_view)
            self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock_topo)
            self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock_sig)
            self.splitDockWidget(dock_topo, dock_sig, QtCore.Qt.Horizontal)
        act_side.triggered.connect(_side)
        view_menu.addAction(act_side)

    # --- time sync ---
    def on_time_slider(self, idx: int):
        self.signal_view.on_time_index_changed(idx)
        self.topomap_view.on_time_index_changed(idx)

    def on_view_time_change(self, idx: int):
        self.slider.set_time(idx)
        self.topomap_view.on_time_index_changed(idx)