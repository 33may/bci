# bci_toolkit/app.py (в самом верху файла)
from __future__ import annotations
import os, site, glob

def _ensure_qt_platform():
    # Prefer platform plugins that ship with PySide6 (Qt6), not conda's Qt5
    site_dirs = site.getsitepackages()
    qt6_candidates = []
    for p in site_dirs:
        qt6_candidates += [
            os.path.join(p, "PySide6", "Qt", "plugins", "platforms"),
            os.path.join(p, "PyQt6", "Qt6", "plugins", "platforms"),
        ]
    for d in qt6_candidates:
        if glob.glob(os.path.join(d, "libq*.so")):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = d
            break

    # pick platform
    if "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "wayland" if os.environ.get("XDG_SESSION_TYPE") == "wayland" else "xcb"

    # very important: do NOT search conda's generic plugins (Qt5)
    os.environ.pop("QT_PLUGIN_PATH", None)

# ---- дальше как было ----
import numpy as np
import mne
from PySide6 import QtWidgets
from .core.data_model import DataModel
from .ui.main_window import MainWindow

def run_with_arrays(data: np.ndarray, coords2d: np.ndarray, sfreq: float, ch_names=None):
    _ensure_qt_platform()  # <-- ВАЖНО: до QApplication
    dm = DataModel(data=data, coords2d=coords2d, sfreq=sfreq, ch_names=ch_names)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(dm)
    win.resize(1100, 800)
    win.show()
    app.exec()

def run_with_mne_file(file_path_or_raw, coords2d: np.ndarray = None):
    """Load and display MNE file (GDF, EDF, etc.) or MNE Raw object with events and annotations."""
    _ensure_qt_platform()
    
    # Set MNE log level to reduce noise
    mne.set_log_level("WARNING")
    
    # Handle both file path and MNE Raw object
    if isinstance(file_path_or_raw, str):
        # Load the raw file
        print(f"Loading MNE file: {file_path_or_raw}")
        raw = mne.io.read_raw_gdf(file_path_or_raw, preload=True)
    else:
        # Already a MNE Raw object
        print("Using provided MNE Raw object")
        raw = file_path_or_raw
    
    # Create DataModel from MNE raw
    dm = DataModel.from_mne_raw(raw, coords2d)
    
    print(f"Loaded data: {dm.n_ch} channels, {dm.n_times} time points, {dm.sfreq} Hz")
    if dm.events is not None:
        print(f"Found {len(dm.events)} events")
    if dm.annotations is not None and len(dm.annotations) > 0:
        print(f"Found {len(dm.annotations)} annotations")
    
    # Create and show the application
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(dm)
    win.resize(1100, 800)
    win.show()
    app.exec()

def run_with_mne_files(file_paths: list[str], coords2d: np.ndarray = None):
    """Load and display multiple MNE files."""
    _ensure_qt_platform()
    
    # Set MNE log level to reduce noise
    mne.set_log_level("WARNING")
    
    # For now, just load the first file
    # TODO: Implement file switching functionality
    if file_paths:
        run_with_mne_file(file_paths[0], coords2d)
    else:
        raise ValueError("No file paths provided")