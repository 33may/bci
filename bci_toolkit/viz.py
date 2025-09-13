import numpy as np
from scipy.interpolate import griddata
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg


class TopomapSignalViewer(QtWidgets.QWidget):
    def __init__(self, coords, data, sfreq=250.0, window_sec=10.0):
        super().__init__()
        self.coords = coords
        self.data = data
        self.sfreq = float(sfreq)
        self.n_ch, self.n_times = data.shape
        self.window_sec = float(window_sec)
        self.window_pts = int(self.window_sec * self.sfreq)

        # --- Build UI ---
        self.setWindowTitle("Topomap + Signals (single time controller)")
        layout = QtWidgets.QVBoxLayout(self)

        # Top: topomap view
        self.topomap_view = pg.GraphicsLayoutWidget()
        self.topomap_vb = self.topomap_view.addViewBox(lockAspect=True)
        self.topomap_img = pg.ImageItem()
        self.topomap_vb.addItem(self.topomap_img)
        self.topomap_img.setLookupTable(pg.colormap.get("CET-CBD1").getLookupTable(nPts=512))
        layout.addWidget(self.topomap_view, stretch=2)

        # Bottom: signals view
        self.sig_plot = pg.PlotWidget()
        self.sig_plot.setMenuEnabled(False)
        self.sig_plot.setMouseEnabled(x=True, y=False)
        self.sig_plot.showGrid(x=True, y=False, alpha=0.2)
        self.sig_plot.setLabel("bottom", "Time (s)")
        layout.addWidget(self.sig_plot, stretch=3)

        # Slider (делаем его толще и с метками)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.n_times - 1)
        self.slider.setValue(0)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(int(self.sfreq))  # примерно 1 сек
        self.slider.setFixedHeight(30)  # хорошо видно
        layout.addWidget(self.slider)

        # --- Prepare signals plot ---
        self.time = np.arange(self.n_times) / self.sfreq
        self.offsets = np.linspace(self.n_ch - 1, 0, self.n_ch)
        self.offset_scale = 1.0 / max(np.std(self.data, axis=1).mean(), 1e-12)
        self.decim = max(1, int(self.n_times / 3000))
        self._build_signals()

        # Cursor line
        self.cursor = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen((50, 200, 50), width=2))
        self.sig_plot.addItem(self.cursor)
        self.cursor.setX(0.0)

        # Topomap grid
        self.grid_res = 120
        lim = 0.11
        self.gx = np.linspace(-lim, lim, self.grid_res)
        self.gy = np.linspace(-lim, lim, self.grid_res)
        self.GX, self.GY = np.meshgrid(self.gx, self.gy)
        R = np.sqrt(self.GX**2 + self.GY**2)
        self.mask = (R <= lim * 0.98)
        self.topomap_vb.setRange(xRange=(-lim, lim), yRange=(-lim, lim), padding=0)

        # --- Connect events ---
        self.slider.valueChanged.connect(self._on_slider)
        self.cursor.sigPositionChanged.connect(self._on_cursor_drag)

        # enable wheel scroll for horizontal shift
        self.sig_plot.wheelEvent = self._wheelEvent

        self._update_all(0)

    # ---------- Signals ----------
    def _build_signals(self):
        self.curves = []
        end = min(self.window_pts, self.n_times)
        x = self.time[:end]
        for ch in range(self.n_ch):
            y = self.data[ch, :end] * self.offset_scale + self.offsets[ch]
            c = pg.PlotCurveItem(x, y, pen=pg.mkPen(120))
            self.sig_plot.addItem(c)
            self.curves.append(c)
        self._update_xrange(0)

    def _update_signals_window(self, center_idx):
        half = self.window_pts // 2
        start = max(0, center_idx - half)
        end = min(self.n_times, start + self.window_pts)
        start = max(0, end - self.window_pts)

        x = self.time[start:end:self.decim]
        for ch in range(self.n_ch):
            y = (self.data[ch, start:end:self.decim] * self.offset_scale) + self.offsets[ch]
            self.curves[ch].setData(x, y)
        self._update_xrange_idx(start, end)

    def _update_xrange(self, center_sec):
        half = self.window_sec / 2.0
        self.sig_plot.setXRange(center_sec - half, center_sec + half, padding=0)

    def _update_xrange_idx(self, start, end):
        self.sig_plot.setXRange(self.time[start], self.time[end - 1], padding=0)

    # ---------- Topomap ----------
    def _update_topomap(self, idx):
        vals = self.data[:, idx]
        Z = griddata(self.coords, vals, (self.GX, self.GY), method="cubic", fill_value=0.0)
        Z = np.where(self.mask, Z, np.nan)
        self.topomap_img.setImage(Z.T, autoLevels=True)
        self.topomap_img.setRect(pg.QtCore.QRectF(self.gx[0], self.gy[0],
                                                  self.gx[-1]-self.gx[0], self.gy[-1]-self.gy[0]))

    # ---------- Events ----------
    def _on_slider(self, idx):
        t = float(idx) / self.sfreq
        self.cursor.blockSignals(True)
        self.cursor.setX(t)
        self.cursor.blockSignals(False)
        self._update_signals_window(idx)
        self._update_topomap(idx)

    def _on_cursor_drag(self, line):
        t = float(line.value())
        idx = int(np.clip(round(t * self.sfreq), 0, self.n_times - 1))
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self._update_signals_window(idx)
        self._update_topomap(idx)

    def _update_all(self, idx):
        self._update_signals_window(idx)
        self._update_topomap(idx)

    # ---------- Mouse wheel horizontal scroll ----------
    def _wheelEvent(self, ev):
        # delta < 0 → scroll right, delta > 0 → left
        steps = ev.angleDelta().y() / 120  # 1 step = 120
        shift = int(steps * self.window_pts // 5)
        cur_idx = self.slider.value()
        new_idx = np.clip(cur_idx - shift, 0, self.n_times - 1)
        self.slider.setValue(new_idx)
        ev.accept()


def demo():
    n_ch = 32
    n_times = 60 * 250
    sfreq = 250.0

    rng = np.random.default_rng(42)
    coords = rng.uniform(-0.08, 0.08, size=(n_ch, 2))

    t = np.arange(n_times) / sfreq
    data = 1e-6 * (0.5 * np.sin(2*np.pi*10*t)[None, :]
                   + 0.3 * np.sin(2*np.pi*22*t + rng.uniform(0, 2*np.pi, (n_ch, 1)))
                   + 0.05 * rng.standard_normal((n_ch, n_times)))

    app = QtWidgets.QApplication([])
    w = TopomapSignalViewer(coords, data, sfreq=sfreq, window_sec=10.0)
    w.resize(900, 800)
    w.show()
    app.exec()


if __name__ == "__main__":
    demo()
