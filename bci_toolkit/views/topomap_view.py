from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from scipy.interpolate import griddata
from PySide6 import QtWidgets
from .base_view import BaseView

class TopomapView(BaseView):
    """Interpolated heatmap over head plane with a fixed outline."""
    def __init__(self, parent, data_model, grid_res: int = 120, lim: float = 0.11):
        super().__init__(parent)
        self.dm = data_model
        self.grid_res = int(grid_res)
        self.lim = float(lim)

        layout = QtWidgets.QVBoxLayout(self)
        self.canvas = pg.GraphicsLayoutWidget()
        self.vb = self.canvas.addViewBox(lockAspect=True)
        self.img = pg.ImageItem()
        self.vb.addItem(self.img)
        self.img.setLookupTable(pg.colormap.get("CET-CBD1").getLookupTable(nPts=512))
        self.vb.setRange(xRange=(-self.lim, self.lim), yRange=(-self.lim, self.lim), padding=0)
        layout.addWidget(self.canvas)

        self._prepare_grid()
        self.on_time_index_changed(0)

    def _prepare_grid(self):
        self.gx = np.linspace(-self.lim, self.lim, self.grid_res)
        self.gy = np.linspace(-self.lim, self.lim, self.grid_res)
        self.GX, self.GY = np.meshgrid(self.gx, self.gy)
        R = np.sqrt(self.GX**2 + self.GY**2)
        self.mask = (R <= self.lim * 0.98)

    def _update_image(self, vals: np.ndarray):
        Z = griddata(self.dm.coords2d, vals, (self.GX, self.GY), method="cubic", fill_value=0.0)
        Z = np.where(self.mask, Z, np.nan)
        self.img.setImage(Z.T, autoLevels=True)
        self.img.setRect(pg.QtCore.QRectF(self.gx[0], self.gy[0], self.gx[-1]-self.gx[0], self.gy[-1]-self.gy[0]))

    def on_time_index_changed(self, t_idx: int):
        vals = self.dm.values_at(t_idx)
        self._update_image(vals)