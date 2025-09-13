from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore
from .base_view import BaseView
from ..core.signals import SignalScaler, compute_offsets, robust_scale_from_percentile
from .channel_popup import ChannelWindow


class SignalView(BaseView):
    timeChanged = QtCore.Signal(int)

    def __init__(self, parent, data_model, window_sec: float = 10.0, min_ch_px: int = 30):
        super().__init__(parent)
        self.dm = data_model
        self.window_sec = float(window_sec)
        self.window_pts = int(self.window_sec * self.dm.sfreq)
        self.min_ch_px = int(min_ch_px)

        # ---- UI ----
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # no margins on main layout
        
        # Use GraphicsLayoutWidget for more control
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot = self.plot_widget.addPlot()
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)  # we control scroll ourselves
        self.plot.showGrid(x=True, y=False, alpha=0.2)
        self.plot.setLabel("bottom", "Time (s)")
        self.plot_widget.setBackground("w")
        
        # Remove all margins and padding
        self.plot_widget.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget.ci.layout.setSpacing(0)
        
        layout.addWidget(self.plot_widget)

        # remove extra left margin so signals start at the very left
        plot_item = self.plot
        plot_item.layout.setContentsMargins(0, 0, 0, 0)  # no margins
        plot_item.layout.setColumnSpacing(0, 0)
        plot_item.layout.setColumnFixedWidth(0, 0)  # no width for labels column


        pg.setConfigOptions(antialias=True)

        # ViewBox reference and padding
        self.vb = self.plot.vb
        self.vb.setDefaultPadding(0.0)
        self.vb.setContentsMargins(-10, 0, 0, 0)  # negative left margin

        # ---- data-dependent state ----
        self.offsets = compute_offsets(self.dm.n_ch)
        self.global_scale = robust_scale_from_percentile(
            self.dm.data, target_fraction=0.8, percentile=95.0
        )
        # Try scaling data 2 times
        self.global_scale *= 0.5
        self.scaler = SignalScaler(self.dm.n_ch)

        # Left axis ticks with channel labels
        left_axis = self.plot.getAxis("left")
        ticks = [[(float(self.offsets[i]), str(self.dm.ch_names[i] if self.dm.ch_names else i))
                  for i in range(self.dm.n_ch)]]
        left_axis.setTicks(ticks)

        # decimation and runtime state
        self.decim = max(1, int(self.dm.n_times / 3000))
        self._top_index = 0
        self._visible_count = 0
        self._cur_idx = 0

        # Build curves and cursor, then connect interactions
        self._build_curves()
        self._build_cursor()
        self._build_events_display()
        self._connect_interactions()
        
        # Initialize channel highlighting
        self.highlighted_channels = set()
        self.highlight_color = '#FFD700'  # Gold color for highlighting

        # Lock x-range width to window_sec and set limits to [0, T]
        self._apply_x_limits(self.dm.time[-1])
        self._update_xrange(0.0)
        # Force ViewBox to start at the very beginning
        self.vb.setXRange(0.0, self.window_sec, padding=0)

    # ---------- helpers: x-limits / x-range ----------
    def _apply_x_limits(self, t_total: float):
        """Lock the X range width to exactly window_sec and clamp to [0, t_total]."""
        w = float(self.window_sec)
        self.vb.setLimits(xMin=0.0, xMax=t_total, minXRange=w, maxXRange=w)

    def _update_xrange(self, center_sec: float):
        """Center the view around center_sec while keeping width = window_sec."""
        half = self.window_sec / 2.0
        t0 = max(0.0, min(center_sec - half, self.dm.time[-1] - self.window_sec))
        t1 = t0 + self.window_sec
        self.vb.setXRange(t0, t1, padding=0)

    def _update_xrange_idx(self, start: int, end: int):
        """Set X range by indices while keeping width = window_sec."""
        t0 = self.dm.time[start]
        t1 = min(self.dm.time[-1], t0 + self.window_sec)
        t0 = max(0.0, t1 - self.window_sec)
        self.vb.setXRange(t0, t1, padding=0)

    # ---------- build ----------
    def _build_curves(self):
        self.curves = []
        self._update_visible_count()
        end = min(self.window_pts, self.dm.n_times)
        x = self.dm.time[:end]
        for ch in range(self.dm.n_ch):
            y = (self.dm.data[ch, :end] * self.global_scale * self.scaler.get(ch)) + self.offsets[ch]
            c = pg.PlotCurveItem(x, y, pen=pg.mkPen(150))
            self.plot.addItem(c)
            self.curves.append(c)
            c.setVisible(self._is_channel_visible(ch))
        self._enforce_min_channel_px()

    def _build_cursor(self):
        self.cursor = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen((50, 200, 50), width=2))
        self.plot.addItem(self.cursor)
        self.cursor.setX(0.0)
        # Force cursor to start at the very beginning
        self.cursor.setPos(0.0)

    def _connect_interactions(self):
        self.cursor.sigPositionChanged.connect(self._on_cursor_drag)
        self.plot.scene().sigMouseClicked.connect(self._on_mouse_click)
        self.plot.wheelEvent = self._wheelEvent
        self.plot.resizeEvent = self._resizeEvent
        # Also connect wheel events to the plot widget
        self.plot_widget.wheelEvent = self._wheelEvent
        
        # Enable key events for keyboard shortcuts
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    # ---------- visibility / layout ----------
    def _update_visible_count(self):
        h = max(self.height(), 1)
        min_px = max(24, self.min_ch_px)  # hard floor for readability
        self._visible_count = max(1, (h - 24) // min_px)

    def _is_channel_visible(self, ch: int) -> bool:
        return self._top_index <= ch < self._top_index + self._visible_count

    def _apply_visibility(self):
        for ch, curve in enumerate(self.curves):
            curve.setVisible(self._is_channel_visible(ch))

    def _enforce_min_channel_px(self):
        """Clamp Y range to the visible channel band, no padding."""
        top = self._top_index
        bottom = min(self.dm.n_ch - 1, top + self._visible_count - 1)
        y_min = self.offsets[bottom] - 0.5
        y_max = self.offsets[top] + 0.5
        # use ViewBox to eliminate extra margins
        self.vb.setYRange(y_min, y_max, padding=0)

    # ---------- updates ----------
    def _update_signals_window(self, center_idx: int):
        half = self.window_pts // 2
        start = max(0, center_idx - half)
        end = min(self.dm.n_times, start + self.window_pts)
        start = max(0, end - self.window_pts)

        x = self.dm.time[start:end:self.decim]
        print(f"Update signals: center_idx={center_idx}, start={start}, end={end}, global_scale={self.global_scale}")  # Debug
        for ch, curve in enumerate(self.curves):
            raw_data = self.dm.data[ch, start:end:self.decim]
            scaled_data = raw_data * self.global_scale * self.scaler.get(ch)
            y = scaled_data + self.offsets[ch]
            print(f"  Ch{ch}: raw_data range=[{raw_data.min():.6f}, {raw_data.max():.6f}], scaled range=[{scaled_data.min():.6f}, {scaled_data.max():.6f}], offset={self.offsets[ch]:.3f}")  # Debug
            curve.setData(x, y)

        self._update_xrange_idx(start, end)

    # ---------- events ----------
    def _on_cursor_drag(self, line):
        t = float(line.value())
        idx = int(np.clip(round(t * self.dm.sfreq), 0, self.dm.n_times - 1))
        self._cur_idx = idx
        self._update_signals_window(idx)
        self.timeChanged.emit(idx)

    def _on_mouse_click(self, ev):
        if ev.double():
            mouse_point = self.plot.plotItem.vb.mapSceneToView(ev.scenePos())
            y = mouse_point.y()
            ch = int(np.argmin(np.abs(self.offsets - y)))
            self.scaler.toggle_boost(ch, factor=2.5)
            self._update_signals_window(self._cur_idx)
            if ev.button() == 1:
                self._open_channel_popup(ch)

    def _open_channel_popup(self, ch: int):
        t = self.dm.time
        y = self.dm.data[ch] * self.global_scale * self.scaler.get(ch)
        name = self.dm.ch_names[ch] if self.dm.ch_names else f"Ch {ch}"
        dlg = ChannelWindow(self, name, t, y)
        dlg.update_time_cursor(self._cur_idx / self.dm.sfreq)
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dlg.show()

    def _wheelEvent(self, ev):
        mods = ev.modifiers()
        steps = ev.angleDelta().y() / 120.0
        
        print(f"Wheel event: mods={mods}, steps={steps}")  # Debug info
        
        # Check if we have actual wheel movement (lower threshold for Alt/Ctrl)
        if mods == QtCore.Qt.KeyboardModifier.NoModifier and abs(steps) < 0.1:
            ev.ignore()
            return
        elif mods in [QtCore.Qt.KeyboardModifier.AltModifier, QtCore.Qt.KeyboardModifier.ControlModifier] and abs(steps) < 0.01:
            ev.ignore()
            return
            
        if mods == QtCore.Qt.KeyboardModifier.ShiftModifier:
            # Shift + wheel = horizontal scroll in time (slower, in ms)
            shift_ms = int(steps * 100)  # 100ms per step
            shift_pts = int(shift_ms * self.dm.sfreq / 1000)  # convert ms to points
            new_idx = int(np.clip(self._cur_idx - shift_pts, 0, self.dm.n_times - 1))
            self.set_time_index(new_idx)
            ev.accept()
            
        elif mods == QtCore.Qt.KeyboardModifier.ControlModifier:
            # Ctrl + wheel = change scale by multiplying
            if steps > 0:
                self.global_scale *= 1.2  # double the scale
            else:
                self.global_scale *= 0.8  # halve the scale
            self._update_signals_window(self._cur_idx)
            ev.accept()
            
        elif mods == QtCore.Qt.KeyboardModifier.AltModifier:
            # Alt + wheel = change window width (time range)
            width_change = 0.5 if steps > 0 else -0.5  # 0.5 seconds per step
            new_window_sec = max(1.0, min(60.0, self.window_sec + width_change))  # clamp between 1 and 60 seconds
            if new_window_sec != self.window_sec:
                self.window_sec = new_window_sec
                self.window_pts = int(self.window_sec * self.dm.sfreq)
                self._apply_x_limits(self.dm.time[-1])
                self._update_xrange(self._cur_idx / self.dm.sfreq)
                self._update_signals_window(self._cur_idx)
            ev.accept()
            
        elif mods == (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier):
            # Ctrl+Shift + wheel = change window width (time range) - alternative to Alt
            width_change = 0.5 if steps > 0 else -0.5  # 0.5 seconds per step
            new_window_sec = max(1.0, min(60.0, self.window_sec + width_change))  # clamp between 1 and 60 seconds
            print(f"Window change (Ctrl+Shift): {self.window_sec} -> {new_window_sec}")  # Debug
            if new_window_sec != self.window_sec:
                self.window_sec = new_window_sec
                self.window_pts = int(self.window_sec * self.dm.sfreq)
                self._apply_x_limits(self.dm.time[-1])
                self._update_xrange(self._cur_idx / self.dm.sfreq)
                self._update_signals_window(self._cur_idx)
            ev.accept()
            
        elif mods == QtCore.Qt.KeyboardModifier.NoModifier:
            # No modifier = vertical scroll (change visible channels)
            step_ch = int(np.sign(steps)) * 1
            self._top_index = int(np.clip(self._top_index - step_ch, 0, max(0, self.dm.n_ch - self._visible_count)))
            self._apply_visibility()
            self._enforce_min_channel_px()
            self._update_left_axis_ticks()
            ev.accept()
        else:
            ev.ignore()

    def _resizeEvent(self, ev):
        # keep x-window width and visible channel count consistent
        self.window_pts = int(self.window_sec * self.dm.sfreq)
        self._update_visible_count()
        self._apply_visibility()
        self._enforce_min_channel_px()
        self._apply_x_limits(self.dm.time[-1])
        self._update_xrange(self._cur_idx / self.dm.sfreq)
        # Don't call super().resizeEvent(ev) as it causes type error with GraphicsLayoutWidget

    # ---------- public API ----------
    def set_time_index(self, idx: int):
        idx = int(np.clip(idx, 0, self.dm.n_times - 1))
        self._cur_idx = idx
        t = idx / self.dm.sfreq
        self.cursor.blockSignals(True)
        self.cursor.setX(t)
        self.cursor.blockSignals(False)
        self._update_signals_window(idx)
        self.timeChanged.emit(idx)

    def on_time_index_changed(self, t_idx: int):
        self.set_time_index(t_idx)

    # ---------- Channel Highlighting ----------
    def _on_mouse_click(self, event):
        """Handle mouse click to highlight/unhighlight channels."""
        if event.button() == QtCore.Qt.LeftButton:
            # Get click position in plot coordinates
            pos = self.plot.vb.mapSceneToView(event.pos())
            x, y = pos.x(), pos.y()
            
            # Get the visible range
            vb_range = self.plot.vb.viewRange()
            y_min, y_max = vb_range[1]  # y-axis range
            
            # Calculate which channel was clicked
            # Channels are stacked vertically, with channel 0 at the top
            # Each channel takes up (y_max - y_min) / n_channels of vertical space
            channel_height = (y_max - y_min) / self.dm.n_ch
            
            # Calculate channel index (0 is at the top, so we invert)
            # Add 0.5 to center the click detection on each channel
            clicked_channel = int((y_max - y) / channel_height + 2)
            
            # Clamp to valid channel range
            clicked_channel = max(0, min(clicked_channel, self.dm.n_ch - 1))
            
            print(f"Click at y={y:.2f}, channel_height={channel_height:.2f}, clicked_channel={clicked_channel}")
            
            # Toggle highlighting for this channel
            if clicked_channel in self.highlighted_channels:
                self.highlighted_channels.remove(clicked_channel)
                print(f"Unhighlighted channel {clicked_channel}")
            else:
                self.highlighted_channels.add(clicked_channel)
                print(f"Highlighted channel {clicked_channel}")
            
            # Update the curve color
            self._update_channel_colors()
    
    def _update_channel_colors(self):
        """Update colors of all channel curves based on highlighting."""
        for ch, curve in enumerate(self.curves):
            if ch in self.highlighted_channels:
                # Highlighted channel - use gold color
                curve.setPen(pg.mkPen(self.highlight_color, width=2))
            else:
                # Normal channel - use default color
                curve.setPen(pg.mkPen('black', width=1))
    
    def clear_highlights(self):
        """Clear all channel highlights."""
        self.highlighted_channels.clear()
        self._update_channel_colors()
        print("Cleared all channel highlights")
    
    def highlight_channel(self, channel: int):
        """Programmatically highlight a specific channel."""
        if 0 <= channel < self.dm.n_ch:
            self.highlighted_channels.add(channel)
            self._update_channel_colors()
            print(f"Highlighted channel {channel}")
    
    def unhighlight_channel(self, channel: int):
        """Programmatically unhighlight a specific channel."""
        if channel in self.highlighted_channels:
            self.highlighted_channels.remove(channel)
            self._update_channel_colors()
            print(f"Unhighlighted channel {channel}")
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for channel highlighting."""
        if event.key() == QtCore.Qt.Key_C and event.modifiers() == QtCore.Qt.ControlModifier:
            # Ctrl+C: Clear all highlights
            self.clear_highlights()
        elif event.key() == QtCore.Qt.Key_H and event.modifiers() == QtCore.Qt.ControlModifier:
            # Ctrl+H: Toggle highlight for current visible channels
            visible_channels = [ch for ch in range(self.dm.n_ch) if self._is_channel_visible(ch)]
            if visible_channels:
                # If any visible channel is highlighted, unhighlight all
                if any(ch in self.highlighted_channels for ch in visible_channels):
                    for ch in visible_channels:
                        if ch in self.highlighted_channels:
                            self.highlighted_channels.remove(ch)
                else:
                    # Highlight all visible channels
                    for ch in visible_channels:
                        self.highlighted_channels.add(ch)
                self._update_channel_colors()
                print(f"Toggled highlights for visible channels: {visible_channels}")
        else:
            # Pass other key events to parent
            super().keyPressEvent(event)

    # ---------- Events and Annotations Display ----------
    def _build_events_display(self):
        """Build visual elements for events and annotations."""
        self.event_lines = []
        self.annotation_rects = []
        
        # Add event markers
        if self.dm.events is not None:
            for event in self.dm.events:
                event_time_idx, _, event_id = event
                event_time_sec = event_time_idx / self.dm.sfreq
                
                # Create vertical line for event
                line = pg.InfiniteLine(
                    pos=event_time_sec, 
                    angle=90, 
                    pen=pg.mkPen('red', width=2, style=QtCore.Qt.PenStyle.DashLine)
                )
                self.plot.addItem(line)
                self.event_lines.append(line)
        
        # Add annotation markers with different colors and labels
        if self.dm.annotations is not None and len(self.dm.annotations) > 0:
            # Define colors for different annotation types - high contrast on white background
            colors = ['#E74C3C', '#2ECC71', '#3498DB', '#9B59B6', '#E67E22', '#1ABC9C', '#34495E', '#E91E63']
            
            for i, (onset, duration, description) in enumerate(zip(
                self.dm.annotations.onset,
                self.dm.annotations.duration,
                self.dm.annotations.description
            )):
                # Choose color based on annotation type or index
                color_idx = hash(str(description)) % len(colors)
                color = colors[color_idx]
                
                # Create vertical line for annotation start
                line = pg.InfiniteLine(
                    pos=onset, 
                    angle=90, 
                    pen=pg.mkPen(color, width=3, style=QtCore.Qt.PenStyle.SolidLine)
                )
                line.setZValue(10)  # Make sure annotations are on top
                self.plot.addItem(line)
                self.annotation_rects.append(line)
                
                # Add text label for annotation
                text = pg.TextItem(
                    text=str(description),
                    color=color,
                    anchor=(0, 0.5),
                    border=pg.mkPen(color, width=1),
                    fill=pg.mkBrush('white', alpha=200)
                )
                text.setPos(onset, self.dm.n_ch * 0.8)  # Position text near top
                text.setZValue(20)  # Text on top of lines
                self.plot.addItem(text)
                self.annotation_rects.append(text)
                
                # Add duration indicator if annotation is long enough
                if duration > 1.0:  # Only for longer annotations
                    duration_text = pg.TextItem(
                        text=f"{duration:.1f}s",
                        color=color,
                        anchor=(0, 0.5),
                        border=pg.mkPen(color, width=1),
                        fill=pg.mkBrush('white', alpha=150)
                    )
                    duration_text.setPos(onset, self.dm.n_ch * 0.9)
                    duration_text.setZValue(20)
                    self.plot.addItem(duration_text)
                    self.annotation_rects.append(duration_text)

    def _update_events_display(self, center_idx: int):
        """Update events and annotations display for current time window."""
        # This could be used to show/hide events based on current view
        # For now, we keep all events visible
        pass
    
    # add inside SignalView
    def _update_left_axis_ticks(self):
        """Show ticks only for currently visible channels to avoid clutter."""
        left_axis = self.plot.getAxis("left")
        vis = range(self._top_index, min(self.dm.n_ch, self._top_index + self._visible_count))
        ticks = [[(float(self.offsets[i]), str(self.dm.ch_names[i] if self.dm.ch_names else f"Ch{i:02d}"))
                for i in vis]]
        left_axis.setTicks(ticks)
        # lock left axis column width so the plot area starts right next to it
        try:
            self.plot.layout.setColumnFixedWidth(0, 0)  # no width
        except Exception:
            pass

