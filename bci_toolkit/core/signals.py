from __future__ import annotations
import numpy as np

class SignalScaler:
    """Per-channel visual scaling (not altering underlying data)."""
    def __init__(self, n_channels: int):
        self._scale = np.ones(n_channels, dtype=float)

    def toggle_boost(self, ch: int, factor: float = 2.0):
        """Toggle between 1.0 and boosted factor for a channel."""
        self._scale[ch] = 1.0 if np.isclose(self._scale[ch], factor) else factor

    def get(self, ch: int) -> float:
        return float(self._scale[ch])

def compute_offsets(n_channels: int) -> np.ndarray:
    return np.linspace(n_channels - 1, 0, n_channels, dtype=float)

def robust_scale_from_percentile(data: np.ndarray, target_fraction: float = 0.8, percentile: float = 95.0) -> float:
    """Return a global scale so that median 0..P95 amplitude fits into target_fraction of channel spacing.
    Channel spacing assumed to be 1.0 in Y-units."""
    # per-channel robust amplitude (0..P95) around zero
    amp = np.percentile(np.abs(data), percentile, axis=1)  # (n_ch,)
    m = float(np.median(amp))
    m = max(m, 1e-12)
    # We want ±m to fit into target_fraction/2 of spacing (spacing==1.0)
    return (target_fraction / 2.0) / m