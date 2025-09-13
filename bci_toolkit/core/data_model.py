from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import mne

@dataclass
class DataModel:
    """Holds time-series multichannel data and 2D sensor coordinates."""
    data: np.ndarray           # shape: (n_channels, n_times)
    coords2d: np.ndarray       # shape: (n_channels, 2)
    sfreq: float               # sampling frequency (Hz)
    ch_names: list[str] | None = None
    raw: Optional[mne.io.Raw] = None  # MNE raw object
    events: Optional[np.ndarray] = None  # MNE events array
    event_id: Optional[Dict[str, int]] = None  # MNE event_id mapping
    annotations: Optional[mne.Annotations] = None  # MNE annotations

    def __post_init__(self):
        assert self.data.ndim == 2, "data should be (n_channels, n_times)"
        assert self.coords2d.shape[0] == self.data.shape[0], "coords2d != n_channels"
        self.n_ch, self.n_times = self.data.shape
        self.time = np.arange(self.n_times, dtype=float) / float(self.sfreq)

    def values_at(self, t_idx: int) -> np.ndarray:
        """Return channel values at integer time index."""
        t_idx = int(np.clip(t_idx, 0, self.n_times - 1))
        return self.data[:, t_idx]

    @classmethod
    def from_mne_raw(cls, raw: mne.io.Raw, coords2d: Optional[np.ndarray] = None) -> 'DataModel':
        """Create DataModel from MNE Raw object."""
        # Extract data and metadata
        data = raw.get_data()  # shape: (n_channels, n_times)
        sfreq = raw.info['sfreq']
        ch_names = raw.ch_names
        
        # Get events and annotations if available
        events = None
        event_id = None
        try:
            events, event_id = mne.events_from_annotations(raw)
        except ValueError:
            # No annotations found
            pass
        
        # Get 2D coordinates if available
        if coords2d is None:
            # Try to get coordinates from raw.info
            if hasattr(raw.info, 'chs') and raw.info['chs']:
                coords2d = np.array([[ch['loc'][0], ch['loc'][1]] for ch in raw.info['chs'] 
                                   if ch['loc'] is not None and len(ch['loc']) >= 2])
                if len(coords2d) != len(ch_names):
                    # Fallback: generate random coordinates
                    coords2d = np.random.uniform(-0.1, 0.1, (len(ch_names), 2))
            else:
                # Fallback: generate random coordinates
                coords2d = np.random.uniform(-0.1, 0.1, (len(ch_names), 2))
        
        return cls(
            data=data,
            coords2d=coords2d,
            sfreq=sfreq,
            ch_names=ch_names,
            raw=raw,
            events=events,
            event_id=event_id,
            annotations=raw.annotations
        )

    def get_events_at_time(self, t_idx: int, window_pts: int = 1) -> List[Dict[str, Any]]:
        """Get events that occur at or near the given time index."""
        if self.events is None:
            return []
        
        t_sec = t_idx / self.sfreq
        window_sec = window_pts / self.sfreq
        
        # Find events within the time window
        event_times = self.events[:, 0] / self.sfreq
        mask = (event_times >= t_sec - window_sec/2) & (event_times <= t_sec + window_sec/2)
        nearby_events = self.events[mask]
        
        result = []
        for event in nearby_events:
            event_time_idx, _, event_id = event
            event_name = None
            if self.event_id:
                event_name = next((name for name, eid in self.event_id.items() if eid == event_id), None)
            
            result.append({
                'time_idx': int(event_time_idx),
                'time_sec': event_time_idx / self.sfreq,
                'event_id': int(event_id),
                'event_name': event_name
            })
        
        return result

    def get_annotations_at_time(self, t_idx: int, window_pts: int = 1) -> List[Dict[str, Any]]:
        """Get annotations that occur at or near the given time index."""
        if self.annotations is None or len(self.annotations) == 0:
            return []
        
        t_sec = t_idx / self.sfreq
        window_sec = window_pts / self.sfreq
        
        result = []
        for i, (onset, duration, description) in enumerate(zip(
            self.annotations.onset, 
            self.annotations.duration, 
            self.annotations.description
        )):
            # Check if annotation overlaps with time window
            if (onset <= t_sec + window_sec/2 and 
                onset + duration >= t_sec - window_sec/2):
                result.append({
                    'onset_sec': float(onset),
                    'onset_idx': int(onset * self.sfreq),
                    'duration_sec': float(duration),
                    'duration_idx': int(duration * self.sfreq),
                    'description': str(description)
                })
        
        return result
