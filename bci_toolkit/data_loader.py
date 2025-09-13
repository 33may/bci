"""Data loading utilities for BCI Toolkit."""

import mne
import re
from typing import Tuple

# Standard 10-20 electrode positions
VALID_1020 = [
    # Frontal / Prefrontal
    'Fp1','Fpz','Fp2','AF7','AF3','AFz','AF4','AF8',
    'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    # Frontocentral / Central
    'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
    # Temporo-parietal / Parietal
    'T9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','T10',
    'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
    # Parieto-occipital / Occipital
    'PO7','PO3','POz','PO4','PO8','O1','Oz','O2','Iz'
]

CANON = {name.lower(): name for name in VALID_1020}

def normalize_ch_name(name: str) -> str:
    """Normalize channel name to standard 10-20 format."""
    base = re.sub(r"\.", "", name)
    key = base.lower()
    return CANON.get(key, base)

def normalize_and_set_montage(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Normalize channel names and set standard 10-20 montage."""
    mapping = {old: normalize_ch_name(old) for old in raw.ch_names}
    raw.rename_channels(mapping)

    ch_types = {}
    for ch in raw.ch_names:
        low = ch.lower()
        if 'sti' in low:
            ch_types[ch] = 'stim'
        elif low.startswith('eog') or low in ('eogl','eogr','veog','heog'):
            ch_types[ch] = 'eog'
    
    if ch_types:
        raw.set_channel_types(ch_types)

    raw.set_montage('standard_1020')
    return raw

def load_bci(subject: int = 1, runs: list = [6, 10]) -> Tuple[mne.io.Raw, mne.io.Raw]:
    """Load BCI data for a subject and specific runs."""
    rawfiles = mne.datasets.eegbci.load_data(subject, runs)
    
    raw_l = mne.io.read_raw_edf(rawfiles[0], preload=True)
    raw_r = mne.io.read_raw_edf(rawfiles[1], preload=True)
    
    raw_l = normalize_and_set_montage(raw_l)
    raw_r = normalize_and_set_montage(raw_r)
    
    return raw_l, raw_r