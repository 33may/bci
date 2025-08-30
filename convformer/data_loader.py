from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal, Any

from glob import glob
import mne
from loguru import logger as log

@dataclass
class PreprocConfig:
    montage: str = "standard_1020"
    l_freq: float = 0.5
    h_freq: float = 40.0
    notch: Optional[float] = 50.0  # None to skip
    resample_hz: Optional[int] = 250  # None to skip
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: Optional[Tuple[float, float]] = (None, 0.0)  # (start, end)
    event_id_map: Dict[str, int] = None  # e.g., {"left": 7, "right": 8, "foot": 9, "tongue": 10}
    picks: Optional[List[str]] = None     # None = all EEG
    reject_by_annotation: bool = True     # drop segments with bad annotations
    ica_n_components: Optional[int] = None  # None to skip ICA
    ica_max_iter: int = 512

@dataclass
class SaveConfig:
    out_dir: str
    format: Literal["npz", "pt"] = "npz"  # "npz" for NumPy, "pt" for torch.save
    normalize: Literal["none", "per_channel_z", "per_epoch_z"] = "per_channel_z"
    dtype: Literal["float32", "float64"] = "float32"
    one_hot: bool = False

# -----------------------------
# 2) Data loading & raw prep
# -----------------------------


class BCILoader:
    def __init__(self, config: PreprocConfig, data_root: str):
        self.config = config
        self.data_path = data_root

        self.files = self.find_files(data_path)

        log.debug(f"=== Files Loaded: ===== \n {self.files}")

        self.file_map, self.labels = self.separate_test_eval_files_map()

        log.debug(f"=== Train/Eval Separated ===== \n Train: {self.labels['counts']['train']} Eval: {self.labels['counts']['eval']}")

        all_data = [self.load_raw(file_path) for file_path in self.files]
        self.raw_files, self.events, self.event_ids = zip(*all_data)

        self.raw_files = list(self.raw_files)
        self.events = list(self.events)
        self.event_ids = list(self.event_ids)

        log.debug(f"=== Raw files Loaded: ===== \n {self.raw_files}")

        self.rename_channels()

        log.debug(f"=== Channesls renameds =====")


    def find_files(self, data_root: str, pattern: str = "*.gdf") -> List[str]:
        files = glob(f"{data_path}/{pattern}")
        return files

    def separate_test_eval_files_map(self) -> Tuple[List[bool], Dict]:

        file_map = [("E" in f) for f in self.files]

        n_eval = sum(file_map)
        n_train = len(file_map) - n_eval

        labels = {
            1: "eval",
            0: "train",
            "counts": {
                "eval": n_eval,
                "train": n_train
            }
        }
        
        return file_map, labels
        


    def load_raw(self, file_path: str) -> Tuple:
        """
        Load a single EEG recording (e.g., with mne.io.read_raw_*).
        Return an MNE Raw-like object.
        """
        raw = mne.io.read_raw_gdf( file_path, preload=True)

        events, event_id = mne.events_from_annotations(raw)

        return raw, events, event_id
    
    # def apply_band_pass_filter(self, ):

    def rename_channels(self):
        rename_map = {
        'EEG-Fz': 'Fz',
        'EEG-0': 'FC3',
        'EEG-1': 'FC1',
        'EEG-2': 'FCz',
        'EEG-3': 'FC2',
        'EEG-4': 'FC4',
        'EEG-5': 'C5',
        'EEG-C3': 'C3',
        'EEG-6': 'C1',
        'EEG-Cz': 'Cz',
        'EEG-7': 'C2',
        'EEG-C4': 'C4',
        'EEG-8': 'C6',
        'EEG-9': 'CP3',
        'EEG-10': 'CP1',
        'EEG-11': 'CPz',
        'EEG-12': 'CP2',
        'EEG-13': 'CP4',
        'EEG-14': 'P1',
        'EEG-Pz': 'Pz',
        'EEG-15': 'P2',
        'EEG-16': 'POz',
        'EOG-left': 'EOG-left',
        'EOG-central': 'EOG-central',
        'EOG-right': 'EOG-right'
        }

        types = {'EOG-left': 'eog', 'EOG-right': 'eog', 'EOG-central': 'eog'}

        for raw in self.raw_files:
            raw.rename_channels(rename_map)

            raw.set_montage("standard_1020", match_case=False, on_missing="ignore")

            raw.set_channel_types(types)

config = PreprocConfig()

data_path = "data/BCICIV_2a_gdf"

loader = BCILoader(config, data_path)

