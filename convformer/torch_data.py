import torch, os, glob
from torch.utils.data import Dataset, DataLoader

class EEGEpochDataset(Dataset):
    """Dataset over exported .pt files (each file = one recording)."""
    def __init__(self, root, split="train"):
        self.files = []
        for f in glob.glob(os.path.join(root, "*.pt")):
            d = torch.load(f, map_location="cpu")
            if split == "all" or d["split"] == split:
                self.files.append(f)

        # flatten into list of (file, local_idx)
        self.index = []
        for f in self.files:
            d = torch.load(f, map_location="cpu")
            n = d["X"].shape[0]
            for i in range(n):
                self.index.append((f, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f, i = self.index[idx]
        d = torch.load(f, map_location="cpu")
        X = d["X"][i]     # (C, T)
        y = d["y"][i]     # scalar
        return X, y
