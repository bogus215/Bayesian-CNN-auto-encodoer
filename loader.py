import os
import sys
import glob
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset
from random import randint

sys.path.append('./')

class DefoggingDataset(Dataset):

    def __init__(self, args):

        self.root_dir = args.data_dir
        self.size = args.size
        self.matchup = args.matchup
        self.use_count = args.use_count

        pattern = os.path.join(self.root_dir, self.matchup, f"**/{self.size}x{self.size}/*.npz")
        self.samples = glob.glob(pattern, recursive=True)
        self.replay_names = [pathlib.Path(p).parent.parent.name for p in self.samples]

    def __getitem__(self, idx):

        npz_file = self.samples[idx]
        with np.load(npz_file) as f:
            inp = torch.from_numpy(f['input']).float()
            tar = torch.from_numpy(f['target']).float()

        inp = torch.rot90(inp , randint(1,3) , [1,2])
        tar = torch.rot90(tar , randint(1,3) , [1,2])

        out = dict(input=inp, target=tar)

        if self.use_count:
            return out
        else:
            return {k: torch.clamp_max(v, max=1.) for k, v in out.items()}


    def __len__(self):
        return len(self.samples)
