import os

import lightning as L
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm


class ChestCTDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

    def setup_dataloader(self, run_type):
        dataset = ChestCTDataset(self.config, run_type)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size if run_type == "train" else self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return dataloader

    def train_dataloader(self):
        return self.setup_dataloader("train")

    def val_dataloader(self):
        return self.setup_dataloader("valid")


class ChestCTDataset(Dataset):
    def __init__(self, config, run_type):
        self.config = config
        self.run_type = run_type

        self.load_dataset()
        if self.run_type != "custom":
            self.sample_dataset()
            self.perform_checks()

    def load_dataset(self):
        self.dataset = pd.read_csv(self.config.csvpath, low_memory=False)

    def sample_dataset(self):
        train, valid = train_test_split(
            self.dataset,
            train_size=0.8,
            random_state=42,
        )

        if self.run_type == "train":
            self.dataset = train
        elif self.run_type == "valid":
            self.dataset = valid
        elif self.run_type == "custom":
            pass
        elif self.run_type == "all":
            pass
        else:
            raise NotImplementedError

        # if self.config.fast_dev_run:
        #     self.dataset = self.dataset.sample(128, random_state=42)

        self.dataset = self.dataset.reset_index(drop=True)

    def perform_checks(self):
        for i in tqdm(self.dataset.index, self.run_type):
            uid = self.dataset.loc[i, "SeriesUID"]
            filepath = os.path.join(self.config.datapath, rf"{uid}.safetensors")
            if not os.path.exists(filepath):
                continue

            self.dataset.loc[i, "filepath"] = filepath

        self.dataset = self.dataset.dropna(subset=["filepath"])

    def augment(self, scan, gt):
        return scan, gt

    def reshape(self, scan):
        shape = scan.shape[-3:]
        desired = (288, 256, 384)

        to_pad = []
        for s, d in zip(shape, desired):
            if s == d:
                to_pad.append(0)
                to_pad.append(0)
                continue

            l = np.random.randint(0, d - s)
            r = d - s - l
            to_pad.append(l)
            to_pad.append(r)

        scan = torch.nn.functional.pad(scan, to_pad[::-1], "constant", 0)

        return scan

    def __getitem__(self, i):
        i = self.dataset.index[i]

        filepath = self.dataset.loc[i, "filepath"]
        if filepath.endswith(".safetensors"):
            with safe_open(filepath, framework="pt") as f:
                scan = f.get_tensor("resampled")
        else:
            raise ValueError

        scan = self.reshape(scan)

        gt = torch.tensor(-1).type(torch.float32)

        if self.config.augment and self.run_type == "train":
            scan, gt = self.augment(scan, gt)

        if self.run_type == "custom":
            return scan, gt, self.dataset.loc[i, "SeriesUID"]

        return scan, gt

    def __len__(self):
        return len(self.dataset)
