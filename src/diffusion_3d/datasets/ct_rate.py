import ast
import os
import random
from collections import Counter

import lightning as L
import pandas as pd
import torch
from clearml import Logger
from ct_pretraining import augmentations as all_augmentations
from monai.data.dataloader import DataLoader
from monai.data.dataset import CacheDataset
from safetensors import safe_open
from sklearn.model_selection import train_test_split
from torch.distributed import get_rank
from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm


class CTRATEDataModule(L.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.save_hyperparameters()

        self.config = data_config

    def create_dataloader(self, run_type):
        dataset = CTRATEDataset(self.config, run_type)

        sampler = None
        if run_type == "train" and self.config.train_sample_size > 0:
            weights = dataset.get_weights_for_sampling(self.config.sample_balance_cols)
            sampler = WeightedRandomSampler(weights, self.config.train_sample_size, replacement=True)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
            persistent_workers=True,
        )
        return dataloader

    def train_dataloader(self):
        return self.create_dataloader("train")

    def val_dataloader(self):
        return self.create_dataloader("valid")


class CTRATEDataset(CacheDataset):
    def __init__(self, config, run_type):
        self.config = config
        self.run_type = run_type

        self.load_csv()
        if self.run_type != "custom":
            self.sample_dataset()
            self.perform_checks()

        print(f"No. of {run_type} datapoints: {len(self)}")

        self.tables = [
            self.dataset[cols].value_counts()
            for cols in [
                ["Modality"],
                ["BodyPart"],
                ["Source"],
            ]
        ]

        logger = Logger.current_logger()
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if logger is not None and local_rank == 0:
            for table in self.tables:
                logger.report_table(
                    f"{run_type} - global rank {get_rank()} - {table.index.names}",
                    "PD with index",
                    0,
                    table_plot=table.to_frame(),
                )

    def load_csv(self):
        self.dataset = pd.read_csv(self.config.csvpath, low_memory=False)
        self.dataset = self.dataset[self.dataset.Source == "ct_rate"]

        self.dataset.Shape = self.dataset.Shape.apply(ast.literal_eval)
        self.dataset.Spacing = self.dataset.Spacing.apply(ast.literal_eval)

    def sample_dataset(self):
        study_uids = self.dataset.StudyUID.drop_duplicates()
        train_study_uids, valid_study_uids = train_test_split(
            study_uids,
            train_size=0.9,
            random_state=42,
        )
        train = self.dataset[self.dataset.StudyUID.isin(train_study_uids)]
        valid = self.dataset[self.dataset.StudyUID.isin(valid_study_uids)]

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

        if self.config.limited_dataset_size is not None:
            self.dataset = self.dataset.sample(self.config.limited_dataset_size)

        self.dataset = self.dataset.reset_index(drop=True)

    def perform_checks(self):
        for i in tqdm(self.dataset.index, self.run_type):
            # uid = self.dataset.loc[i, "SeriesUID"]

            shape = self.dataset.loc[i, "Shape"]
            spacing = self.dataset.loc[i, "Spacing"]

            if not shape or not spacing:
                continue

            if len(shape) != 3 or len(spacing) != 3:
                continue

            allowed = True
            for spacing_val, target_val in zip(spacing, self.config.allowed_spacings):
                if target_val[0] != -1 and spacing_val < target_val[0]:
                    allowed = False
                    break
                if target_val[1] != -1 and spacing_val > target_val[1]:
                    allowed = False
                    break

            if not allowed:
                continue

            for shape_val, target_val in zip(shape, self.config.allowed_shapes):
                if target_val[0] != -1 and shape_val < target_val[0]:
                    allowed = False
                    break
                if target_val[1] != -1 and shape_val > target_val[1]:
                    allowed = False
                    break

            if not allowed:
                continue

            filepath = os.path.join(self.config.datapath, self.dataset.loc[i, "FilePath"])
            if not os.path.exists(filepath):
                continue

            self.dataset.loc[i, "__filepath__"] = filepath

        self.dataset = self.dataset.dropna(subset=["__filepath__"])

    def augment(self, scan, spacing, i, augmentations):
        other = self.dataset.loc[i]

        for augmentation in augmentations:
            if isinstance(augmentation, list):
                probabilities = augmentation[0]
                chosen_path_index = random.choices(
                    population=list(range(1, len(probabilities) + 1)), weights=probabilities, k=1
                )[0]
                chosen_path = augmentation[chosen_path_index]
                scan, spacing = self.augment(scan, spacing, i, chosen_path)
            else:
                augmentation_fn = getattr(all_augmentations, augmentation.__fn_name__)
                scan, spacing = augmentation_fn(scan, spacing, augmentation, other)

        return scan, spacing

    def load_scan(self, filepath):
        with safe_open(filepath, framework="pt") as f:
            scan: torch.Tensor = f.get_tensor("images")
            spacing: torch.Tensor = f.get_tensor("spacing")

        scan = scan.type(torch.float32)
        spacing = spacing.type(torch.float16)

        return scan, spacing

    def __getitem__(self, i):
        i = self.dataset.index[i]

        uid = self.dataset.loc[i, "SeriesUID"]
        filepath = self.dataset.loc[i, "__filepath__"]
        scan, spacing = self.load_scan(filepath)

        if self.run_type == "train":
            scan, spacing = self.augment(scan, spacing, i, self.config.train_augmentations)
        elif self.run_type == "valid":
            scan, spacing = self.augment(scan, spacing, i, self.config.val_augmentations)

        scan = scan.type(torch.float32).unsqueeze(0)
        # scan: torch.Tensor = scan.nan_to_num()
        # assert not scan.isnan().any().item(), i

        if self.run_type == "custom":
            return scan, spacing, self.dataset.loc[i, "SeriesUID"]

        return {
            "scan": scan,
            "spacing": spacing,
            "uid": uid,
            "index": i,
        }

    def __len__(self):
        return len(self.dataset)

    def get_weights_for_sampling(self, colnames):
        weights = []
        counters = {colname: Counter() for colname in colnames}

        for i in tqdm(self.dataset.index, "Calculating sample weights"):
            for colname in colnames:
                counters[colname][self.dataset.loc[i, colname]] += 1

        for i in self.dataset.index:
            weight = 1
            for colname in colnames:
                weight *= 1 / counters[colname][self.dataset.loc[i, colname]]

            weights.append(weight)

        return weights


if __name__ == "__main__":
    from diffusion_3d.chestct.autoencoder.config import get_config

    cfg = get_config()

    dataset = CTRATEDataset(cfg.data, "train")
    print(dataset[0]["scan"].shape)

    dataset = CTRATEDataset(cfg.data, "valid")
    print(dataset[0]["scan"].shape)
