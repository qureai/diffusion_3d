import ast
import os
from collections import Counter

import lightning as L
import pandas as pd
import torch
from clearml import Logger
from hydra.utils import instantiate
from monai.data.dataloader import DataLoader
from monai.data.dataset import PersistentDataset
from monai.transforms import Compose, LoadImaged
from safetensors import safe_open
from sklearn.model_selection import train_test_split
from torch.distributed import get_rank
from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm
from vision_architectures.image_readers.safetensors_reader import SafetensorsReader


class CTRATEDataModule(L.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.save_hyperparameters()

        self.config = data_config

    def create_dataloader(self, run_type):
        dataset = CTRATEDataset(self.config, run_type)

        sampler = None
        if run_type == "train":
            batch_size = self.config.train_batch_size

            if self.config.train_sample_size > 0:
                weights = dataset.get_weights_for_sampling(self.config.sample_balance_cols)
                sampler = WeightedRandomSampler(weights, self.config.train_sample_size, replacement=True)
        elif run_type == "valid":
            batch_size = self.config.val_batch_size
        elif run_type == "test":
            batch_size = 1

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
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

    def test_dataloader(self):
        return self.create_dataloader("test")


class CTRATEDataset(PersistentDataset):
    def __init__(self, config, run_type):
        self.config = config
        self.run_type = run_type

        # Finalize transforms
        self.load_image_transform = LoadImaged(keys=["image"], reader=SafetensorsReader(), dtype=torch.float32)

        if run_type == "train":
            self.transforms = Compose([self.load_image_transform, instantiate(self.config.train_augmentations)])
            cache_dir = None  # no point caching as most is random
        elif run_type == "valid":
            self.transforms = Compose([self.load_image_transform, instantiate(self.config.val_augmentations)])
            # cache_dir = f"/raid/arjun/training_cache/ct_rate/valid"  # raid10
            cache_dir = None
        elif run_type == "test":
            self.transforms = Compose([self.load_image_transform, instantiate(self.config.test_augmentations)])
            cache_dir = None
        self.transforms = self.transforms.flatten()

        # Load all data, perform checks, filter, and finalize
        self.load_csv()
        self.sample_dataset()
        self.perform_checks()
        self.finalize_data()

        print(f"No. of {run_type} datapoints: {len(self.data)}")

        super().__init__(
            data=self.data,
            transform=self.transforms,
            cache_dir=cache_dir,
        )

        # ClearML Log
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

        self.dataset["SliceThickness"] = self.dataset.Spacing.str[0]

    def sample_dataset(self):
        study_uids = self.dataset.StudyUID.drop_duplicates()
        train_study_uids, valid_study_uids = train_test_split(
            study_uids,
            # train_size=0.95,
            test_size=500,
            random_state=42,
        )
        train = self.dataset[self.dataset.StudyUID.isin(train_study_uids)]
        valid = self.dataset[self.dataset.StudyUID.isin(valid_study_uids)]

        if self.run_type == "train":
            self.dataset = train
        elif self.run_type == "valid" or self.run_type == "test":
            # valid = valid.sort_values("SliceThickness")
            valid = valid.drop_duplicates(subset=["StudyUID"], keep="first")
            self.dataset = valid
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

            self.dataset.loc[i, "image"] = filepath

        self.dataset = self.dataset.dropna(subset=["image"])

    def finalize_data(self):
        self.data = self.dataset.to_dict(orient="records")

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
    from diffusion_3d.chestct.autoencoder.vae.config import get_config

    cfg = get_config()

    dataset = CTRATEDataset(cfg.data, "train")
    print(dataset[0][0]["image"].shape)

    dataset = CTRATEDataset(cfg.data, "valid")
    print(dataset[0][0]["image"].shape)

    dataset = CTRATEDataset(cfg.data, "test")
    print(dataset[0]["image"].shape)
