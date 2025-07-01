import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

from utils import *


default_label_mapping = {
    "bonafide": 0,
    "replay_bonafide": 1,
    "fake": 2,
    "replay_fake": 3,
}


class HuggingFaceAudioDataset(Dataset):
    def __init__(
        self, hf_dataset, max_len=64000, pad_mode="random", label_mapping=None
    ):
        self.dataset = hf_dataset
        self.max_len = max_len
        if pad_mode == "random":
            self.pad = pad_random
        else:
            self.pad = pad

        self.label_mapping = label_mapping or default_label_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item["path"]
        waveform = self.pad(audio["array"], max_len=self.max_len)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        label = torch.tensor(self.label_mapping[item["label"]], dtype=torch.long)
        return waveform, label


class EchoFakeModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_path,
        batch_size=8,
        sample_rate=16000,
        num_workers=4,
        max_len=64000,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.num_workers = num_workers
        self.max_len = max_len

    def setup(self, stage=None):
        dataset = load_from_disk(self.dataset_path)

        self.train_dataset = HuggingFaceAudioDataset(dataset["train"], self.max_len)
        self.val_dataset = HuggingFaceAudioDataset(dataset["dev"], self.max_len)
        self.test_dataset = HuggingFaceAudioDataset(
            dataset["closed_set_eval"], self.max_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
