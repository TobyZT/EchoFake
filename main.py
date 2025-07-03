"""
Training stage 2 - train classifier for audio deepfake detection

"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
from pathlib import Path
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
import json

from train import RawNet2Trainer, AASISTTrainer
from data import EchoFakeModule, ASVspoof2019

CUDA_VISIBLE_DEVICES = [1]


def get_trainer(model_name):
    if model_name.lower() == "rawnet2":
        return RawNet2Trainer
    elif model_name.lower() == "aasist":
        return AASISTTrainer


def get_dataset(dataset_name, config):
    with open("configs/datasets.json", "r") as f:
        datasets_config = json.load(f)

    config["num_classes"] = datasets_config["num_classes"]

    args = {
        "num_workers": config["train"]["num_workers"],
        "max_len": config["train"]["max_len"],
        "batch_size": config["train"]["batch_size"],
        "label_mapping": datasets_config["label_mapping"],
    }

    if dataset_name.lower() == "echofake":
        return EchoFakeModule(datasets_config["echofake"], **args)
    if dataset_name.lower() == "asvspoof2019":
        config["num_classes"] = 2
        return ASVspoof2019(datasets_config["asvspoof2019"], **args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="RawNet2")
    parser.add_argument("--trainset", type=str, default="echofake")
    parser.add_argument("--evalset", type=str, default="echofake")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    assert args.model.lower() in ["rawnet2", "aasist"], "Not supported model."

    # read config
    config_name = f"configs/{args.model.lower()}.json"
    with open(config_name, "r") as f:
        config = json.load(f)
    exp_name = config["train"]["exp_name"]
    seed = config["train"]["seed"]
    num_epochs = config["train"]["num_epochs"]

    check_val_every_n_epoch = config["train"]["check_val_every_n_epoch"]
    save_top_k = config["train"]["save_top_k"]
    # set seed
    L.seed_everything(seed=seed)

    # set logger
    version = "eval" if args.eval else None
    logger = pl_loggers.TensorBoardLogger(
        save_dir="exp", name=exp_name, version=version
    )
    logger.log_hyperparams(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/eer",
        dirpath=logger.log_dir + "/checkpoints",
        filename="best",
        save_top_k=save_top_k,
        save_last=True,
        mode="min",
        every_n_epochs=1,
    )

    # set devices
    trainer = L.Trainer(
        devices=CUDA_VISIBLE_DEVICES,
        accelerator="gpu",
        max_epochs=num_epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[checkpoint_callback],
        logger=logger,
    )
    model_cls = get_trainer(args.model)

    if not args.eval:
        dataloader = get_dataset(args.trainset, config)
        model = model_cls(config=config)
        trainer.fit(model, dataloader)
        exit(0)

    dataloader = get_dataset(args.evalset, config)

    if list(Path(logger.root_dir).glob("version_*")):
        last_run_version = sorted(
            list(Path(logger.root_dir).glob("version_*")), reverse=True
        )[0]
        last_model_path = last_run_version / "checkpoints" / "best.ckpt"
        model = model_cls.load_from_checkpoint(last_model_path, config=config)
    else:
        model = model_cls(config=config)
        print("No recent checkpoints found.")

    trainer.test(model, dataloader)
    print("Test finished.")
