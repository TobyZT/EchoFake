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

from train import get_trainer
from data import EchoFakeModule

CUDA_VISIBLE_DEVICES = [1]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="RawNet2")
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
    num_workers = config["train"]["num_workers"]
    max_len = config["train"]["max_len"]
    batch_size = config["train"]["batch_size"]
    check_val_every_n_epoch = config["train"]["check_val_every_n_epoch"]
    save_top_k = config["train"]["save_top_k"]
    trainset_path = config["path"]["trainset"]
    # set seed
    L.seed_everything(seed=seed)

    # set logger
    version = "eval" if args.eval else None
    logger = pl_loggers.TensorBoardLogger(
        save_dir="exp", name=exp_name, version=version
    )
    logger.log_hyperparams(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/f1",
        dirpath=logger.log_dir + "/checkpoints",
        filename="best",
        save_top_k=save_top_k,
        save_last=True,
        mode="max",
        every_n_epochs=1,
    )

    # set devices
    trainer = L.Trainer(
        devices=CUDA_VISIBLE_DEVICES,
        accelerator="gpu",
        max_epochs=num_epochs,
        log_every_n_steps=20,
        check_val_every_n_epoch=check_val_every_n_epoch,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[checkpoint_callback],
        logger=logger,
    )
    model_cls = get_trainer(args.model)
    model = model_cls(config=config)
    dataloader = EchoFakeModule(
        trainset_path, batch_size=batch_size, num_workers=num_workers, max_len=max_len
    )

    if not args.eval:
        trainer.fit(model, dataloader)
        exit(0)

    last_run_version = sorted(
        list(Path(logger.root_dir).glob("version_*")), reverse=True
    )[0]
    last_model_path = last_run_version / "checkpoints" / "best.ckpt"
    model = model_cls.load_from_checkpoint(last_model_path, config=config)

    trainer.test(model, dataloader)
    print("Test finished.")
