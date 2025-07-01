import lightning as L
import torch

from models import RawNet2, AASIST
from utils import *


def get_trainer(model_name):
    if model_name.lower() == "rawnet2":
        return RawNet2Trainer
    elif model_name.lower() == "aasist":
        return AASISTTrainer


class RawNet2Trainer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = RawNet2(nb_classes=4)
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        lr = float(self.config["optimizer"]["lr"])
        weight_decay = float(self.config["optimizer"]["weight_decay"])
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        emb, logits = self.model(x)

        loss = self.ce_loss(logits, y)
        self.log("train/loss", loss)

        return loss

    def on_validation_epoch_start(self):
        self.labels = []
        self.preds = []

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        emb, logits = self.model(x)

        preds = torch.argmax(logits, dim=1)
        self.labels.extend(y.tolist())
        self.preds.extend(preds.tolist())

    def on_validation_epoch_end(self):
        precision, recall, f1 = compute_f1(self.labels, self.preds)
        eer = compute_eer(self.preds, self.labels)
        self.log("val/precision", precision, sync_dist=True)
        self.log("val/recall", recall, sync_dist=True)
        self.log("val/f1", f1, sync_dist=True)
        self.log("val/eer", eer, sync_dist=True)

    def on_test_epoch_start(self):
        self.labels = []
        self.preds = []

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        emb, logits = self.model(x)

        preds = torch.argmax(logits, dim=1)
        self.labels.extend(y.tolist())
        self.preds.extend(preds.tolist())

    def on_test_epoch_end(self):
        precision, recall, f1 = compute_f1(self.labels, self.preds)
        eer = compute_eer(self.preds, self.labels)
        self.log("test/precision", precision, sync_dist=True)
        self.log("test/recall", recall, sync_dist=True)
        self.log("test/f1", f1, sync_dist=True)
        self.log("test/eer", eer, sync_dist=True)


class AASISTTrainer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = AASIST(d_args=config["d_args"], num_classes=4)
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        lr = float(self.config["optimizer"]["lr"])
        weight_decay = float(self.config["optimizer"]["weight_decay"])
        betas = tuple(self.config["optimizer"]["betas"])
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
        total_steps = self.trainer.estimated_stepping_batches
        min_lr = float(self.config["scheduler"]["min_lr"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                min_lr / lr,
            ),
        )
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        emb, logits = self.model(x)

        loss = self.ce_loss(logits, y)
        self.log("train/loss", loss)

        return loss

    def on_validation_epoch_start(self):
        self.labels = []
        self.preds = []

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        emb, logits = self.model(x)

        preds = torch.argmax(logits, dim=1)
        self.labels.extend(y.tolist())
        self.preds.extend(preds.tolist())

    def on_validation_epoch_end(self):
        precision, recall, f1 = compute_f1(self.labels, self.preds)
        eer = compute_eer(self.preds, self.labels)
        self.log("val/precision", precision, sync_dist=True)
        self.log("val/recall", recall, sync_dist=True)
        self.log("val/f1", f1, sync_dist=True)
        self.log("val/eer", eer, sync_dist=True)

    def on_test_epoch_start(self):
        self.labels = []
        self.preds = []

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        emb, logits = self.model(x)

        preds = torch.argmax(logits, dim=1)
        self.labels.extend(y.tolist())
        self.preds.extend(preds.tolist())

    def on_test_epoch_end(self):
        precision, recall, f1 = compute_f1(self.labels, self.preds)
        eer = compute_eer(self.preds, self.labels)
        self.log("test/precision", precision, sync_dist=True)
        self.log("test/recall", recall, sync_dist=True)
        self.log("test/f1", f1, sync_dist=True)
        self.log("test/eer", eer, sync_dist=True)
