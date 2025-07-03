import lightning as L
import torch

from models import RawNet2, AASIST
from utils import *
from data import EchoFakeModule


class BaseTrainer(L.LightningModule):
    def on_validation_epoch_start(self):
        self.labels = []
        self.preds = []
        self.logits = []
        self.target_scores = []
        self.nontarget_scores = []

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        emb, logits = self.model(x)

        preds = torch.argmax(logits, dim=1)
        self.labels.extend(y.tolist())
        self.preds.extend(preds.tolist())

        if self.config["num_classes"] == 2:
            self.logits.extend(logits[:, 1].tolist())
            for i in range(x.size(0)):
                if y[i] == 1:
                    self.target_scores.append(logits[i, 1].item())
                else:
                    self.nontarget_scores.append(logits[i, 1].item())

    def on_validation_epoch_end(self):
        if self.config["num_classes"] == 2:
            eer, threshold = compute_eer(self.target_scores, self.nontarget_scores)
            self.log("val/eer", eer, sync_dist=True)
            self.preds = [1 if score >= threshold else 0 for score in self.logits]

        precision, recall, f1 = compute_f1(self.labels, self.preds)

        self.log("val/precision", precision, sync_dist=True)
        self.log("val/recall", recall, sync_dist=True)
        self.log("val/f1", f1, sync_dist=True)

    def on_test_epoch_start(self):
        self.labels = []
        self.preds = []
        self.logits = []
        self.target_scores = []
        self.nontarget_scores = []

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        emb, logits = self.model(x)

        preds = torch.argmax(logits, dim=1)
        self.labels.extend(y.tolist())
        self.preds.extend(preds.tolist())
        if self.config["num_classes"] == 2:
            self.logits.extend(logits[:, 1].tolist())
            for i in range(x.size(0)):
                if y[i] == 1:
                    self.target_scores.append(logits[i, 1].item())
                else:
                    self.nontarget_scores.append(logits[i, 1].item())

    def on_test_epoch_end(self):
        if self.config["num_classes"] == 2:
            eer, threshold = compute_eer(self.target_scores, self.nontarget_scores)
            self.log("test/eer", eer, sync_dist=True)
            self.preds = [1 if score >= threshold else 0 for score in self.logits]

        precision, recall, f1 = compute_f1(self.labels, self.preds)

        self.log("test/precision", precision, sync_dist=True)
        self.log("test/recall", recall, sync_dist=True)
        self.log("test/f1", f1, sync_dist=True)


class RawNet2Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = RawNet2(num_classes=config["num_classes"])
        # weight = torch.FloatTensor([0.1, 0.9])
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


class AASISTTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = AASIST(d_args=config["d_args"], num_classes=config["num_classes"])
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
