import argparse

import pandas as pd
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

import utils.losses


class GATForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss="mse",
        pre_len: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore='model')
        self.model = model
        self.regressor = (
            nn.Linear(
                self.model.hyperparameters.get("hid_channels")
                or self.model.hyperparameters.get("out_channels"),
                self.hparams.pre_len,
            )
        )
        self._loss = loss
        self.feat_max_val = feat_max_val
        self._loss = loss
        self.feat_max_val = feat_max_val
        self.data = pd.DataFrame()

    def forward(self, batch):
        x, edge_index, edge_weight = batch.x, batch.edge_index, batch.edge_attr
        # (batch_size * num_nodes, hidden_dim)
        hidden = self.model(x=x, edge_index=edge_index, edge_weight=edge_weight)
        # (batch_size * num_nodes, pre_len)
        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
        return predictions

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"train_loss": 1e3, "val_loss": 1e7})

    def training_step(self, batch, batch_idx):
        y = batch.y
        predictions = self(batch)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch.y
        predictions = self(batch)
        predictions = predictions * self.feat_max_val
        y = y * self.feat_max_val
        loss = self.loss(predictions, y)
        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        metrics = {
            "val_loss": loss,
            "val_RMSE": rmse,
            "val_MAE": mae,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics

    def test_step(self, batch, batch_idx):
        y = batch.y
        predictions = self(batch)
        predictions = predictions * self.feat_max_val
        y = y * self.feat_max_val
        loss = self.loss(predictions, y)
        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        metrics = {
            "test_loss": loss,
            "test_RMSE": rmse,
            "test_MAE": mae,
        }
        self.log_dict(metrics, on_step=True, on_epoch=False)
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def test_epoch_end(self, outputs):
        rmse = torch.stack([x['test_RMSE'] for x in outputs])
        mae = torch.stack([x['test_MAE'] for x in outputs])
        metrics = {
            "test_RMSE_mean": rmse.mean(),
            "test_RMSE_std": rmse.std(dim=0),
            "test_MAE_mean": mae.mean(),
            "test_MAE_std": mae.std(dim=0),
        }
        self.log_dict(metrics)

        rmse = [x['test_RMSE'].item() for x in outputs]
        mae = [x['test_MAE'].item() for x in outputs]
        df = pd.DataFrame({'RMSE': rmse, 'MAE': mae})
        self.data = df
        return metrics

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parent_parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parent_parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parent_parser.add_argument("--loss", type=str, default="mse")
        return parent_parser