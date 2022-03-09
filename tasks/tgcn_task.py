import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import pandas as pd
import utils.metrics
import utils.losses


class TGCNForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        regressor="linear",
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
                self.model.hyperparameters.get("hidden_dim")
                or self.model.hyperparameters.get("output_dim"),
                self.hparams.pre_len,
            )
            if regressor == "linear"
            else regressor
        )
        self._loss = loss
        self.feat_max_val = feat_max_val
        self.data = pd.DataFrame()

    def forward(self, x):
        # (batch_size, seq_len, num_nodes)
        batch_size, _, num_nodes = x.size()
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.model(x)
        # (batch_size * num_nodes, hidden_dim)
        hidden = hidden.reshape((-1, hidden.size(2)))
        # (batch_size * num_nodes, pre_len)
        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions

    def shared_step(self, batch, batch_idx):
        # (batch_size, seq_len/pre_len, num_nodes)
        x, y = batch
        num_nodes = x.size(2)
        predictions = self(x)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y = y.reshape((-1, y.size(2)))
        return predictions, y

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
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
        self.log_dict(metrics)
        return metrics

    # def validation_step(self, batch, batch_idx):
    #     predictions, y = self.shared_step(batch, batch_idx)
    #     predictions = predictions * self.feat_max_val
    #     y = y * self.feat_max_val
    #     loss = self.loss(predictions, y)
    #     metrics = self.metrics_window(y, predictions)
    #     metrics["val_loss"] = loss
    #     self.log_dict(metrics)
    #     return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def metrics_window(self, ytrue, yhat):
        metrics = dict()
        for i in range(0, ytrue.size(1)):
            ytruei = ytrue[:, i, :]
            # ytruei = ytruei.reshape((-1, ytruei.size(1)))
            yhati = yhat[:, i, :]
            # yhati = yhati.reshape((-1, yhati.size(1)))
            metrics[f"RMSE_{i+1}"] = torch.sqrt(torchmetrics.functional.mean_squared_error(yhati, ytruei))
            metrics[f"MAE_{i+1}"] = torchmetrics.functional.mean_absolute_error(yhati, ytruei)
        return metrics

    def test_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
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

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--loss", type=str, default="mse")
        return parser
