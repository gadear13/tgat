import argparse
import traceback
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import datamodules
import pandas as pd
import os

DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
    "m30": {"feat": "data/m30_speed.csv", "adj": "data/m30_speed_adj.csv"},
    "m302": {"feat": "data/m30_speed.csv", "adj": "data/m30_speed2_adj.csv"},
}


def get_model(args, dm):
    model = None
    if args.model_name == "GCN":
        model = models.GCN(in_channels=args.seq_len, out_channels=args.hid_channels, improved=args.improved)
    if args.model_name == "GCNo":
        model = models.GCNo(adj=dm.adj, input_dim=args.seq_len, output_dim=args.hidden_dim)
    if args.model_name == "GRU":
        model = models.GRU(batch_size=args.batch_size, hid_channels=args.hid_channels)
    if args.model_name == "TGCN":
        model = models.TGCN(hid_channels=args.hid_channels, improved=args.improved)
    if args.model_name == "TGCNo":
        model = models.TGCNo(adj=dm.adj, hidden_dim=args.hidden_dim)
    if args.model_name == "TGCNv2":
        model = models.TGCNv2(batch_size=args.batch_size, hid_channels=args.hid_channels, improved=args.improved)
    if args.model_name == "TGAT":
        model = models.TGAT(hid_channels=args.hid_channels, heads=args.heads, concat=args.concat,
                            dropout=args.dropout, share_weights=args.share_weights)
    if args.model_name == "TGATv2":
        model = models.TGATv2(hid_channels=args.hid_channels, heads=args.heads, concat=args.concat,
                              dropout=args.dropout, share_weights=args.share_weights)
    if args.model_name == "TGATv3":
        model = models.TGATv3(batch_size=args.batch_size, hid_channels=args.hid_channels, heads=args.heads,
                              concat=args.concat, dropout=args.dropout, share_weights=args.share_weights)
    if args.model_name == "TGATo":
        model = models.TGATo(adj=dm.adj, hidden_dim=args.hidden_dim, heads=args.heads, concat=args.concat,
                             dropout=args.dropout, share_weights=args.share_weights)
    if args.model_name == "GAT":
        model = models.GAT(in_channels=args.seq_len, out_channels=args.hid_channels, heads=args.heads,
                           concat=args.concat, dropout=args.dropout, share_weights=args.share_weights)
    return model


def save_results(results, args):
    f = int(args.pre_len / 3) if args.data == 'losloop' else args.pre_len
    path = f'{args.result_path}/{args.data}/{f}'
    if not os.path.exists(path):
        os.makedirs(path)
    if args.model_name in ['TGAT', 'TGATv2', 'TGATv3', 'GAT']:
        filename = f'{path}/{args.model_name}_{args.data}_{args.hid_channels}_{args.dropout}_{args.weight_decay}.csv'
    elif args.model_name in ['GCNo', 'TGCNo']:
        filename = f'{path}/{args.model_name}_{args.data}_{args.hidden_dim}.csv'
    else:
        filename = f'{path}/{args.model_name}_{args.data}_{args.hid_channels}_{args.weight_decay}.csv'
    df = pd.DataFrame(results)
    df.to_csv(filename, sep='|', index=False)


def main_tgcn(args):
    dm = datamodules.TGCNDataModule(
        feat_path=DATA_PATHS[args.data]["feat"], adj_path=DATA_PATHS[args.data]["adj"], **vars(args)
    )
    model = get_model(args, dm)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="train_loss")
    task = tasks.TGCNForecastTask(model=model, feat_max_val=dm.feat_max_val, **vars(args))
    # trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(task, dm)
    trainer.test(dataloaders=dm, ckpt_path='best')
    results = task.data
    return results


def main_gat(args):
    dm = datamodules.GATDataModule(
        feat_path=DATA_PATHS[args.data]["feat"], adj_path=DATA_PATHS[args.data]["adj"], **vars(args)
    )
    model = get_model(args, dm)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="train_loss")
    task = tasks.GATForecastTask(model=model, feat_max_val=dm.feat_max_val, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    # trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(task, dm)
    trainer.test(dataloaders=dm, ckpt_path='best')
    results = task.data
    return results


def main(args):
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)
    return results


if __name__ == '__main__':
    # pl.seed_everything(0, workers=True)
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--data",
                        type=str,
                        help="The name of the dataset",
                        choices=("shenzhen", "losloop", "m30", "m302"),
                        default="m30")
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("GCN", "GCNo", "GRU", "TGCN", "TGCNo", "TGCNv2", "TGAT", "TGATo", "TGATv2", "TGATv3", "GAT"),
        default="GAT",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. tgc learning",
        choices=("gat", "tgcn"),
        default="gat",
    )
    parser.add_argument("--log_path", type=str, default='lightning_logs', help="Path to the output console log file")
    parser.add_argument("--log_name", type=str, default=None, help="Name of the log directory")
    parser.add_argument("--result_path", type=str, default="results", help="Path to results")

    temp_args, _ = parser.parse_known_args()
    if temp_args.log_name is None:
        log_name = temp_args.data
    else:
        log_name = temp_args.log_name
    logger = TensorBoardLogger(temp_args.log_path, name=log_name, default_hp_metric=False)

    parser = getattr(datamodules, temp_args.settings.upper() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.upper() + "ForecastTask").add_task_specific_arguments(parser)

    args = parser.parse_args()
    args.logger = logger
    args.gpus = [args.gpus]
    if args.log_name is None:
        args.log_name = f'logs_{args.data}'
    results = main(args)

    if args.result_path:
        save_results(results, args)
