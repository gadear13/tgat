# import argparse
#
# import pytorch_lightning as pl
# from torch_geometric.loader import DataLoader
# import pandas as pd
# import numpy as np
# import torch
#
# from .spatiotemporal_dt import SpatioTemporalDT
#
#
# def load_features(feat_path, dtype=np.float32):
#     feat_df = pd.read_csv(feat_path)
#     feat = np.array(feat_df, dtype=dtype)
#     return feat
#
#
# def load_adjacency_matrix(adj_path, dtype=np.float32):
#     adj_df = pd.read_csv(adj_path, header=None)
#     adj = np.array(adj_df, dtype=dtype)
#     return adj
#
#
# def generate_dataset(
#         data, seq_len, pre_len, time_len=None, val_size=0.1, test_size=0.1, normalize=True
# ):
#     if time_len is None:
#         time_len = data.shape[0]
#     train_size = int(time_len * (1 - val_size - test_size))
#     val_size = int(time_len * (1 - test_size))
#     train_data = data[:train_size]
#     val_data = data[train_size:val_size]
#     test_data = data[val_size:time_len]
#     if normalize:
#         max_val = np.max(train_data)
#         train_data = train_data / max_val
#         val_data = val_data / max_val
#         test_data = test_data / max_val
#
#     train_X, train_Y, val_X, val_Y, test_X, test_Y = list(), list(), list(), list(), list(), list()
#     for i in range(len(train_data) - seq_len - pre_len):
#         train_X.append(np.array(train_data[i: i + seq_len]))
#         train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len]))
#     for i in range(len(val_data) - seq_len - pre_len):
#         val_X.append(np.array(val_data[i: i + seq_len]))
#         val_Y.append(np.array(val_data[i + seq_len: i + seq_len + pre_len]))
#     for i in range(len(test_data) - seq_len - pre_len):
#         test_X.append(np.array(test_data[i: i + seq_len]))
#         test_Y.append(np.array(test_data[i + seq_len: i + seq_len + pre_len]))
#     return np.array(train_X), np.array(train_Y), np.array(val_X), np.array(val_Y), np.array(test_X), np.array(test_Y)
#
#
# def generate_torch_datasets(
#         data, adj, seq_len, pre_len, time_len=None, val_size=0.1, test_size=0.1, normalize=True
# ):
#     train_X, train_Y, val_X, val_Y, test_X, test_Y = generate_dataset(
#         data,
#         seq_len,
#         pre_len,
#         time_len=time_len,
#         val_size=val_size,
#         test_size=test_size,
#         normalize=normalize,
#     )
#     train_dataset = SpatioTemporalDT(torch.FloatTensor(train_X), torch.FloatTensor(train_Y),
#                                      torch.FloatTensor(adj))
#     val_dataset = SpatioTemporalDT(torch.FloatTensor(val_X), torch.FloatTensor(val_Y), torch.FloatTensor(adj))
#     test_dataset = SpatioTemporalDT(torch.FloatTensor(test_X), torch.FloatTensor(test_Y), torch.FloatTensor(adj))
#     return train_dataset, val_dataset, test_dataset
#
#
# class GATDataModule(pl.LightningDataModule):
#     def __init__(self,
#                  feat_path: str,
#                  adj_path: str,
#                  batch_size: int = 64,
#                  seq_len: int = 12,
#                  pre_len: int = 3,
#                  val_size=0.1,
#                  test_size=0.1,
#                  normalize: bool = True,
#                  **kwargs):
#         super().__init__()
#         self._feat_path = feat_path
#         self._adj_path = adj_path
#         self.batch_size = batch_size
#         self.seq_len = seq_len
#         self.pre_len = pre_len
#         self.val_size = val_size
#         self.test_size = test_size
#         self.normalize = normalize
#         self._feat = load_features(self._feat_path)
#         self._feat_max_val = np.max(self._feat)
#         self._adj = load_adjacency_matrix(self._adj_path)
#
#     def setup(self, stage: str = None):
#         (
#             self.train_dataset,
#             self.val_dataset,
#             self.test_dataset,
#         ) = generate_torch_datasets(
#             self._feat,
#             self._adj,
#             self.seq_len,
#             self.pre_len,
#             val_size=self.val_size,
#             test_size=self.test_size,
#             normalize=self.normalize,
#         )
#
#     def train_dataloader(self):
#         # return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
#         return DataLoader(self.train_dataset, batch_size=self.batch_size)
#
#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size)
#
#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size)
#
#     @property
#     def feat_max_val(self):
#         return self._feat_max_val
#
#     @property
#     def adj(self):
#         return self._adj
#
#     @staticmethod
#     def add_data_specific_arguments(parent_parser):
#         parent_parser.add_argument("--batch_size", type=int, default=32)
#         parent_parser.add_argument("--seq_len", type=int, default=12)
#         parent_parser.add_argument("--pre_len", type=int, default=3)
#         parent_parser.add_argument("--val_size", type=float, default=0.15)
#         parent_parser.add_argument("--test_size", type=float, default=0.15)
#         parent_parser.add_argument("--normalize", type=bool, default=True)
#         return parent_parser
#
#
# if __name__ == "__main__":
#     DATA_PATHS = {
#         "shenzhen": {"feat": "../data/sz_speed.csv", "adj": "../data/sz_adj.csv"},
#         "losloop": {"feat": "../data/los_speed.csv", "adj": "../data/los_adj.csv"},
#         "m30": {"feat": "../data/m30_speed.csv", "adj": "../data/m30_speed_adj.csv"},
#     }
#
#     dm = GATDataModule(feat_path=DATA_PATHS['shenzhen']["feat"], adj_path=DATA_PATHS['shenzhen']["adj"])
#     dm.setup()
#     for a in dm.train_dataloader():
#         print(a)
#
#

import argparse

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import torch

from .spatiotemporal_dt import SpatioTemporalDT


def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj


def generate_dataset(
        data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i: i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i: i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len: i + seq_len + pre_len]))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(
        data, adj, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y,test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = SpatioTemporalDT(torch.FloatTensor(train_X), torch.FloatTensor(train_Y),
                                     torch.FloatTensor(adj))
    test_dataset = SpatioTemporalDT(torch.FloatTensor(test_X), torch.FloatTensor(test_Y), torch.FloatTensor(adj))
    return train_dataset, test_dataset


class GATDataModule(pl.LightningDataModule):
    def __init__(self,
                 feat_path: str,
                 adj_path: str,
                 batch_size: int = 64,
                 seq_len: int = 12,
                 pre_len: int = 3,
                 split_ratio: float = 0.8,
                 normalize: bool = True,
                 **kwargs):
        super().__init__()
        self._feat_path = feat_path
        self._adj_path = adj_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self._feat = load_features(self._feat_path)
        self._feat_max_val = np.max(self._feat)
        self._adj = load_adjacency_matrix(self._adj_path)

    def setup(self, stage: str = None):
        (
            self.train_dataset,
            self.val_dataset,
        ) = generate_torch_datasets(
            self._feat,
            self._adj,
            self.seq_len,
            self.pre_len,
            split_ratio=self.split_ratio,
            normalize=self.normalize,
        )

    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)

    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def adj(self):
        return self._adj

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parent_parser.add_argument("--batch_size", type=int, default=32)
        parent_parser.add_argument("--seq_len", type=int, default=12)
        parent_parser.add_argument("--pre_len", type=int, default=3)
        parent_parser.add_argument("--split_ratio", type=float, default=0.8)
        parent_parser.add_argument("--normalize", type=bool, default=True)
        return parent_parser


if __name__ == "__main__":
    from alfa.datamodules import TGCNDataModule

    DATA_PATHS = {
        "shenzhen": {"feat": "../data/sz_speed.csv", "adj": "../data/sz_adj.csv"},
        "losloop": {"feat": "../data/los_speed.csv", "adj": "../data/los_adj.csv"},
        "m30": {"feat": "../data/m30_speed.csv", "adj": "../data/m30_speed_adj.csv"},
    }

    dm1 = GATDataModule(feat_path=DATA_PATHS['shenzhen']["feat"], adj_path=DATA_PATHS['shenzhen']["adj"])
    dm2 = TGCNDataModule(feat_path=DATA_PATHS['shenzhen']["feat"], adj_path=DATA_PATHS['shenzhen']["adj"])
    dm1.setup()
    dm2.setup()
    for a, b in zip(dm1.train_dataloader(), dm2.train_dataloader()):
        x1 = a.x
        x2 = b[0]
        y1 = a.y
        y2 = b[1]
        x2 = x2.transpose(1,2)
        x2 = x2.reshape((64*156, 12))
        y2 = y2.transpose(1,2)
        y2 = y2.reshape((64*156, 3))
        print(x1)
