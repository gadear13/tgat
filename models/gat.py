import argparse
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse


class GAT(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 concat: bool = True, dropout: float = 0., share_weights: bool = False, **kwargs):
        super().__init__()
        if concat:
            assert out_channels % heads == 0, 'incorrect number heads'
            hid_channels = out_channels // heads
        else:
            hid_channels = out_channels
        self._in_channels = in_channels  # seq_len for prediction
        self._out_channels = out_channels  # hidden_dim for prediction
        self._heads = heads
        self._concat = concat
        self._dropout = dropout
        self._share_weights = share_weights
        self.conv = GATv2Conv(in_channels, hid_channels, heads=heads, concat=concat, dropout=dropout,
                              share_weights=share_weights)

    def forward(self, x, edge_index, edge_weight):
        # (batch_size*num_nodes, seq_len)
        ax = self.conv(x, edge_index)
        # (batch_size*num_nodes, output_dim)
        outputs = torch.tanh(ax)
        # act(AXW) (num_nodes * batch_size, output_dim)
        return outputs

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parent_parser.add_argument("--hid_channels", type=int, default=64)
        parent_parser.add_argument("--heads", type=int, default=2)
        parent_parser.add_argument("--dropout", type=float, default=0.)
        parent_parser.add_argument("--concat", type=bool, default=True)
        parent_parser.add_argument("--share_weights", type=bool, default=False)
        return parent_parser

    @property
    def hyperparameters(self):
        return {
            "in_channels": self._in_channels,
            "out_channels": self._out_channels,
            "heads": self._heads,
            "concat": self._concat,
            "dropout": self._dropout,
            "share_weights": self._share_weights
        }
