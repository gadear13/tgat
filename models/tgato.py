import argparse
import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GATv2Conv


class TGATGraphConvolution(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, heads: int, concat: bool, dropout: float,
                 share_weights: bool, **kwargs):
        super().__init__()
        if concat:
            assert output_dim % heads == 0
            hidden_dim = output_dim // heads
        else:
            hidden_dim = output_dim
        self._output_dim = output_dim
        self._num_gru_units = num_gru_units
        self.conv = GATv2Conv(self._num_gru_units + 1, hidden_dim, heads=heads, concat=concat, dropout=dropout,
                              share_weights=share_weights, **kwargs)

    def forward(self, inputs, edge_index, edge_att, hidden_state):
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (batch_size * num_nodes, num_gru_units + 1)
        x = concatenation.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = self.conv(x, edge_index)

        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGATCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, heads: int, concat: bool, dropout: float, share_weights: bool,
                 **kwargs):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.graph_conv1 = TGATGraphConvolution(self._hidden_dim, self._hidden_dim * 2, heads=heads, concat=concat,
                                                dropout=dropout, share_weights=share_weights, **kwargs)
        self.graph_conv2 = TGATGraphConvolution(self._hidden_dim, self._hidden_dim, heads=heads, concat=concat,
                                                dropout=dropout, share_weights=share_weights, **kwargs)

    def forward(self, inputs, edge_index, edge_att, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, edge_index, edge_att, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, edge_index, edge_att, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGATo(nn.Module):
    def __init__(self, adj, hidden_dim: int, heads: int, concat: bool,
                 dropout: float, share_weights: bool, **kwargs):
        super().__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self._heads = heads
        self._concat = concat
        self._dropout = dropout
        self._share_weights = share_weights
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGATCell(self._input_dim, self._hidden_dim, heads=heads, concat=concat, dropout=dropout,
                                  share_weights=share_weights, **kwargs)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        adj = self.adj.type_as(inputs)
        adj_list = [adj.unsqueeze(0) for _ in range(batch_size)]
        adj = torch.cat(adj_list, dim=0)
        edge_index, edge_att = dense_to_sparse(adj)
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], edge_index, edge_att, hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--heads", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.)
        parser.add_argument("--concat", type=bool, default=True)
        parser.add_argument("--share_weights", type=bool, default=False)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim, "heads": self._heads,
                "concat": self._concat, "dropout": self._dropout, "share_weights": self._share_weights}
