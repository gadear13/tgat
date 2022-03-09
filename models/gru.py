import torch
from torch_geometric.nn import GCNConv


class GRUCell(torch.nn.Module):
    def __init__(
            self,
            batch_size: int,
            in_channels: int,
            out_channels: int,
            **kwargs

    ):
        super().__init__()
        self._batch_size = batch_size
        self._in_channels = in_channels
        self._out_channels = out_channels

        self.linear1 = torch.nn.Linear(1 + out_channels, 2 * out_channels)
        self.linear2 = torch.nn.Linear(1 + out_channels, out_channels)


    def forward(self, x, edge_index, edge_weight, h):
        # cat1
        cat1 = torch.concat([x, h], dim=1)

        # r, u
        ru = torch.sigmoid(self.linear1(cat1))
        ru = ru.reshape((self._batch_size, -1, 2 * self._out_channels)).reshape((self._batch_size, -1))
        r, u = torch.chunk(ru, chunks=2, dim=1)
        r = r.reshape((self._batch_size, -1, self._out_channels)).reshape((-1, self._out_channels))
        u = u.reshape((self._batch_size, -1, self._out_channels)).reshape((-1, self._out_channels))

        # c
        cat2 = torch.concat([x, r * h], dim=1)
        c = torch.tanh(self.linear2(cat2))

        # h
        h = u * h + (1.0 - u) * c
        return h

    @property
    def hyperparameters(self):
        return {"in_channels": self._in_channels, "out_channels": self._out_channels}

class GRU(torch.nn.Module):
    def __init__(self,
                 batch_size: int,
                 hid_channels: int,
                 **kwargs
                 ):
        super().__init__()

        self._batch_size = batch_size
        self._hid_channels = hid_channels
        self.grucell = GRUCell(batch_size, 1, hid_channels, **kwargs)

    def forward(self, x, edge_index, edge_weight):
        batch_nodes = x.size(0)
        pre_len = x.size(1)
        h = torch.zeros(batch_nodes, self._hid_channels).type_as(x)

        for step in range(pre_len):
            xt = x[:, step].unsqueeze(1)
            h = self.grucell(xt, edge_index, edge_weight, h)
        return h

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parent_parser.add_argument("--hid_channels", type=int, default=64)
        return parent_parser

    @property
    def hyperparameters(self):
        return {
            "batch_size": self._batch_size,
            "hid_channels": self._hid_channels,
        }
