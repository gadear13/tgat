import torch
from torch_geometric.nn import GCNConv


class TGCNCell(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            improved: bool = True,
            **kwargs

    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._improved = improved

        self.linear1 = torch.nn.Linear(2 * out_channels, 2 * out_channels)
        self.linear2 = torch.nn.Linear(2 * out_channels, out_channels)
        self.conv = GCNConv(in_channels, out_channels, improved=improved, **kwargs)

    def forward(self, x, edge_index, edge_weight, h):
        # Convolution
        f = torch.sigmoid(self.conv(x, edge_index, edge_weight))

        # r, u
        cat1 = torch.concat([f, h], dim=1)
        ru = torch.sigmoid(self.linear1(cat1))
        r, u = torch.chunk(ru, chunks=2, dim=1)

        # c
        cat2 = torch.concat([f, r * h], dim=1)
        c = torch.tanh(self.linear2(cat2))

        # h
        h = u * h + (1.0 - u) * c
        return h

    @property
    def hyperparameters(self):
        return {"in_channels": self._in_channels, "out_channels": self._out_channels}

class TGCN(torch.nn.Module):
    def __init__(self,
                 hid_channels: int,
                 improved: bool = True,
                 **kwargs
                 ):
        super().__init__()

        self._hid_channels = hid_channels
        self._improved = improved
        self.gcncell = TGCNCell(1, hid_channels, improved=improved, **kwargs)

    def forward(self, x, edge_index, edge_weight):
        batch_nodes = x.size(0)
        pre_len = x.size(1)
        h = torch.zeros(batch_nodes, self._hid_channels).type_as(x)

        for step in range(pre_len):
            xt = x[:, step].unsqueeze(1)
            h = self.gcncell(xt, edge_index, edge_weight, h)
        return h

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parent_parser.add_argument("--hid_channels", type=int, default=64)
        parent_parser.add_argument("--improved", type=bool, default=True)
        return parent_parser

    @property
    def hyperparameters(self):
        return {
            "hid_channels": self._hid_channels,
            "improved": self._improved,
        }
