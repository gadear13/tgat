from typing import Union, List, Tuple
import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse


class SpatioTemporalDT(Dataset):
    def __init__(self, tensor_x: Tensor, tensor_y: Tensor, adj: Tensor) -> None:
        super(SpatioTemporalDT, self).__init__()
        assert tensor_x.size(0) == tensor_y.size(0), "Size mismatch between tensors"
        self.tensor_x = tensor_x
        self.tensor_y = tensor_y
        self.edges, self.edge_weights = self.get_edges(adj)

    def get(self, index):
        return Data(x=self.tensor_x[index].T, edge_index=self.edges, edge_attr=self.edge_weights,
                    y=self.tensor_y[index].T)

    def len(self):
        return self.tensor_x.size(0)

    @staticmethod
    def get_edges(adj):
        edge_indices, values = dense_to_sparse(adj)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        edges = torch.LongTensor(edge_indices)
        edge_weights = torch.FloatTensor(values)
        return edges, edge_weights
