from __future__ import annotations

import math
import os
import torch

import pandas as pd
import numpy as np
import torch.nn as nn

from torch import Tensor
from torch import Tensor as T
from torch.nn.functional import relu
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Dataset


class CustomDataset(Dataset):
    def __init__(self, dataframes, use_cuda = False):
        self.dataframes = dataframes
        self.use_cuda = use_cuda
        
    def __len__(self):
        return len(self.dataframes)
    
    def __getitem__(self, idx):

        df = self.dataframes[idx]
        
        input_data = df[['phi','eta','z']]
        output_data = df['particle_id']

        # Create a tensor of node features by stacking the columns of the input data
        node_features = torch.tensor(input_data.values)

        phi = torch.from_numpy(input_data[['phi']].values)
        eta = torch.from_numpy(input_data[['eta']].values)
        Y = torch.from_numpy(output_data.values)

        if self.use_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            phi = phi.to(device)
            eta = eta.to(device)
            Y = Y.to(device)

        # Compute the pairwise differences between the phi and eta columns of adjacent nodes
        phi_diff = phi.unsqueeze(0) - phi.unsqueeze(-1)
        eta_diff = eta.unsqueeze(0) - eta.unsqueeze(-1)

        deltaR = torch.sqrt(phi_diff**2 + eta_diff**2)

        # Create a binary adjacency matrix based on a threshold of 1.7
        adjacency_matrix = torch.where(
            (deltaR < .1), 
            torch.ones_like(deltaR), 
            torch.zeros_like(deltaR)
        )

        # Convert the adjacency matrix to a list of edge indices
        edge_indices = adjacency_matrix.squeeze(-1).nonzero(as_tuple=False).t()

        
        Y_adj = (Y.unsqueeze(0) == Y.unsqueeze(-1)).fill_diagonal_(0)
        
        Y_edge_index = Y_adj.squeeze(-1).nonzero(as_tuple=False).t()
        
        
        # Create a tensor of edge features by concatenating the phi and eta differences    
        phi_diff = phi_diff[edge_indices[0], edge_indices[1]]
        eta_diff = eta_diff[edge_indices[0], edge_indices[1]]
        edge_features = torch.cat((
            phi_diff.unsqueeze(-1),
            eta_diff.unsqueeze(-1)
        ), -1)

        if self.use_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            node_features = node_features.to(device) 
            edge_indices = edge_indices.to(device) 
            edge_features = edge_features.to(device)
            
        # Create a PyTorch Geometric Data object
        data = Data(
            x          = node_features, 
            edge_index = edge_indices, 
            edge_attr  = edge_features,
            y          = Y_edge_index
        )
        return data




class EdgeClassifier(nn.Module):
    def __init__(
        self,
        node_indim,
        edge_indim,
        L=4,
        node_latentdim=8,
        edge_latentdim=12,
        r_hidden_size=32,
        o_hidden_size=32,
    ):
        super().__init__()
        self.node_encoder = MLP(node_indim, node_latentdim, 64, L=1)
        self.edge_encoder = MLP(edge_indim, edge_latentdim, 64, L=1)
        gnn_layers = []
        for _l in range(L):
            # fixme: Wrong parameters?
            gnn_layers.append(
                InteractionNetwork(
                    node_latentdim,
                    edge_latentdim,
                    node_outdim=node_latentdim,
                    edge_outdim=edge_latentdim,
                    edge_hidden_dim=r_hidden_size,
                    node_hidden_dim=o_hidden_size,
                )
            )
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.W = MLP(edge_latentdim, 1, 32, L=2)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        node_latent = self.node_encoder(x)
        edge_latent = self.edge_encoder(edge_attr)
        for layer in self.gnn_layers:
            node_latent, edge_latent = layer(node_latent, edge_index, edge_latent.squeeze(1))
        edge_weights = torch.sigmoid(self.W(edge_latent.unsqueeze(1)))
        
        return edge_weights


class ECForGraphTCN(nn.Module):
    def __init__(
        self,
        *,
        node_indim: int,
        edge_indim: int,
        interaction_node_hidden_dim=5,
        interaction_edge_hidden_dim=4,
        h_dim=5,
        e_dim=4,
        hidden_dim=40,
        L_ec=3,
        alpha_ec: float = 0.5,
    ):
        """Edge classification step to be used for Graph Track Condensor network
        (Graph TCN)

        Args:
            node_indim: Node feature dim
            edge_indim: Edge feature dim
            interaction_node_hidden_dim: Hidden dimension of interaction networks.
                Defaults to 5 for backward compatibility, but this is probably
                not reasonable.
            interaction_edge_hidden_dim: Hidden dimension of interaction networks
                Defaults to 4 for backward compatibility, but this is probably
                not reasonable.
            h_dim: node dimension in latent space
            e_dim: edge dimension in latent space
            hidden_dim: width of hidden layers in all perceptrons (edge and node
                encoders, hidden dims for MLPs in object and relation networks)
            L_ec: message passing depth for edge classifier
            alpha_ec: strength of residual connection for EC
        """
        super().__init__()
        self.relu = nn.ReLU()

        # specify the edge classifier
        self.ec_node_encoder = MLP(
            node_indim, h_dim, hidden_dim=hidden_dim, L=2, bias=False
        )
        self.ec_edge_encoder = MLP(
            edge_indim, e_dim, hidden_dim=hidden_dim, L=2, bias=False
        )
        self.ec_resin = ResIN.identical_in_layers(
            node_indim=h_dim,
            edge_indim=e_dim,
            node_outdim=h_dim,
            edge_outdim=e_dim,
            node_hidden_dim=interaction_node_hidden_dim,
            edge_hidden_dim=interaction_edge_hidden_dim,
            object_hidden_dim=hidden_dim,
            relational_hidden_dim=hidden_dim,
            alpha=alpha_ec,
            n_layers=L_ec,
        )

        self.W = MLP(
            e_dim + self.ec_resin.length_concatenated_edge_attrs, 1, hidden_dim, L=3
        )

    def forward(
        self,
        data: Data,
    ) -> Tensor:
        # apply the edge classifier to generate edge weights
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h_ec = self.relu(self.ec_node_encoder(x))
        edge_attr_ec = self.relu(self.ec_edge_encoder(edge_attr))
        h_ec, _, edge_attrs_ec = self.ec_resin(h_ec, edge_index, edge_attr_ec)

        # append edge weights as new edge features
        edge_attrs_ec = torch.cat(edge_attrs_ec, dim=1)
        edge_weights = torch.sigmoid(self.W(edge_attrs_ec))
        return edge_weights


class PerfectEdgeClassification(nn.Module):
    def __init__(self, tpr=1.0, tnr=1.0, false_below_pt=0.0):
        """An edge classifier that is perfect because it uses the truth information.
        If TPR or TNR is not 1.0, noise is added to the truth information.

        Args:
            tpr: True positive rate
            tnr: False positive rate
            false_below_pt: If not 0.0, all true edges between hits corresponding to
                particles with a pt lower than this threshold are set to false.
                This is not counted towards the TPR/TNR but applied afterwards.
        """
        super().__init__()
        assert 0.0 <= tpr <= 1.0
        self.tpr = tpr
        assert 0.0 <= tnr <= 1.0
        self.tnr = tnr
        self.false_below_pt = false_below_pt

    def forward(self, data: Data) -> Tensor:
        r = data.y.bool()
        if not np.isclose(self.tpr, 1.0):
            true_mask = r.detach().clone()
            rand = torch.rand(int(true_mask.sum()), device=r.device)
            r[true_mask] = rand <= self.tpr
        if not np.isclose(self.tnr, 1.0):
            false_mask = (~r).detach().clone()
            rand = torch.rand(int(false_mask.sum()), device=r.device)
            r[false_mask] = ~(rand <= self.tnr)
        if self.false_below_pt > 0.0:
            false_mask = data.pt < self.false_below_pt
            r[false_mask] = False
        return r


# noinspection PyAbstractClass
class InteractionNetwork(MessagePassing):
    def __init__(
        self,
        node_indim: int,
        edge_indim: int,
        node_outdim=3,
        edge_outdim=4,
        node_hidden_dim=40,
        edge_hidden_dim=40,
        aggr="add",
    ):
        """Interaction Network, consisting of a relational model and an object model,
        both represented as MLPs.

        Args:
            node_indim: Node feature dimension
            edge_indim: Edge feature dimension
            node_outdim: Output node feature dimension
            edge_outdim: Output edge feature dimension
            node_hidden_dim: Hidden dimension for the object model MLP
            edge_hidden_dim: Hidden dimension for the relational model MLP
            aggr: How to aggregate the messages
        """
        super().__init__(aggr=aggr, flow="source_to_target")
        self.relational_model = MLP(
            2 * node_indim + edge_indim,
            edge_outdim,
            edge_hidden_dim,
        )
        self.object_model = MLP(
            node_indim + edge_outdim,
            node_outdim,
            node_hidden_dim,
        )
        self._e_tilde: T | None = None

    def forward(self, x: T, edge_index: T, edge_attr: T) -> tuple[T, T]:
        """Forward pass

        Args:
            x: Input node features
            edge_index:
            edge_attr: Input edge features

        Returns:
            Output node embedding, output edge embedding
        """
        x_tilde = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        assert self._e_tilde is not None  # mypy
        return x_tilde, self._e_tilde

    # noinspection PyMethodOverriding
    def message(self, x_i: T, x_j: T, edge_attr: T) -> T:
        """Calculate message of an edge

        Args:
            x_i: Features of node 1 (node where the edge ends)
            x_j: Features of node 2 (node where the edge starts)
            edge_attr: Edge features

        Returns:
            Message
        """
        m = torch.cat([x_i, x_j, edge_attr], dim=1)
        self._e_tilde = self.relational_model(m)
        assert self._e_tilde is not None  # mypy
        return self._e_tilde

    # noinspection PyMethodOverriding
    def update(self, aggr_out: T, x: T) -> T:
        """Update for node embedding

        Args:
            aggr_out: Aggregated messages of all edges
            x: Node features for the node that receives all edges

        Returns:
            Updated node features/embedding
        """
        c = torch.cat([x, aggr_out], dim=1)
        return self.object_model(c)


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_dim,
        L=3,
        *,
        bias=True,
        include_last_activation=False,
    ):
        """Multi Layer Perceptron, using ReLu as activation function.

        Args:
            input_size: Input feature dimension
            output_size:  Output feature dimension
            hidden_dim: Feature dimension of the hidden layers
            L: Total number of layers (1 initial layer, L-2 hidden layers, 1 output
                layer)
            bias: Include bias in linear layer?
            include_last_activation: Include activation function for the last layer?
        """
        super().__init__()
        layers = [nn.Linear(input_size, hidden_dim, bias=bias)]
        for _l in range(1, L - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_size, bias=bias))
        if include_last_activation:
            layers.append(nn.ReLU())
        self.layers = nn.ModuleList(layers)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@torch.jit.script
def _convex_combination(*, delta: T, residue: T, alpha_residue: float) -> T:
    return alpha_residue * residue + (1 - alpha_residue) * relu(delta)


_IDENTITY = nn.Identity()


def convex_combination(
    *,
    delta: T,
    residue: T,
    alpha_residue: float,
    residue_encoder: nn.Module = _IDENTITY,
) -> T:
    """Convex combination of ``relu(delta)`` and the residue."""
    if math.isclose(alpha_residue, 0.0):
        return relu(delta)
    assert 0 <= alpha_residue <= 1
    return _convex_combination(
        delta=delta, residue=residue_encoder(residue), alpha_residue=alpha_residue
    )


class ResIN(nn.Module):
    def __init__(
        self,
        layers: list[nn.Module],
        *,
        length_concatenated_edge_attrs: int,
        alpha: float = 0.5,
    ):
        """Apply a list of layers in sequence with residual connections for the nodes.
        Built for interaction networks, but any network that returns a node feature
        tensor and an edge feature tensor should
        work.

        Note that a ReLu activation function is applied to the node result of the
        layer.

        Args:
            layers: List of layers
            length_concatenated_edge_attrs: Length of the concatenated edge attributes
                (from all the different layers)
            alpha: Strength of the node embedding residual connection
        """
        super().__init__()
        self._layers = nn.ModuleList(layers)
        self._alpha = alpha
        #: Because of the residual connections, we need map the output of the previous
        #: layer to the dimension of the next layer (if they are different). This
        #: can be done with these encoders.
        self._residue_node_encoders = nn.ModuleList([nn.Identity() for _ in layers])
        self.length_concatenated_edge_attrs = length_concatenated_edge_attrs

    @staticmethod
    def _get_residue_encoder(
        *, in_dim: int, out_dim: int, hidden_dim: int
    ) -> nn.Module:
        if in_dim == out_dim:
            return nn.Identity()
        return MLP(
            input_size=in_dim,
            output_size=out_dim,
            hidden_dim=hidden_dim,
            include_last_activation=True,
            L=2,
        )

    @classmethod
    def identical_in_layers(
        cls,
        *,
        node_indim: int,
        edge_indim: int,
        node_hidden_dim: int,
        edge_hidden_dim: int,
        node_outdim=3,
        edge_outdim=4,
        object_hidden_dim=40,
        relational_hidden_dim=40,
        alpha: float = 0.5,
        n_layers=1,
    ) -> ResIN:
        """Create a ResIN with identical layers of interaction networks except for
        the first and last one (different dimensions)

        If the input/hidden/output dimensions for the nodes are not the same, MLPs are
        used to map the previous output for the residual connection.

        Args:
            node_indim: Node feature dimension
            edge_indim: Edge feature dimension
            node_hidden_dim: Node feature dimension for the hidden layers
            edge_hidden_dim: Edge feature dimension for the hidden layers
            node_outdim: Output node feature dimension
            edge_outdim: Output edge feature dimension
            object_hidden_dim: Hidden dimension for the object model MLP
            relational_hidden_dim: Hidden dimension for the relational model MLP
            alpha: Strength of the residual connection
            n_layers: Total number of layers
        """
        node_dims = [
            node_indim,
            *[node_hidden_dim for _ in range(n_layers - 1)],
            node_outdim,
        ]
        edge_dims = [
            edge_indim,
            *[edge_hidden_dim for _ in range(n_layers - 1)],
            edge_outdim,
        ]
        assert len(node_dims) == len(edge_dims) == n_layers + 1
        layers = [
            InteractionNetwork(
                node_indim=node_dims[i],
                edge_indim=edge_dims[i],
                node_outdim=node_dims[i + 1],
                edge_outdim=edge_dims[i + 1],
                node_hidden_dim=object_hidden_dim,
                edge_hidden_dim=relational_hidden_dim,
            )
            for i in range(n_layers)
        ]
        length_concatenated_edge_attrs = edge_hidden_dim * (n_layers - 1) + edge_outdim
        mod = cls(
            layers,
            length_concatenated_edge_attrs=length_concatenated_edge_attrs,
            alpha=alpha,
        )
        mod._residue_node_encoders = nn.ModuleList(
            [
                cls._get_residue_encoder(
                    in_dim=node_dims[i],
                    out_dim=node_dims[i + 1],
                    hidden_dim=node_dims[i + 1],
                )
                for i in range(n_layers)
            ]
        )
        return mod

    def forward(
        self, x, edge_index, edge_attr
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """Forward pass

        Args:
            x: Node features
            edge_index:
            edge_attr: Edge features

        Returns:
            node embedding, node embedding at each layer (including the input and
            final node embedding), edge embedding at each layer (including the input)
        """
        edge_attrs = [edge_attr]
        xs = [x]
        for layer, re in zip(self._layers, self._residue_node_encoders):
            delta_x, edge_attr = layer(x, edge_index, edge_attr)
            x = convex_combination(
                delta=delta_x, residue=x, alpha_residue=self._alpha, residue_encoder=re
            )
            xs.append(x)
            edge_attrs.append(edge_attr)
        return x, xs, edge_attrs

