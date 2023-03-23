import torch
from torch_geometric.data import Data, Dataset
import os
import pandas as pd

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
        node_features = torch.tensor(input_data.values).half()
        output_features = torch.tensor(output_data.values).half()

        phi = torch.from_numpy(input_data[['phi']].values)
        eta = torch.from_numpy(input_data[['eta']].values)

        if self.use_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            phi = phi.to(device)
            eta = eta.to(device)

        # Compute the pairwise differences between the phi and eta columns of adjacent nodes
        phi_diff = phi.unsqueeze(0) - phi.unsqueeze(-1)
        eta_diff = eta.unsqueeze(0) - eta.unsqueeze(-1)

        #this is deltaR
        diff_norm = torch.sqrt(phi_diff**2 + eta_diff**2)

        # Create a binary adjacency matrix based on a threshold of 1.7
        adjacency_matrix = torch.where(
            (diff_norm < 1.7) & (phi_diff > 0) & (eta_diff > 0), 
            torch.ones_like(diff_norm), 
            torch.zeros_like(diff_norm)
        )

        # Convert the adjacency matrix to a list of edge indices
        edge_indices = adjacency_matrix.squeeze(-1).nonzero(as_tuple=False).t()


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
            output_features = output_features.to(device)   
            
        # Create a PyTorch Geometric Data object
        data = Data(
            x          = node_features, 
            edge_index = edge_indices, 
            edge_attr  = edge_features,
            y          = output_features
        )
        return data
