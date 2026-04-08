"""
Data processing module for CGL-DV.
Handles loading and preprocessing citation network datasets.
"""

import os
import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.model_selection import train_test_split


class CitationDataset:

    def __init__( self, name, root = "./data", use_fixed_split = True ):
        self.name = name
        self.root = root

        # Load dataset using PyTorch Geometric
        self.dataset = Planetoid(
                root = root,
                name = name,
                transform = NormalizeFeatures()
        )

        self.data = self.dataset[ 0 ]
        self.num_features = self.dataset.num_features
        self.num_classes = self.dataset.num_classes

        # Store feature and label tensors
        self.x = self.data.x
        self.y = self.data.y
        self.edge_index = self.data.edge_index

        if use_fixed_split:
            self._use_fixed_split()
        else:
            self._default_split()

    def _use_fixed_split( self ):
        pass

    def _default_split( self ):
        num_nodes = self.x.size( 0 )

        # Get labeled nodes
        labeled_nodes = torch.where( self.data.train_mask )[ 0 ].numpy()

        labels = self.y[ labeled_nodes ].numpy()
        min_class_count = min( np.bincount( labels ) )

        if min_class_count >= 6:  # 确保每个类至少有6个样本才能做0.333分割
            train_idx, temp_idx = train_test_split(
                    labeled_nodes,
                    test_size = 0.333,
                    random_state = 42,
                    stratify = labels
            )
            val_idx, test_idx = train_test_split(
                    temp_idx,
                    test_size = 0.5,
                    random_state = 42,
                    stratify = self.y[ temp_idx ].numpy()
            )
        else:
            # Fallback: random split without stratification
            train_idx, temp_idx = train_test_split(
                    labeled_nodes,
                    test_size = 0.333,
                    random_state = 42
            )
            val_idx, test_idx = train_test_split(
                    temp_idx,
                    test_size = 0.5,
                    random_state = 42
            )

        # Create masks
        train_mask = torch.zeros( num_nodes, dtype = torch.bool )
        val_mask = torch.zeros( num_nodes, dtype = torch.bool )
        test_mask = torch.zeros( num_nodes, dtype = torch.bool )

        train_mask[ train_idx ] = True
        val_mask[ val_idx ] = True
        test_mask[ test_idx ] = True

        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.data.test_mask = test_mask

    def get_split_masks( self ):
        return self.data.train_mask, self.data.val_mask, self.data.test_mask

    def get_labeled_unlabeled_split( self, label_rate = 0.6 ):
        """
        Split nodes by label rate for semi-supervised setting.
        Returns new masks without modifying the original fixed split.
        """
        num_nodes = self.x.size( 0 )

        # Get nodes with labels from original split
        original_train_nodes = torch.where( self.data.train_mask )[ 0 ].numpy()

        # Sample from labeled nodes
        labeled_idx, _ = train_test_split(
                original_train_nodes,
                train_size = int( label_rate * len( original_train_nodes ) ),
                random_state = 42,
                stratify = self.y[ original_train_nodes ].numpy()
        )

        # Create new train mask (subset of original training nodes)
        train_mask = torch.zeros( num_nodes, dtype = torch.bool )
        train_mask[ labeled_idx ] = True

        # Keep val and test masks unchanged
        return train_mask, self.data.val_mask, self.data.test_mask

    def get_neighbors( self, node_idx, k = 2 ):
        """
        Get top-k similar neighbors for a given node.
        Uses cosine similarity on node features.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        node_feat = self.x[ node_idx ].unsqueeze( 0 )
        similarities = cosine_similarity( node_feat, self.x.cpu().numpy() )[ 0 ]

        # Exclude self
        similarities[ node_idx ] = -1

        # Get top-k
        top_k_idx = torch.topk( torch.tensor( similarities ), k = k ).indices

        return top_k_idx.tolist()

    def get_subgraph( self, node_indices ):
        """Extract subgraph containing specified nodes."""
        import torch_geometric.utils as pyg_utils

        subgraph_edge_index, _ = pyg_utils.subgraph(
                node_indices,
                self.edge_index,
                relabel_nodes = True
        )

        return subgraph_edge_index

    def __repr__( self ):
        return (f"{self.name.upper()} Dataset(\n"
                f"  nodes: {self.x.size( 0 )},\n"
                f"  edges: {self.edge_index.size( 1 )},\n"
                f"  features: {self.num_features},\n"
                f"  classes: {self.num_classes},\n"
                f"  train: {self.data.train_mask.sum().item()},\n"
                f"  val: {self.data.val_mask.sum().item()},\n"
                f"  test: {self.data.test_mask.sum().item()}\n"
                f")")


def load_citation_dataset( name, root = "./data", use_fixed_split = True ):
    valid_names = [ 'cora', 'citeseer', 'pubmed' ]
    name = name.lower()

    if name not in valid_names:
        raise ValueError( f"Dataset must be one of {valid_names}, got '{name}'" )

    return CitationDataset( name = name, root = root, use_fixed_split = use_fixed_split )
