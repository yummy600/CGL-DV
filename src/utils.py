"""
Utility functions for CGL-DV.
"""

import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import json
import yaml
from pathlib import Path


def set_seed( seed: int = 42 ):
    """
    Set random seed for reproducibility.
    """
    random.seed( seed )
    np.random.seed( seed )
    torch.manual_seed( seed )
    torch.cuda.manual_seed_all( seed )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy( pred: torch.Tensor, labels: torch.Tensor ) -> float:
    """
    Compute accuracy.
    Returns:
        Accuracy value
    """
    return (pred == labels).float().mean().item()


def compute_metrics(
        pred: torch.Tensor,
        labels: torch.Tensor
) -> Dict[ str, float ]:
    """
    Compute various classification metrics.
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )

    pred_np = pred.cpu().numpy()
    labels_np = labels.cpu().numpy()

    metrics = {
        'accuracy': accuracy_score( labels_np, pred_np ),
        'precision': precision_score( labels_np, pred_np, average = 'macro', zero_division = 0 ),
        'recall': recall_score( labels_np, pred_np, average = 'macro', zero_division = 0 ),
        'f1': f1_score( labels_np, pred_np, average = 'macro', zero_division = 0 )
    }

    return metrics


def load_config( config_path: str ) -> Dict:
    """
    Load configuration from YAML file.
    Returns:
        Configuration dictionary
    """
    with open( config_path, 'r' ) as f:
        config = yaml.safe_load( f )
    return config


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
        path: str
):
    """
    Save model checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save( checkpoint, path )


def load_checkpoint(
        model: torch.nn.Module,
        optimizer: Optional[ torch.optim.Optimizer ],
        path: str
) -> Tuple[ int, Dict ]:
    """
    Load model checkpoint.
    Returns:
        Tuple of (epoch, metrics)
    """
    checkpoint = torch.load( path )

    model.load_state_dict( checkpoint[ 'model_state_dict' ] )

    if optimizer is not None:
        optimizer.load_state_dict( checkpoint[ 'optimizer_state_dict' ] )

    return checkpoint[ 'epoch' ], checkpoint[ 'metrics' ]


def get_dataset_info( dataset_name: str ) -> Dict:
    """
    Get dataset information.
    Returns:
        Dataset information dictionary
    """
    info = {
        'cora': {
            'num_nodes': 2708,
            'num_edges': 5429,
            'num_features': 1433,
            'num_classes': 7,
            'citation': 'McCallum et al., 2000'
        },
        'citeseer': {
            'num_nodes': 3186,
            'num_edges': 4277,
            'num_features': 3703,
            'num_classes': 6,
            'citation': 'Giles et al., 1998'
        },
        'pubmed': {
            'num_nodes': 19717,
            'num_edges': 44338,
            'num_features': 500,
            'num_classes': 3,
            'citation': 'Sen et al., 2008'
        }
    }

    return info.get( dataset_name.lower(), { } )


class EarlyStopping:

    def __init__(
            self,
            patience: int = 10,
            min_delta: float = 0.0,
            mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__( self, score: float ) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def visualize_embeddings(
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        save_path: Optional[ str ] = None
):
    """
    Visualize embeddings using t-SNE.
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # Convert to numpy
    emb_np = embeddings.cpu().detach().numpy()
    labels_np = labels.cpu().numpy()

    # t-SNE
    tsne = TSNE( n_components = 2, random_state = 42 )
    emb_2d = tsne.fit_transform( emb_np )

    # Plot
    plt.figure( figsize = (10, 10) )
    scatter = plt.scatter( emb_2d[ :, 0 ], emb_2d[ :, 1 ], c = labels_np, cmap = 'tab10', s = 5 )
    plt.colorbar( scatter )
    plt.title( 't-SNE Visualization of Node Embeddings' )
    plt.xlabel( 'Dimension 1' )
    plt.ylabel( 'Dimension 2' )

    if save_path:
        plt.savefig( save_path, dpi = 300, bbox_inches = 'tight' )

    plt.close()


def print_model_summary( model: torch.nn.Module ):
    total_params = sum( p.numel() for p in model.parameters() )
    trainable_params = sum( p.numel() for p in model.parameters() if p.requires_grad )

    print( "=" * 50 )
    print( "Model Summary" )
    print( "=" * 50 )
    print( f"Total Parameters: {total_params:,}" )
    print( f"Trainable Parameters: {trainable_params:,}" )
    print( "=" * 50 )
