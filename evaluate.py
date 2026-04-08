"""
Evaluation script for CGL-DV.
"""

import argparse
import torch
import sys
import os
from pathlib import Path

# Add to path
sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from src.model import CGLDV
from src.utils import set_seed, compute_metrics, visualize_embeddings


def parse_args():
    parser = argparse.ArgumentParser( description = 'Evaluate CGL-DV' )

    parser.add_argument( '--dataset', type = str, default = 'cora',
                         choices = [ 'cora', 'citeseer', 'pubmed' ] )
    parser.add_argument( '--checkpoint', type = str, required = True,
                         help = 'Path to checkpoint' )
    parser.add_argument( '--device', type = str, default = 'cuda' )
    parser.add_argument( '--visualize', action = 'store_true',
                         help = 'Save t-SNE visualization' )
    parser.add_argument( '--data_dir', type = str, default = './data' )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed( 42 )

    device = args.device if torch.cuda.is_available() else 'cpu'
    print( f"Using device: {device}" )

    # Load data
    print( f"\nLoading {args.dataset} dataset..." )
    from data.dataset import load_citation_dataset

    dataset = load_citation_dataset( args.dataset, root = args.data_dir )
    data = dataset.data

    # Create model
    model = CGLDV(
            num_features = dataset.num_features,
            num_classes = dataset.num_classes,
            device = device
    )

    # Load checkpoint
    print( f"\nLoading checkpoint from {args.checkpoint}..." )
    checkpoint = torch.load( args.checkpoint, map_location = device )
    model.load_state_dict( checkpoint )
    model.to( device )
    model.eval()

    # Prepare inputs
    x = data.x.to( device )
    edge_index = data.edge_index.to( device )

    # Forward pass
    print( "\nEvaluating..." )
    with torch.no_grad():
        forward_out = model(
                x, edge_index,
                texts = None,
                train_mask = None
        )

    logits = forward_out[ 'logits' ]
    probs = forward_out[ 'probs' ]
    predictions = logits.argmax( dim = 1 )

    # Compute metrics on test set
    print( "\n" + "=" * 50 )
    print( "Test Set Results" )
    print( "=" * 50 )

    test_metrics = compute_metrics(
            predictions[ data.test_mask ],
            data.y[ data.test_mask ]
    )

    for metric, value in test_metrics.items():
        print( f"{metric.capitalize()}: {value:.4f}" )

    print( "=" * 50 )

    # Per-class accuracy
    print( "\nPer-class Accuracy:" )
    for c in range( dataset.num_classes ):
        mask_c = (data.y == c) & data.test_mask
        if mask_c.sum() > 0:
            acc_c = (predictions[ mask_c ] == data.y[ mask_c ]).float().mean()
            print( f"  Class {c}: {acc_c:.4f} ({mask_c.sum().item()} samples)" )

    # Visualization
    if args.visualize:
        print( "\nGenerating t-SNE visualization..." )
        embeddings = forward_out[ 'embeddings' ]

        vis_path = Path( args.checkpoint ).parent / f"{args.dataset}_tsne.png"
        visualize_embeddings( embeddings, data.y, save_path = str( vis_path ) )
        print( f"Visualization saved to {vis_path}" )

    return test_metrics


if __name__ == '__main__':
    main()
