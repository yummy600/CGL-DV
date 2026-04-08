"""
Training script for CGL-DV.
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from src.model import CGLDV, CGLDVTrainer
from src.utils import set_seed, load_config, print_model_summary, EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser( description = 'Train CGL-DV model' )

    # Data
    parser.add_argument( '--dataset', type = str, default = 'cora',
                         choices = [ 'cora', 'citeseer', 'pubmed' ],
                         help = 'Dataset name' )
    parser.add_argument( '--data_dir', type = str, default = './data',
                         help = 'Data directory' )

    # Model
    parser.add_argument( '--hidden_dim', type = int, default = 256,
                         help = 'Hidden dimension' )
    parser.add_argument( '--num_layers', type = int, default = 3,
                         help = 'Number of propagation layers' )
    parser.add_argument( '--dropout', type = float, default = 0.5,
                         help = 'Dropout rate' )
    parser.add_argument( '--temperature', type = float, default = 0.07,
                         help = 'Temperature for contrastive learning' )

    # Training
    parser.add_argument( '--epochs', type = int, default = 100,
                         help = 'Number of training epochs' )
    parser.add_argument( '--batch_size', type = int, default = 256,
                         help = 'Batch size' )
    parser.add_argument( '--lr', type = float, default = 1e-5,
                         help = 'Learning rate' )
    parser.add_argument( '--weight_decay', type = float, default = 1e-4,
                         help = 'Weight decay' )
    parser.add_argument( '--warmup_epochs', type = int, default = 50,
                         help = 'Warmup epochs' )
    parser.add_argument( '--early_stopping', type = int, default = 20,
                         help = 'Early stopping patience' )

    # LLM settings
    parser.add_argument( '--embedding_model', type = str, default = 'roberta-base',
                         help = 'Embedding model name' )
    parser.add_argument( '--llm_model', type = str, default = 'llama3.1',
                         help = 'LLM model name' )
    parser.add_argument( '--num_neighbors', type = int, default = 2,
                         help = 'Number of neighbors for prompt' )
    parser.add_argument( '--use_llm', action = 'store_true',
                         help = 'Use LLM for augmentation' )

    # Other
    parser.add_argument( '--device', type = str, default = 'cuda',
                         help = 'Device to use' )
    parser.add_argument( '--seed', type = int, default = 42,
                         help = 'Random seed' )
    parser.add_argument( '--config', type = str, default = None,
                         help = 'Config file path' )
    parser.add_argument( '--save_dir', type = str, default = './checkpoints',
                         help = 'Save directory' )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    set_seed( args.seed )

    # Create save directory
    save_dir = Path( args.save_dir )
    save_dir.mkdir( parents = True, exist_ok = True )

    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print( f"Using device: {device}" )

    # Load data
    print( f"\nLoading {args.dataset} dataset..." )
    from data.dataset import load_citation_dataset

    dataset = load_citation_dataset( args.dataset, root = args.data_dir )
    data = dataset.data

    print( f"Dataset: {dataset}" )
    print( f"  Train nodes: {data.train_mask.sum().item()}" )
    print( f"  Val nodes: {data.val_mask.sum().item()}" )
    print( f"  Test nodes: {data.test_mask.sum().item()}" )

    # Create model
    print( "\nCreating model..." )
    model = CGLDV(
            num_features = dataset.num_features,
            num_classes = dataset.num_classes,
            hidden_dim = args.hidden_dim,
            num_layers = args.num_layers,
            dropout = args.dropout,
            temperature = args.temperature,
            embedding_model = args.embedding_model,
            llm_model = args.llm_model,
            device = device,
            num_neighbors = args.num_neighbors,
            use_llm = args.use_llm
    )

    # Set dataset for CSA
    model.set_dataset( args.dataset )

    # Print model summary
    print_model_summary( model )

    # Optimizer
    optimizer = optim.Adam(
            model.parameters(),
            lr = args.lr,
            weight_decay = args.weight_decay
    )

    # Learning rate scheduler (linear warmup + linear decay)
    total_steps = args.epochs
    warmup_steps = args.warmup_epochs

    def lr_lambda( step ):
        if step < warmup_steps:
            return float( step ) / float( max( 1, warmup_steps ) )
        return max( 0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps) )

    scheduler = optim.lr_scheduler.LambdaLR( optimizer, lr_lambda )

    # Trainer
    trainer = CGLDVTrainer(
            model = model,
            optimizer = optimizer,
            device = device,
            scheduler = scheduler
    )

    # Prepare texts (convert bag-of-words to text)
    # For citation networks, we can use simple text representation
    texts = None
    if args.use_llm:
        print( "\nPreparing text data for LLM..." )
        # For simplicity, use feature indices as pseudo-text
        # In practice, use actual paper abstracts/titles
        texts = [ f"Node feature vector with {dataset.num_features} dimensions"
                  for _ in range( data.x.size( 0 ) ) ]

    # Training
    print( "\n" + "=" * 50 )
    print( "Starting training..." )
    print( "=" * 50 )

    best_acc = trainer.fit(
            x = data.x,
            edge_index = data.edge_index,
            labels = data.y,
            train_mask = data.train_mask,
            val_mask = data.val_mask,
            test_mask = data.test_mask,
            texts = texts,
            epochs = args.epochs,
            early_stopping_patience = args.early_stopping
    )

    # Save final model
    final_path = save_dir / f"{args.dataset}_final.pt"
    torch.save( model.state_dict(), final_path )
    print( f"\nModel saved to {final_path}" )

    print( "\n" + "=" * 50 )
    print( "Training completed!" )
    print( f"Best test accuracy: {best_acc:.4f}" )
    print( "=" * 50 )


if __name__ == '__main__':
    main()
