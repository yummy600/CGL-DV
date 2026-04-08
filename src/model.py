"""
Main CGL-DV Model.
Unified framework for node classification on Text-Attributed Graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

from .csa import CSAModule
from .dcf import DualViewContrastiveFusion
from .cgp import FullCGPModule


class CGLDV( nn.Module ):

    def __init__(
            self,
            num_features: int,
            num_classes: int,
            hidden_dim: int = 256,
            num_layers: int = 3,
            dropout: float = 0.5,
            temperature: float = 0.07,
            embedding_model: str = "roberta-base",
            llm_model: str = "llama3.1",
            device: str = "cuda",
            num_neighbors: int = 2,
            use_llm: bool = True,
            use_graphcl: bool = True,
            decay_rate: float = 1.0,
            adaptive_aggregation: bool = True
    ):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_llm = use_llm

        # CSA Module
        self.csa = CSAModule(
                num_features = num_features,
                num_classes = num_classes,
                hidden_dim = hidden_dim,
                embedding_model = embedding_model,
                llm_model = llm_model,
                device = device,
                num_neighbors = num_neighbors
        )

        # DCF Module
        self.dcf = DualViewContrastiveFusion(
                hidden_dim = hidden_dim,
                num_classes = num_classes,
                temperature = temperature,
                use_graphcl = use_graphcl
        )

        # CGP Module
        self.cgp = FullCGPModule(
                hidden_dim = hidden_dim,
                num_classes = num_classes,
                num_layers = num_layers,
                dropout = dropout,
                decay_rate = decay_rate,
                adaptive_aggregation = adaptive_aggregation
        )

        # Feature transformation
        self.feature_transform = nn.Linear( num_features, hidden_dim )

        # LLM projection
        if use_llm:
            self.llm_projection = nn.Sequential(
                    nn.Linear( 768, hidden_dim ),  # RoBERTa embedding dim
                    nn.ReLU(),
                    nn.Linear( hidden_dim, hidden_dim )
            )

    def set_dataset( self, dataset_name: str ):
        self.csa.set_dataset( dataset_name )

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            texts: Optional[ list ] = None,
            train_mask: Optional[ torch.Tensor ] = None,
            pseudo_labels: Optional[ torch.Tensor ] = None,
            confidences: Optional[ torch.Tensor ] = None,
            use_llm: bool = None
    ) -> Dict[ str, torch.Tensor ]:
        use_llm = use_llm if use_llm is not None else self.use_llm
        num_nodes = x.size( 0 )

        # Transform input features
        h_features = self.feature_transform( x )

        # CSA: Generate augmentations if texts provided
        if use_llm and texts is not None:
            csa_out = self.csa(
                    x, texts, use_llm = True
            )

            # Project LLM embeddings
            h_text = self.llm_projection( csa_out[ 'text_embeddings' ] )
            h_expl = csa_out.get( 'explanation_embeddings', None )
            if h_expl is not None:
                h_expl = self.llm_projection( h_expl )

            # Get labels and confidences
            if pseudo_labels is None:
                pseudo_labels = csa_out.get( 'pseudo_labels' )
            if confidences is None:
                confidences = csa_out.get( 'confidences' )

            # Use fused features from CSA
            h_features = csa_out[ 'features' ]
        else:
            h_text = h_features
            h_expl = h_features

        # Default confidences if not provided
        if confidences is None:
            confidences = torch.ones( num_nodes, device = x.device ) * 0.5

        # Labels for training
        if pseudo_labels is None:
            pseudo_labels = torch.zeros( num_nodes, dtype = torch.long, device = x.device )

        # DCF: Dual-view contrastive fusion
        if use_llm and h_expl is not None:
            dcf_out = self.dcf(
                    h_text,
                    h_expl,
                    pseudo_labels,
                    confidences,
                    edge_index,
                    train_mask
            )
            h_fused = dcf_out[ 'h_fused' ]
        else:
            h_fused = h_features

        # CGP: Confidence-guided propagation
        cgp_out = self.cgp(
                h_fused,
                edge_index,
                confidences,
                train_mask
        )

        return {
            'logits': cgp_out[ 'logits' ],
            'probs': cgp_out[ 'probs' ],
            'embeddings': cgp_out[ 'embeddings' ],
            'pseudo_labels': pseudo_labels,
            'confidences': confidences,
            'h_features': h_features,
            'h_fused': h_fused,
            'layer_outputs': cgp_out[ 'layer_outputs' ]
        }

    def get_embeddings( self, x: torch.Tensor, edge_index: torch.Tensor ) -> torch.Tensor:
        h = self.feature_transform( x )
        return h


class CGLDVTrainer:

    def __init__(
            self,
            model: CGLDV,
            optimizer: torch.optim.Optimizer,
            device: str = "cuda",
            scheduler: Optional[ torch.optim.lr_scheduler._LRScheduler ] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.model.to( device )

        # Training history
        self.history = {
            'train_loss': [ ],
            'train_acc': [ ],
            'val_loss': [ ],
            'val_acc': [ ],
            'test_acc': [ ]
        }

    def train_epoch(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            labels: torch.Tensor,
            train_mask: torch.Tensor,
            texts: Optional[ list ] = None
    ) -> Dict[ str, float ]:
        self.model.train()

        x = x.to( self.device )
        edge_index = edge_index.to( self.device )
        labels = labels.to( self.device )
        train_mask = train_mask.to( self.device )

        self.optimizer.zero_grad()

        # Forward pass
        forward_out = self.model(
                x, edge_index,
                texts = texts,
                train_mask = train_mask
        )

        # Classification loss
        logits = forward_out[ 'logits' ]
        loss_cls = F.cross_entropy( logits[ train_mask ], labels[ train_mask ] )

        # Total loss
        loss = loss_cls

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_( self.model.parameters(), 1.0 )
        self.optimizer.step()

        # Compute accuracy
        pred = logits.argmax( dim = 1 )
        acc = (pred[ train_mask ] == labels[ train_mask ]).float().mean().item()

        return {
            'loss': loss.item(),
            'accuracy': acc
        }

    @torch.no_grad()
    def evaluate(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            labels: torch.Tensor,
            mask: torch.Tensor,
            texts: Optional[ list ] = None
    ) -> Dict[ str, float ]:
        self.model.eval()

        x = x.to( self.device )
        edge_index = edge_index.to( self.device )
        labels = labels.to( self.device )
        mask = mask.to( self.device )

        # Forward pass
        forward_out = self.model(
                x, edge_index,
                texts = texts,
                train_mask = mask
        )

        # Compute loss
        logits = forward_out[ 'logits' ]
        loss = F.cross_entropy( logits[ mask ], labels[ mask ] ).item()

        # Compute accuracy
        pred = logits.argmax( dim = 1 )
        acc = (pred[ mask ] == labels[ mask ]).float().mean().item()

        return {
            'loss': loss,
            'accuracy': acc
        }

    def fit(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            labels: torch.Tensor,
            train_mask: torch.Tensor,
            val_mask: torch.Tensor,
            test_mask: torch.Tensor,
            texts: Optional[ list ] = None,
            epochs: int = 100,
            early_stopping_patience: int = 20
    ) -> Dict[ str, float ]:
        best_val_acc = 0
        best_test_acc = 0
        patience_counter = 0

        for epoch in range( epochs ):
            # Train
            train_metrics = self.train_epoch(
                    x, edge_index, labels, train_mask, texts
            )

            # Evaluate
            val_metrics = self.evaluate(
                    x, edge_index, labels, val_mask, texts
            )
            test_metrics = self.evaluate(
                    x, edge_index, labels, test_mask, texts
            )

            # Update history
            self.history[ 'train_loss' ].append( train_metrics[ 'loss' ] )
            self.history[ 'train_acc' ].append( train_metrics[ 'accuracy' ] )
            self.history[ 'val_loss' ].append( val_metrics[ 'loss' ] )
            self.history[ 'val_acc' ].append( val_metrics[ 'accuracy' ] )
            self.history[ 'test_acc' ].append( test_metrics[ 'accuracy' ] )

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print( f"Epoch {epoch + 1}/{epochs}" )
                print( f"  Train Loss: {train_metrics[ 'loss' ]:.4f}, Acc: {train_metrics[ 'accuracy' ]:.4f}" )
                print( f"  Val Loss: {val_metrics[ 'loss' ]:.4f}, Acc: {val_metrics[ 'accuracy' ]:.4f}" )
                print( f"  Test Acc: {test_metrics[ 'accuracy' ]:.4f}" )

            # Early stopping
            if val_metrics[ 'accuracy' ] > best_val_acc:
                best_val_acc = val_metrics[ 'accuracy' ]
                best_test_acc = test_metrics[ 'accuracy' ]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print( f"Early stopping at epoch {epoch + 1}" )
                    break

        print( f"\nBest Val Acc: {best_val_acc:.4f}" )
        print( f"Best Test Acc: {best_test_acc:.4f}" )

        return best_test_acc
