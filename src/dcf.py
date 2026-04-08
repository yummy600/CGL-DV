"""
This module performs contrastive learning on original texts and LLM explanations,
followed by confidence-aware feature fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class IntraViewContrastiveLoss( nn.Module ):

    def __init__( self, temperature: float = 0.07 ):
        super().__init__()
        self.temperature = temperature

    def forward(
            self,
            z: torch.Tensor,
            labels: torch.Tensor,
            mask: Optional[ torch.Tensor ] = None
    ) -> torch.Tensor:
        if mask is not None:
            z = z[ mask ]
            labels = labels[ mask ]

        # Normalize embeddings
        z = F.normalize( z, dim = 1 )

        # Compute similarity matrix
        sim = torch.matmul( z, z.T ) / self.temperature

        # Create positive mask (same label)
        labels = labels.view( -1, 1 )
        pos_mask = torch.eq( labels, labels.T ).float()

        # Remove self-contrastive
        diag_mask = torch.eye( z.size( 0 ), device = z.device ).bool()
        pos_mask = pos_mask.masked_fill( diag_mask, 0 )

        # Compute loss
        exp_sim = torch.exp( sim )

        # Denominator: all except self
        denom = exp_sim.sum( dim = 1 ) - exp_sim.diag()

        # Numerator: positive pairs
        pos_exp_sim = (exp_sim * pos_mask).sum( dim = 1 )

        # Loss
        loss = -torch.log( pos_exp_sim / denom + 1e-8 )
        return loss.mean()


class ConfidenceWeightedContrastiveLoss( nn.Module ):

    def __init__( self, temperature: float = 0.07 ):
        super().__init__()
        self.temperature = temperature

    def forward(
            self,
            z: torch.Tensor,
            labels: torch.Tensor,
            confidences: torch.Tensor,
            mask: Optional[ torch.Tensor ] = None
    ) -> torch.Tensor:
        if mask is not None:
            z = z[ mask ]
            labels = labels[ mask ]
            confidences = confidences[ mask ]

        # Normalize embeddings
        z = F.normalize( z, dim = 1 )

        # Compute similarity matrix
        sim = torch.matmul( z, z.T ) / self.temperature

        # Create positive mask
        labels = labels.view( -1, 1 )
        pos_mask = torch.eq( labels, labels.T ).float()

        # Remove self-contrastive
        diag_mask = torch.eye( z.size( 0 ), device = z.device ).bool()
        pos_mask = pos_mask.masked_fill( diag_mask, 0 )

        # Compute confidence weights for positive pairs
        # w_ij = c_i * c_j (confidence product)
        conf_i = confidences.view( -1, 1 )
        conf_j = confidences.view( 1, -1 )
        pair_weights = torch.matmul( conf_i, conf_j )
        pair_weights = pair_weights.masked_fill( diag_mask, 0 )

        # Weighted positive contributions
        exp_sim = torch.exp( sim )
        weighted_pos = (exp_sim * pos_mask * pair_weights).sum( dim = 1 )

        # Denominator
        denom = exp_sim.sum( dim = 1 ) - exp_sim.diag()

        # Loss with confidence weighting
        loss = -torch.log( weighted_pos / denom + 1e-8 )
        return loss.mean()


class PrototypeLayer( nn.Module ):

    def __init__( self, hidden_dim: int, num_classes: int, use_confidence: bool = False ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_confidence = use_confidence

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor,
            confidences: Optional[ torch.Tensor ] = None
    ) -> torch.Tensor:
        prototypes = torch.zeros( self.num_classes, self.hidden_dim, device = embeddings.device )
        counts = torch.zeros( self.num_classes, device = embeddings.device )

        for c in range( self.num_classes ):
            mask = (labels == c)
            if mask.sum() > 0:
                if self.use_confidence and confidences is not None:
                    # Confidence-weighted aggregation
                    weights = confidences[ mask ]
                    weights = weights / (weights.sum() + 1e-8)
                    prototypes[ c ] = (embeddings[ mask ] * weights.unsqueeze( 1 )).sum( dim = 0 )
                else:
                    # Simple mean
                    prototypes[ c ] = embeddings[ mask ].mean( dim = 0 )
                counts[ c ] = mask.sum()

        return prototypes


class ConfidenceAwareFusion( nn.Module ):

    def __init__( self, hidden_dim: int ):
        super().__init__()

        self.similarity_net = nn.Sequential(
                nn.Linear( hidden_dim * 2, hidden_dim ),
                nn.ReLU(),
                nn.Linear( hidden_dim, 1 )
        )

    def forward(
            self,
            z_text: torch.Tensor,
            z_expl: torch.Tensor,
            p_text: torch.Tensor,
            p_expl: torch.Tensor
    ) -> Tuple[ torch.Tensor, torch.Tensor ]:
        # Compute similarity to prototypes
        sim_text = torch.matmul( z_text, p_text.T )  # [num_nodes, num_classes]
        sim_expl = torch.matmul( z_expl, p_expl.T )  # [num_nodes, num_classes]

        # Average similarity per node
        avg_sim_text = sim_text.mean( dim = 1 )
        avg_sim_expl = sim_expl.mean( dim = 1 )

        # Compute fusion weight based on similarity difference
        # alpha = sigmoid(sim_text - sim_expl)
        alpha = torch.sigmoid( avg_sim_text - avg_sim_expl )
        alpha = alpha.unsqueeze( 1 )  # [num_nodes, 1]

        # Fuse embeddings
        h_fused = alpha * z_text + (1 - alpha) * z_expl

        return h_fused, alpha


class GraphCLAugmentation:

    def __init__(
            self,
            feature_dropout: float = 0.2,
            edge_dropout: float = 0.1
    ):
        self.feature_dropout = feature_dropout
        self.edge_dropout = edge_dropout

    def augment_features(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        mask = torch.rand_like( x ) > self.feature_dropout
        return x * mask.float()

    def augment_edges(
            self,
            edge_index: torch.Tensor,
            num_nodes: int
    ) -> torch.Tensor:
        keep_prob = 1 - self.edge_dropout
        keep_mask = torch.rand( edge_index.size( 1 ) ) < keep_prob
        return edge_index[ :, keep_mask ]


class DualViewContrastiveFusion( nn.Module ):

    def __init__(
            self,
            hidden_dim: int,
            num_classes: int,
            temperature: float = 0.07,
            feature_dropout: float = 0.2,
            edge_dropout: float = 0.1,
            use_graphcl: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.use_graphcl = use_graphcl

        # Loss functions
        self.text_loss_fn = IntraViewContrastiveLoss( temperature )
        self.expl_loss_fn = ConfidenceWeightedContrastiveLoss( temperature )

        # Prototype layers
        self.text_prototype = PrototypeLayer(
                hidden_dim, num_classes, use_confidence = False
        )
        self.expl_prototype = PrototypeLayer(
                hidden_dim, num_classes, use_confidence = True
        )

        # Fusion layer
        self.fusion_layer = ConfidenceAwareFusion( hidden_dim )

        # Augmentation
        self.augmentation = GraphCLAugmentation(
                feature_dropout = feature_dropout,
                edge_dropout = edge_dropout
        )

        # Projection heads for contrastive learning
        self.text_projection = nn.Sequential(
                nn.Linear( hidden_dim, hidden_dim ),
                nn.ReLU(),
                nn.Linear( hidden_dim, hidden_dim )
        )

        self.expl_projection = nn.Sequential(
                nn.Linear( hidden_dim, hidden_dim ),
                nn.ReLU(),
                nn.Linear( hidden_dim, hidden_dim )
        )

    def forward(
            self,
            h_text: torch.Tensor,
            h_expl: torch.Tensor,
            labels: torch.Tensor,
            confidences: torch.Tensor,
            edge_index: torch.Tensor,
            train_mask: torch.Tensor
    ) -> Dict[ str, torch.Tensor ]:
        # Apply GraphCL augmentation if enabled
        if self.use_graphcl:
            # Augment text view
            h_text_aug1 = self.augmentation.augment_features( h_text )
            h_text_aug2 = self.augmentation.augment_features( h_text )

            # Augment explanation view
            h_expl_aug1 = self.augmentation.augment_features( h_expl )
            h_expl_aug2 = self.augmentation.augment_features( h_expl )
        else:
            h_text_aug1 = h_text_aug2 = h_text
            h_expl_aug1 = h_expl_aug2 = h_expl

        # Project embeddings
        z_text = self.text_projection( h_text )
        z_expl = self.expl_projection( h_expl )

        # Compute intra-view losses
        loss_text = self.text_loss_fn( z_text, labels, train_mask )
        loss_expl = self.expl_loss_fn( z_expl, labels, confidences, train_mask )

        # Compute prototypes
        p_text = self.text_prototype( z_text, labels )
        p_expl = self.expl_prototype( z_expl, labels, confidences )

        # Compute fused embeddings
        h_fused, fusion_weights = self.fusion_layer(
                h_text, h_expl, p_text, p_expl
        )

        # Total loss
        total_loss = loss_text + loss_expl

        return {
            'loss': total_loss,
            'loss_text': loss_text,
            'loss_expl': loss_expl,
            'h_fused': h_fused,
            'fusion_weights': fusion_weights,
            'prototypes': {
                'text': p_text,
                'expl': p_expl
            }
        }


def create_dcf_loss(
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: torch.Tensor,
        confidences: torch.Tensor,
        temperature: float = 0.07,
        mask: Optional[ torch.Tensor ] = None
) -> torch.Tensor:
    # Normalize
    z1 = F.normalize( z1, dim = 1 )
    z2 = F.normalize( z2, dim = 1 )

    # Text loss (supervised)
    text_loss_fn = IntraViewContrastiveLoss( temperature )
    loss_text = text_loss_fn( z1, labels, mask )

    # Explanation loss (confidence-weighted)
    expl_loss_fn = ConfidenceWeightedContrastiveLoss( temperature )
    loss_expl = expl_loss_fn( z2, labels, confidences, mask )

    return loss_text + loss_expl
