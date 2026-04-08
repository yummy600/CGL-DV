"""
Loss functions for CGL-DV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss( nn.Module ):

    def __init__(
            self,
            alpha: float = 1.0,
            beta: float = 0.1,
            temperature: float = 0.07
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            z1: torch.Tensor,
            z2: torch.Tensor,
            confidences: torch.Tensor,
            mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Classification loss
        if mask is not None:
            loss_cls = F.cross_entropy( logits[ mask ], labels[ mask ] )
        else:
            loss_cls = F.cross_entropy( logits, labels )

        # Contrastive loss (simplified)
        loss_contrast = self._contrastive_loss( z1, z2, labels, confidences, mask )

        # Combined
        loss = self.alpha * loss_cls + self.beta * loss_contrast

        return loss

    def _contrastive_loss(
            self,
            z1: torch.Tensor,
            z2: torch.Tensor,
            labels: torch.Tensor,
            confidences: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        # Simple NT-Xent style loss
        z1 = F.normalize( z1, dim = 1 )
        z2 = F.normalize( z2, dim = 1 )

        sim = torch.matmul( z1, z2.T ) / self.temperature

        if mask is not None:
            sim = sim[ mask ][ :, mask ]

        exp_sim = torch.exp( sim )

        # Positive pairs (diagonal)
        pos_sim = exp_sim.diag()

        # Negative pairs
        denom = exp_sim.sum( dim = 1 ) - exp_sim.diag()

        loss = -torch.log( pos_sim / (denom + 1e-8) + 1e-8 ).mean()

        return loss


class ConfidenceRegularizationLoss( nn.Module ):

    def __init__( self, target_mean: float = 0.7 ):
        super().__init__()
        self.target_mean = target_mean

    def forward( self, confidences: torch.Tensor, mask: torch.Tensor = None ) -> torch.Tensor:
        if mask is not None:
            c = confidences[ mask ]
        else:
            c = confidences

        # Penalize deviation from target mean
        mean_conf = c.mean()
        loss = (mean_conf - self.target_mean) ** 2

        return loss


class DiversityLoss( nn.Module ):

    def __init__( self, hidden_dim: int ):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor,
            mask: torch.Tensor = None
    ) -> torch.Tensor:
        if mask is not None:
            embeddings = embeddings[ mask ]
            labels = labels[ mask ]

        # Compute class means
        unique_labels = torch.unique( labels )
        class_means = [ ]

        for c in unique_labels:
            mask_c = labels == c
            mean_c = embeddings[ mask_c ].mean( dim = 0 )
            class_means.append( mean_c )

        class_means = torch.stack( class_means )  # [num_classes, hidden_dim]

        # Compute variance (higher = more diverse)
        variance = class_means.var( dim = 0 ).mean()

        # We want to MAXIMIZE variance, so minimize negative variance
        return -variance
