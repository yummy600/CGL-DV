"""
This module propagates embeddings over the graph with confidence-modulated
edge weights and layer-wise adaptive aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class ConfidenceWeightedEdge( nn.Module ):

    def __init__( self, decay_rate: float = 1.0 ):
        super().__init__()
        self.decay_rate = decay_rate

    def forward(
            self,
            edge_index: torch.Tensor,
            confidences: torch.Tensor,
            num_nodes: int
    ) -> Tuple[ torch.Tensor, torch.Tensor ]:
        src, dst = edge_index[ 0 ], edge_index[ 1 ]

        # Get confidence for source and destination nodes
        c_src = confidences[ src ]
        c_dst = confidences[ dst ]

        # Compute confidence difference
        conf_diff = torch.abs( c_src - c_dst )

        # Compute weight: exp(-gamma * |c_i - c_j|)
        edge_weights = torch.exp( -self.decay_rate * conf_diff )

        return edge_index, edge_weights

    def get_normalized_edges(
            self,
            edge_index: torch.Tensor,
            confidences: torch.Tensor,
            num_nodes: int
    ) -> Tuple[ torch.Tensor, torch.Tensor ]:
        edge_index, edge_weights = self.forward( edge_index, confidences, num_nodes )

        # Compute degree normalization
        degree = torch.zeros( num_nodes, device = edge_index.device )
        degree.scatter_add_( 0, edge_index[ 0 ], edge_weights )
        degree = torch.sqrt( degree + 1e-8 )

        # Normalize
        norm_src = degree[ edge_index[ 0 ] ]
        norm_dst = degree[ edge_index[ 1 ] ]
        edge_weights_norm = edge_weights / (norm_src * norm_dst + 1e-8)

        return edge_index, edge_weights_norm


class LayerwiseImportance( nn.Module ):
    def __init__( self, hidden_dim: int ):
        super().__init__()

        self.score_net = nn.Sequential(
                nn.Linear( hidden_dim, hidden_dim ),
                nn.ReLU(),
                nn.Linear( hidden_dim, 1 )
        )

    def forward( self, layer_embeddings: torch.Tensor ) -> torch.Tensor:
        num_layers = len( layer_embeddings )
        num_nodes = layer_embeddings[ 0 ].size( 0 )

        # Stack layers: [num_layers, num_nodes, hidden_dim]
        stacked = torch.stack( layer_embeddings, dim = 0 )

        # Compute scores: [num_layers, num_nodes, 1]
        scores = self.score_net( stacked )
        scores = scores.squeeze( -1 )  # [num_layers, num_nodes]

        # Softmax over layers for each node
        weights = F.softmax( scores, dim = 0 )  # [num_layers, num_nodes]

        return weights

    def aggregate(
            self,
            layer_embeddings: List[ torch.Tensor ],
            weights: Optional[ torch.Tensor ] = None
    ) -> torch.Tensor:
        if weights is None:
            weights = self.forward( layer_embeddings )

        num_layers = len( layer_embeddings )

        # Weight each layer
        aggregated = torch.zeros_like( layer_embeddings[ 0 ] )

        for l in range( num_layers ):
            aggregated += weights[ l ] * layer_embeddings[ l ]

        return aggregated


class MessagePassingLayer( nn.Module ):
    """
    Graph message passing layer with confidence-modulated aggregation.
    """

    def __init__( self, hidden_dim: int ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Message transformation
        self.message_net = nn.Sequential(
                nn.Linear( hidden_dim, hidden_dim ),
                nn.ReLU(),
                nn.Linear( hidden_dim, hidden_dim )
        )

        # Self-loop transformation
        self.self_net = nn.Sequential(
                nn.Linear( hidden_dim, hidden_dim ),
                nn.ReLU(),
                nn.Linear( hidden_dim, hidden_dim )
        )

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weights: Optional[ torch.Tensor ] = None
    ) -> torch.Tensor:
        num_nodes = x.size( 0 )

        # Source and destination nodes
        src, dst = edge_index[ 0 ], edge_index[ 1 ]

        # Transform messages
        messages = self.message_net( x[ src ] )  # [num_edges, hidden_dim]

        # Apply edge weights if provided
        if edge_weights is not None:
            messages = messages * edge_weights.unsqueeze( 1 )

        # Aggregate messages
        aggr = torch.zeros( num_nodes, self.hidden_dim, device = x.device )
        aggr.scatter_add_( 0, dst.unsqueeze( 1 ).expand_as( messages ), messages )

        # Self-loop
        self_loop = self.self_net( x )

        # Combine
        out = aggr + self_loop

        return out


class ConfidenceGuidedPropagation( nn.Module ):

    def __init__(
            self,
            hidden_dim: int,
            num_layers: int = 3,
            dropout: float = 0.5,
            decay_rate: float = 1.0,
            adaptive_aggregation: bool = True
    ):
        """
        Initialize CGP module.
        
        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of propagation layers
            dropout: Dropout rate
            decay_rate: Confidence decay rate for edge weights
            adaptive_aggregation: Whether to use adaptive layer aggregation
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.adaptive_aggregation = adaptive_aggregation

        # Edge weight computation
        self.edge_module = ConfidenceWeightedEdge( decay_rate = decay_rate )

        # Message passing layers
        self.layers = nn.ModuleList( [
            MessagePassingLayer( hidden_dim )
            for _ in range( num_layers )
        ] )

        # Layer importance (if adaptive)
        if adaptive_aggregation:
            self.layer_importance = LayerwiseImportance( hidden_dim )

        self.dropout = nn.Dropout( dropout )

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            confidences: torch.Tensor,
            train_mask: Optional[ torch.Tensor ] = None
    ) -> Dict[ str, torch.Tensor ]:
        num_nodes = x.size( 0 )

        # Compute confidence-weighted edge weights
        _, edge_weights = self.edge_module( edge_index, confidences, num_nodes )

        # Store layer outputs
        layer_outputs = [ ]

        # Multi-layer message passing
        h = x
        for i, layer in enumerate( self.layers ):
            h = layer( h, edge_index, edge_weights )
            h = F.relu( h )
            h = self.dropout( h )
            layer_outputs.append( h )

        # Adaptive aggregation if enabled
        if self.adaptive_aggregation and len( layer_outputs ) > 1:
            layer_weights = self.layer_importance( layer_outputs )
            output = self.layer_importance.aggregate( layer_outputs, layer_weights )
        else:
            # Use last layer output
            output = layer_outputs[ -1 ]
            layer_weights = None

        return {
            'output': output,
            'layer_outputs': layer_outputs,
            'layer_weights': layer_weights
        }


class CGPDecoder( nn.Module ):

    def __init__(
            self,
            hidden_dim: int,
            num_classes: int,
            dropout: float = 0.5
    ):
        super().__init__()

        self.classifier = nn.Sequential(
                nn.Linear( hidden_dim, hidden_dim ),
                nn.ReLU(),
                nn.Dropout( dropout ),
                nn.Linear( hidden_dim, num_classes )
        )

    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        return self.classifier( x )


class FullCGPModule( nn.Module ):

    def __init__(
            self,
            hidden_dim: int,
            num_classes: int,
            num_layers: int = 3,
            dropout: float = 0.5,
            decay_rate: float = 1.0,
            adaptive_aggregation: bool = True
    ):
        super().__init__()

        self.propagation = ConfidenceGuidedPropagation(
                hidden_dim = hidden_dim,
                num_layers = num_layers,
                dropout = dropout,
                decay_rate = decay_rate,
                adaptive_aggregation = adaptive_aggregation
        )

        self.decoder = CGPDecoder(
                hidden_dim = hidden_dim,
                num_classes = num_classes,
                dropout = dropout
        )

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            confidences: torch.Tensor,
            train_mask: Optional[ torch.Tensor ] = None
    ) -> Dict[ str, torch.Tensor ]:
        # Propagation
        propagation_out = self.propagation( x, edge_index, confidences, train_mask )

        # Classification
        logits = self.decoder( propagation_out[ 'output' ] )
        probs = F.softmax( logits, dim = -1 )

        return {
            'logits': logits,
            'probs': probs,
            'embeddings': propagation_out[ 'output' ],
            'layer_outputs': propagation_out[ 'layer_outputs' ],
            'layer_weights': propagation_out[ 'layer_weights' ]
        }
