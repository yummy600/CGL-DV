"""
Unit tests for CGL-DV.
"""

import pytest
import torch
import sys
import os

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

from src.model import CGLDV
from src.csa import CSAModule, ConfidenceEstimator
from src.dcf import DualViewContrastiveFusion, IntraViewContrastiveLoss
from src.cgp import ConfidenceGuidedPropagation, FullCGPModule


class TestCGLDV:

    @pytest.fixture
    def model( self ):
        """Create a small model for testing."""
        model = CGLDV(
                num_features = 100,
                num_classes = 7,
                hidden_dim = 64,
                num_layers = 2,
                device = 'cpu'
        )
        return model

    def test_model_initialization( self, model ):
        """Test model can be initialized."""
        assert model is not None
        assert model.num_features == 100
        assert model.num_classes == 7

    def test_forward_pass( self, model ):
        """Test forward pass."""
        num_nodes = 50
        num_edges = 100

        x = torch.randn( num_nodes, 100 )
        edge_index = torch.randint( 0, num_nodes, (2, num_edges) )

        out = model( x, edge_index, texts = None )

        assert 'logits' in out
        assert out[ 'logits' ].shape == (num_nodes, 7)

    def test_confidences( self, model ):
        """Test confidence handling."""
        num_nodes = 50
        num_edges = 100

        x = torch.randn( num_nodes, 100 )
        edge_index = torch.randint( 0, num_nodes, (2, num_edges) )
        confidences = torch.rand( num_nodes )

        out = model( x, edge_index, confidences = confidences )

        assert out[ 'confidences' ].shape == (num_nodes,)


class TestCSAModule:
    """Test cases for CSA module."""

    def test_confidence_estimator( self ):
        """Test confidence estimation."""
        estimator = ConfidenceEstimator( method = 'softmax' )
        logits = torch.randn( 10 )

        conf = estimator.estimate( logits )

        assert 0 <= conf <= 1


class TestDCF:
    """Test cases for DCF module."""

    @pytest.fixture
    def dcf( self ):
        """Create DCF module."""
        return DualViewContrastiveFusion(
                hidden_dim = 64,
                num_classes = 7,
                use_graphcl = False
        )

    def test_forward( self, dcf ):
        """Test DCF forward pass."""
        num_nodes = 50
        hidden_dim = 64
        num_edges = 100
        num_classes = 7

        h_text = torch.randn( num_nodes, hidden_dim )
        h_expl = torch.randn( num_nodes, hidden_dim )
        labels = torch.randint( 0, num_classes, (num_nodes,) )
        confidences = torch.rand( num_nodes )
        edge_index = torch.randint( 0, num_nodes, (2, num_edges) )
        train_mask = torch.zeros( num_nodes, dtype = torch.bool )
        train_mask[ :30 ] = True

        out = dcf( h_text, h_expl, labels, confidences, edge_index, train_mask )

        assert 'loss' in out
        assert 'h_fused' in out
        assert out[ 'h_fused' ].shape == (num_nodes, hidden_dim)


class TestCGP:
    """Test cases for CGP module."""

    @pytest.fixture
    def cgp( self ):
        """Create CGP module."""
        return FullCGPModule(
                hidden_dim = 64,
                num_classes = 7,
                num_layers = 2
        )

    def test_forward( self, cgp ):
        """Test CGP forward pass."""
        num_nodes = 50
        hidden_dim = 64
        num_edges = 100

        x = torch.randn( num_nodes, hidden_dim )
        edge_index = torch.randint( 0, num_nodes, (2, num_edges) )
        confidences = torch.rand( num_nodes )

        out = cgp( x, edge_index, confidences )

        assert 'logits' in out
        assert 'probs' in out
        assert 'embeddings' in out
        assert out[ 'logits' ].shape == (num_nodes, 7)


class TestContrastiveLoss:
    """Test contrastive loss functions."""

    def test_intra_view_loss( self ):
        """Test intra-view contrastive loss."""
        loss_fn = IntraViewContrastiveLoss( temperature = 0.1 )

        z = torch.randn( 20, 64 )
        labels = torch.tensor( [ 0 ] * 5 + [ 1 ] * 5 + [ 2 ] * 5 + [ 3 ] * 5 )
        mask = torch.ones( 20, dtype = torch.bool )

        loss = loss_fn( z, labels, mask )

        assert loss >= 0
        assert not torch.isnan( loss )


if __name__ == '__main__':
    pytest.main( [ __file__, '-v' ] )
