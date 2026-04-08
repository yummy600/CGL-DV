"""
This module generates LLM explanations and pseudo-labels with estimated
node-level confidence scores.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple, List
import numpy as np
from .prompt import PromptBuilder, NeighborRetriever


class ConfidenceEstimator:

    def __init__( self, method: str = "self_consistency" ):
        self.method = method

    def estimate_from_softmax( self, logits: torch.Tensor ) -> float:
        probs = torch.softmax( logits, dim = -1 )
        max_prob = probs.max().item()
        return max_prob

    def estimate_from_entropy( self, logits: torch.Tensor ) -> float:
        probs = torch.softmax( logits, dim = -1 )
        entropy = -torch.sum( probs * torch.log( probs + 1e-10 ), dim = -1 )
        max_entropy = torch.log( torch.tensor( probs.shape[ -1 ] ) )
        confidence = 1 - (entropy / max_entropy)
        return confidence.mean().item()

    def estimate_from_self_consistency(
            self,
            responses: List[ str ],
            labels: List[ int ] = None
    ) -> float:
        if labels is None or len( labels ) == 0:
            return 0.5  # Default confidence

        # Count label agreement
        from collections import Counter
        label_counts = Counter( labels )
        most_common_count = label_counts.most_common( 1 )[ 0 ][ 1 ]

        # Confidence = agreement ratio
        confidence = most_common_count / len( labels )
        return confidence

    def estimate( self, logits: Optional[ torch.Tensor ] = None, **kwargs ) -> float:
        if self.method == "softmax":
            return self.estimate_from_softmax( logits )
        elif self.method == "entropy":
            return self.estimate_from_entropy( logits )
        elif self.method == "self_consistency":
            return self.estimate_from_self_consistency(
                    kwargs.get( 'responses', [ ] ),
                    kwargs.get( 'labels', None )
            )
        else:
            return 0.5


class LLMGenerator:

    def __init__(
            self,
            model_name: str = "llama3.1",
            device: str = "cuda",
            use_cache: bool = True,
            cache_dir: str = "./cache"
    ):
        self.model_name = model_name
        self.device = device
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.cache = { }

        # Initialize Ollama if available
        self._init_ollama()

    def _init_ollama( self ):
        import requests

        self.ollama_url = "http://localhost:11434"
        self._available = self._check_ollama()

        if self._available:
            print( f"Connected to Ollama with model: {self.model_name}" )
        else:
            print( "Warning: Ollama not available. Will use mock responses." )

    def _check_ollama( self ) -> bool:
        """Check if Ollama is running and accessible."""
        import requests
        try:
            response = requests.get( f"{self.ollama_url}/api/tags", timeout = 2 )
            return response.status_code == 200
        except:
            return False

    def generate(
            self,
            prompt: str,
            max_tokens: int = 256,
            temperature: float = 0.7,
            cache_key: Optional[ str ] = None
    ) -> str:
        # Check cache
        if self.use_cache and cache_key and cache_key in self.cache:
            return self.cache[ cache_key ]

        if self._available:
            response = self._generate_ollama(
                    prompt, max_tokens, temperature
            )
        else:
            # Mock response for testing
            response = self._mock_generate( prompt )

        # Store in cache
        if self.use_cache and cache_key:
            self.cache[ cache_key ] = response

        return response

    def _generate_ollama(
            self,
            prompt: str,
            max_tokens: int,
            temperature: float
    ) -> str:
        """Generate using Ollama API."""
        import requests
        import json

        try:
            response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature
                        }
                    },
                    timeout = 60
            )

            if response.status_code == 200:
                return response.json().get( "response", "" )
            else:
                return self._mock_generate( prompt )

        except Exception as e:
            print( f"Ollama generation failed: {e}" )
            return self._mock_generate( prompt )

    def _mock_generate( self, prompt: str ) -> str:
        import random

        # Simulate response structure
        labels = [ "Neural_Networks", "Machine_Learning", "Data_Mining" ]
        explanations = [
            "This paper presents novel methods in the field.",
            "The authors propose an innovative approach.",
            "Experimental results demonstrate effectiveness."
        ]

        label = random.choice( labels )
        explanation = random.choice( explanations )
        confidence = random.uniform( 0.5, 0.95 )

        return f'{{"label": "{label}", "explanation": "{explanation}", "confidence": {confidence:.2f}}}'

    def batch_generate(
            self,
            prompts: List[ str ],
            batch_size: int = 8
    ) -> List[ str ]:
        results = [ ]
        for i in range( 0, len( prompts ), batch_size ):
            batch = prompts[ i:i + batch_size ]
            batch_results = [ self.generate( p, cache_key = f"prompt_{j}" )
                              for j, p in enumerate( batch ) ]
            results.extend( batch_results )
        return results


class SemanticAugmenter( nn.Module ):

    def __init__(
            self,
            embedding_model: str = "roberta-base",
            llm_model: str = "llama3.1",
            embedding_dim: int = 768,
            num_classes: int = 7,
            device: str = "cuda",
            num_neighbors: int = 2
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_neighbors = num_neighbors
        self.device = device

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained( embedding_model )
        self.tokenizer = AutoTokenizer.from_pretrained( embedding_model )

        # Projection heads
        self.text_projection = nn.Linear( embedding_dim, embedding_dim )
        self.explanation_projection = nn.Linear( embedding_dim, embedding_dim )

        # LLM generator
        self.llm_generator = LLMGenerator(
                model_name = llm_model,
                device = device
        )

        # Confidence estimator
        self.confidence_estimator = ConfidenceEstimator( method = "softmax" )

        # Prompt builder
        self.prompt_builder = None  # Initialized with dataset info

    def set_prompt_builder( self, dataset_name: str ):
        self.prompt_builder = PromptBuilder(
                dataset_name = dataset_name,
                num_neighbors = self.num_neighbors
        )

    def encode_text( self, texts: List[ str ] ) -> torch.Tensor:
        inputs = self.tokenizer(
                texts,
                padding = True,
                truncation = True,
                max_length = 512,
                return_tensors = "pt"
        ).to( self.device )

        with torch.no_grad():
            outputs = self.text_encoder( **inputs )
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[ :, 0, : ]

        return embeddings

    def generate_augmentations(
            self,
            node_texts: List[ str ],
            neighbor_texts: List[ List[ str ] ] = None,
            neighbor_labels: List[ List[ int ] ] = None
    ) -> Dict[ str, torch.Tensor ]:
        num_nodes = len( node_texts )

        # Build prompts
        prompts = [ ]
        for i, text in enumerate( node_texts ):
            neighbors = neighbor_texts[ i ] if neighbor_texts else [ ]
            labels = neighbor_labels[ i ] if neighbor_labels else None

            prompt = self.prompt_builder.build_classification_prompt(
                    node_text = text,
                    neighbor_texts = neighbors,
                    neighbor_labels = labels
            )
            prompts.append( prompt )

        # Generate with LLM
        responses = [ ]
        for i, prompt in enumerate( prompts ):
            response = self.llm_generator.generate(
                    prompt,
                    cache_key = f"node_{i}"
            )
            responses.append( response )

        # Parse responses
        pseudo_labels = [ ]
        explanations = [ ]
        confidences = [ ]

        for response in responses:
            parsed = self.prompt_builder.parse_llm_response( response )

            # Convert label string to index
            if parsed[ 'label' ] and self.prompt_builder.class_names:
                try:
                    label_idx = self.prompt_builder.class_names.index( parsed[ 'label' ] )
                except ValueError:
                    label_idx = 0  # Default
            else:
                label_idx = 0

            pseudo_labels.append( label_idx )
            explanations.append( parsed.get( 'explanation', '' ) )
            confidences.append( parsed.get( 'confidence', 0.5 ) )

        # Encode explanations
        explanation_embeddings = self.encode_text( explanations )
        explanation_embeddings = self.explanation_projection( explanation_embeddings )

        return {
            'pseudo_labels': torch.tensor( pseudo_labels, dtype = torch.long ),
            'explanations': explanations,
            'confidences': torch.tensor( confidences, dtype = torch.float32 ),
            'explanation_embeddings': explanation_embeddings
        }

    def forward(
            self,
            texts: List[ str ],
            neighbor_texts: List[ List[ str ] ] = None
    ) -> Dict[ str, torch.Tensor ]:
        # Encode original texts
        text_embeddings = self.encode_text( texts )
        text_embeddings = self.text_projection( text_embeddings )

        result = {
            'text_embeddings': text_embeddings
        }

        # Generate augmentations if prompt builder is set
        if self.prompt_builder is not None:
            augmentations = self.generate_augmentations(
                    texts, neighbor_texts
            )
            result.update( augmentations )

        return result


class CSAModule( nn.Module ):

    def __init__(
            self,
            num_features: int,
            num_classes: int,
            hidden_dim: int = 256,
            embedding_model: str = "roberta-base",
            llm_model: str = "llama3.1",
            device: str = "cuda",
            num_neighbors: int = 2
    ):
        super().__init__()

        self.semantic_augmenter = SemanticAugmenter(
                embedding_model = embedding_model,
                llm_model = llm_model,
                embedding_dim = 768,
                num_classes = num_classes,
                device = device,
                num_neighbors = num_neighbors
        )

        # Feature transformation
        self.feature_transform = nn.Linear( num_features, hidden_dim )

        # Fusion layer
        self.fusion_layer = nn.Linear( hidden_dim + 768, hidden_dim )

        self.hidden_dim = hidden_dim

    def set_dataset( self, dataset_name: str ):
        self.semantic_augmenter.set_prompt_builder( dataset_name )

    def forward(
            self,
            x: torch.Tensor,
            texts: List[ str ] = None,
            neighbor_texts: List[ List[ str ] ] = None,
            use_llm: bool = False
    ) -> Dict[ str, torch.Tensor ]:
        # Transform input features
        h_features = torch.relu( self.feature_transform( x ) )

        result = {
            'features': h_features,
            'text_embeddings': None,
            'pseudo_labels': None,
            'confidences': None,
            'explanation_embeddings': None
        }

        # Generate text embeddings and augmentations
        if texts is not None:
            augmentations = self.semantic_augmenter(
                    texts, neighbor_texts
            )

            result[ 'text_embeddings' ] = augmentations[ 'text_embeddings' ]

            if use_llm:
                # Fuse feature and semantic embeddings
                h_combined = torch.cat( [ h_features, augmentations[ 'text_embeddings' ] ], dim = -1 )
                h_fused = torch.relu( self.fusion_layer( h_combined ) )

                result[ 'features' ] = h_fused
                result[ 'pseudo_labels' ] = augmentations[ 'pseudo_labels' ]
                result[ 'confidences' ] = augmentations[ 'confidences' ]
                result[ 'explanation_embeddings' ] = augmentations[ 'explanation_embeddings' ]

        return result
