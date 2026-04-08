"""
Prompt construction module for LLM-based node augmentation.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


class PromptBuilder:
    # Default class names for citation networks
    CLASS_TEMPLATES = {
        'cora': [ 'Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                  'Probabilistic_Methods', 'Reinforcement_Learning',
                  'Rule_Learning', 'Theory' ],
        'citeseer': [ 'Agents', 'Artificial_Intelligence', 'Database',
                      'Game_Theory', 'Information_Retrieval', 'Machine_Learning' ],
        'pubmed': [ 'Diabetes_Type_1', 'Diabetes_Type_2', 'Experimental' ]
    }

    def __init__( self, dataset_name: str, num_neighbors: int = 2 ):
        self.dataset_name = dataset_name.lower()
        self.num_neighbors = num_neighbors
        self.class_names = self.CLASS_TEMPLATES.get( self.dataset_name, [ ] )

    def build_classification_prompt(
            self,
            node_text: str,
            neighbor_texts: List[ str ],
            neighbor_labels: List[ str ] = None
    ) -> str:
        # Select top-k similar neighbors
        selected_neighbors = neighbor_texts[ :self.num_neighbors ]

        # Build prompt with structure
        prompt_parts = [
            "You are an expert in scientific paper classification.",
            "",
            "Task: Classify the following paper into one of the given categories.",
            ""
        ]

        # Add class descriptions
        if self.class_names:
            prompt_parts.append( "Available categories:" )
            for i, cls in enumerate( self.class_names ):
                prompt_parts.append( f"  {i + 1}. {cls.replace( '_', ' ' )}" )
            prompt_parts.append( "" )

        # Add neighbor context if available
        if neighbor_labels:
            prompt_parts.append( "Context from similar papers (for reference):" )
            for i, (text, label) in enumerate( zip( selected_neighbors, neighbor_labels[ :self.num_neighbors ] ) ):
                label_text = label.replace( '_', ' ' ) if isinstance( label, str ) else f"Class {label}"
                prompt_parts.append( f"  - [{label_text}]: {text[ :200 ]}..." )
            prompt_parts.append( "" )

        # Add target node
        prompt_parts.append( "Paper to classify:" )
        prompt_parts.append( f"  {node_text[ :500 ]}" )
        prompt_parts.append( "" )

        # Add instruction for structured output
        prompt_parts.append( "Provide your classification in the following JSON format:" )
        prompt_parts.append( '{' )
        prompt_parts.append( '  "label": "<predicted_category>",' )
        prompt_parts.append( '  "explanation": "<brief_explanation_of_why>",' )
        prompt_parts.append( '  "confidence": <score_between_0_and_1>' )
        prompt_parts.append( '}' )

        return "\n".join( prompt_parts )

    def build_generation_prompt(
            self,
            node_text: str,
            neighbor_texts: List[ str ],
            top_k: int = 2
    ) -> str:
        prompt_parts = [
            "You are analyzing a scientific paper.",
            "",
            "Paper content:",
            f"  {node_text[ :500 ]}",
            ""
        ]

        # Add context from neighbors
        if neighbor_texts:
            prompt_parts.append( "Context from related papers:" )
            for text in neighbor_texts[ :top_k ]:
                prompt_parts.append( f"  - {text[ :200 ]}..." )
            prompt_parts.append( "" )

        prompt_parts.append( "Generate a brief explanation (2-3 sentences) summarizing the key points:" )
        prompt_parts.append( 'Provide in JSON format:' )
        prompt_parts.append( '{' )
        prompt_parts.append( '  "summary": "<explanation>",' )
        prompt_parts.append( '  "confidence": <score_between_0_and_1>' )
        prompt_parts.append( '}' )

        return "\n".join( prompt_parts )

    def parse_llm_response( self, response: str ) -> Dict:
        import json
        import re

        result = {
            'label': None,
            'explanation': None,
            'confidence': 0.5
        }

        try:
            # Try JSON parsing first
            # Find JSON block in response
            json_match = re.search( r'\{[^}]+\}', response, re.DOTALL )
            if json_match:
                json_str = json_match.group()
                parsed = json.loads( json_str )
                result.update( parsed )
        except (json.JSONDecodeError, ValueError):
            # Fallback to regex parsing
            label_match = re.search( r'"label"\s*:\s*"([^"]+)"', response )
            if label_match:
                result[ 'label' ] = label_match.group( 1 )

            exp_match = re.search( r'"explanation"\s*:\s*"([^"]+)"', response )
            if exp_match:
                result[ 'explanation' ] = exp_match.group( 1 )

            conf_match = re.search( r'"confidence"\s*:\s*([\d.]+)', response )
            if conf_match:
                result[ 'confidence' ] = float( conf_match.group( 1 ) )

        return result


class NeighborRetriever:

    def __init__( self, embeddings: torch.Tensor, k: int = 2 ):
        self.embeddings = embeddings.cpu().numpy()
        self.k = k
        self._build_index()

    def _build_index( self ):
        import faiss

        dimension = self.embeddings.shape[ 1 ]

        # Use Inner Product index for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP( dimension )

        # Normalize embeddings for cosine similarity
        normalized = self.embeddings / np.linalg.norm(
                self.embeddings, axis = 1, keepdims = True
        )

        self.index.add( normalized.astype( np.float32 ) )

    def retrieve( self, node_idx: int ) -> Tuple[ np.ndarray, np.ndarray ]:
        query = self.embeddings[ node_idx ]
        query = query / np.linalg.norm( query )

        similarities, indices = self.index.search(
                query.reshape( 1, -1 ).astype( np.float32 ),
                self.k + 1  # +1 to exclude self
        )

        # Filter out self
        mask = indices[ 0 ] != node_idx
        neighbor_indices = indices[ 0 ][ mask ][ :self.k ]
        neighbor_sims = similarities[ 0 ][ mask ][ :self.k ]

        return neighbor_indices, neighbor_sims

    def batch_retrieve( self, node_indices: List[ int ] ) -> Dict[ int, Tuple[ np.ndarray, np.ndarray ] ]:
        results = { }
        for idx in node_indices:
            results[ idx ] = self.retrieve( idx )
        return results
