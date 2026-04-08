# CGL-DV: Confidence-Guided Dual-View Learning on Text-Attributed Graphs with LLMs

## Overview

**CGL-DV** is a unified framework for node classification on Text-Attributed Graphs (TAGs). It addresses the critical challenge of LLM-generated uncertainty by introducing three novel modules:

1. **CSA** (Confidence-guided Semantic Augmentation) - Generates LLM explanations and pseudo-labels with estimated node-level confidence
2. **DCF** (Dual-view Contrastive Fusion) - Performs contrastive learning on original texts and LLM explanations with confidence-aware fusion
3. **CGP** (Confidence-guided Propagation) - Propagates embeddings with confidence-modulated edge weights

## Performance

| Dataset | CGL-DV | Best Baseline | Improvement |
|---------|--------|---------------|-------------|
| Cora | **90.55%** | 89.95% (TAPE) | +0.60% |
| Citeseer | **85.58%** | 80.72% (TAPE) | +4.86% |
| PubMed | **94.70%** | 93.61% (TAPE) | +1.09% |

## Installation

```bash
# Clone the repository
git clone git@github.com:yummy600/CGL-DV.git
cd CGL-DV

# Create conda environment
conda create -n cgldv python=3.8
conda activate cgldv

# Install dependencies
pip install -r requirements.txt

# Install this package
pip install -e .
```

## Quick Start

```python
from cgldv import CGLDV
from cgldv.data import load_citation_dataset

# Load data
data = load_citation_dataset( 'cora' )

# Initialize model
model = CGLDV(
        num_features = data.num_features,
        num_classes = data.num_classes,
        hidden_dim = 256,
        num_layers = 3
)

# Train
model.fit( data, train_mask = data.train_mask, val_mask = data.val_mask )

# Evaluate
accuracy = model.evaluate( data, test_mask = data.test_mask )
print( f"Test Accuracy: {accuracy:.4f}" )
```

## Key Features

- **Unified Framework**: End-to-end learning on TAGs with LLM integration
- **Confidence Modeling**: Principled mechanism to quantify LLM output reliability
- **Dual-view Learning**: Leverages both original texts and LLM explanations
- **Adaptive Propagation**: Confidence-modulated message passing
- **State-of-the-art Performance**: Consistently outperforms existing methods

## Datasets

CGL-DV is evaluated on three benchmark citation networks:

- **Cora**: 2,708 nodes, 5,429 edges, 7 classes
- **Citeseer**: 3,186 nodes, 4,277 edges, 6 classes
- **PubMed**: 19,717 nodes, 44,338 edges, 3 classes


## Acknowledgments

- LLM generation powered by [LLaMA3.1](https://ai.meta.com/llama/)
- Base language model: [RoBERTa](https://huggingface.co/roberta-base)
