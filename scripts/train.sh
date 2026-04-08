#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Train on Cora
echo "Training on Cora..."
python train.py \
    --dataset cora \
    --epochs 100 \
    --hidden_dim 256 \
    --num_layers 3 \
    --lr 1e-5 \
    --device cuda \
    --seed 42

# Train on Citeseer
echo "Training on Citeseer..."
python train.py \
    --dataset citeseer \
    --epochs 100 \
    --hidden_dim 256 \
    --num_layers 4 \
    --lr 1e-5 \
    --device cuda \
    --seed 42

# Train on PubMed
echo "Training on PubMed..."
python train.py \
    --dataset pubmed \
    --epochs 100 \
    --hidden_dim 256 \
    --num_layers 3 \
    --lr 1e-5 \
    --device cuda \
    --seed 42

echo "All experiments completed!"
