#!/bin/bash

# LLaMA-3 Network Traffic Classification Training Script
# This script runs the complete training pipeline

echo "=== LLaMA-3 Network Traffic Classification Training Pipeline ==="

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null

# Set environment variables for better performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true  # Disable wandb if not needed

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p llama-network-classifier
mkdir -p logs

# Step 1: Data Preprocessing
echo "=== Step 1: Data Preprocessing ==="
echo "Converting CSV data to instruction format..."
# python data_preprocessing.py
echo "That has already done"

# Check if data preprocessing was successful
if [ ! -f "data/train.json" ]; then
    echo "Error: Data preprocessing failed. Please check your CSV file path."
    exit 1
fi

echo "Data preprocessing completed successfully!"

# Step 2: Fine-tuning
echo "=== Step 2: Starting Fine-tuning ==="
echo "This may take several hours depending on your hardware..."

# Run fine-tuning with error handling
python llama_finetuning.py

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "=== Training completed successfully! ==="
    echo "Model saved to: ./llama-network-classifier"
    
    # Step 3: Test inference
    echo "=== Step 3: Testing Inference ==="
    python inference.py
    
    echo "=== Pipeline completed successfully! ==="
    echo "You can now use the fine-tuned model for network traffic classification."
else
    echo "Error: Training failed. Please check the logs for details."
    exit 1
fi