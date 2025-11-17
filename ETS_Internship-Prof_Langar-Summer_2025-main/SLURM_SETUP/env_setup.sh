#!/bin/bash

# Exit the virtual environment if you're currently in one
deactivate

# Load the appropriate Compute Canada modules
module load python/3.11
module load scipy-stack
module load gcc/13.3  # or another available version (use 'module spider [module_name]' to check available versions)
module load cuda/12.6  # if you want the GPU version
module load pytorch/2.1

# Create a new virtual environment using Compute Canada's Python
virtualenv --no-download [venv_name]
source [venv_name]/bin/activate # Activate the virtual environment

# Install PyTorch in your virtual environment with pip
# If you already have a virtual environment (like sth_venv), activate it, then install PyTorch via pip:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies (excluding torch, which is already available through the module)
pip install --no-index pandas scikit-learn # pip install --no-index [module_name]
