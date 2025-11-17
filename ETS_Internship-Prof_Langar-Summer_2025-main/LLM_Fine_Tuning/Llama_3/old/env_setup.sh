#!/bin/bash
# =============================================================================
# Environment Setup Script for Llama-3 Fine-tuning
# =============================================================================
#
# This script sets up the Python environment and installs all required 
# dependencies for fine-tuning Llama-3 on the CIC IoT 2023 dataset.
#
# USAGE:
#   bash env_setup.sh
#
# REQUIREMENTS:
#   - Access to Compute Canada/Digital Research Alliance modules
#   - Internet connection for pip installations
#   - Sufficient disk space for packages (~2-3 GB)
#
# =============================================================================

set -e  # Exit on any error

echo "Setting up environment for Llama-3 fine-tuning..."

# Exit any current virtual environment
echo "Deactivating any existing virtual environment..."
deactivate 2>/dev/null || true

# Load required Compute Canada modules
echo "Loading Compute Canada modules..."
module load python/3.11
module load scipy-stack
module load gcc/13.3
module load cuda/12.6

echo "Modules loaded successfully"

# Display loaded modules for verification
echo "Currently loaded modules:"
module list

# Create virtual environment
VENV_NAME="masterenv"
echo "Creating virtual environment: $VENV_NAME"

if [ -d "$VENV_NAME" ]; then
    echo "⚠️  Virtual environment '$VENV_NAME' already exists. Removing..."
    rm -rf "$VENV_NAME"
fi

virtualenv --no-download "$VENV_NAME"
source "$VENV_NAME/bin/activate"

echo "Virtual environment created and activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.6 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core data science libraries
echo "Installing core data science libraries..."
pip install --no-index --upgrade pip
pip install --no-index pandas scikit-learn

# Install transformers and related libraries
echo "Installing Hugging Face libraries..."
pip install transformers>=4.36.0
pip install datasets
pip install tokenizers
pip install accelerate

# Install PEFT for LoRA fine-tuning
echo "Installing PEFT (Parameter-Efficient Fine-Tuning)..."
pip install peft

# Install BitsAndBytes for quantization
echo "Installing BitsAndBytes for 4-bit quantization..."
pip install bitsandbytes

# Install additional utilities
echo "Installing additional utilities..."
pip install evaluate
pip install tqdm
pip install wandb  # Optional: for experiment tracking
pip install tensorboard  # Optional: for logging

# Install development tools (optional but recommended)
echo "Installing development tools..."
pip install ipython
pip install jupyter
pip install matplotlib
pip install seaborn

# Verify installations
echo "Verifying critical installations..."

echo "Checking Python version:"
python --version

echo "Checking PyTorch installation:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null || echo "CUDA version info not available"

echo "Checking Transformers installation:"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo "Checking PEFT installation:"
python -c "import peft; print(f'PEFT version: {peft.__version__}')"

echo "Checking BitsAndBytes installation:"
python -c "import bitsandbytes; print(f'BitsAndBytes available: True')" || echo "⚠️  BitsAndBytes installation may have issues"

echo "Checking other core libraries:"
python -c "import pandas, datasets, accelerate; print('Core libraries imported successfully')"

# Create requirements.txt for reference
echo "Creating requirements.txt file..."
pip freeze > requirements.txt

# Display final information
echo ""
echo "Environment setup completed successfully!"
echo ""
echo "NEXT STEPS:"
echo "1. Set your Hugging Face token: export HF_TOKEN=your_token_here"
echo "2. Prepare your dataset CSV with 'text' and 'label' columns"
echo "3. Update paths in test_llama.sh if needed"
echo "4. Submit your job: sbatch test_llama.sh"
echo ""
echo "Virtual environment location: $(pwd)/$VENV_NAME"
echo "Requirements saved to: $(pwd)/requirements.txt"
echo ""
echo "To activate this environment in the future:"
echo "source $(pwd)/$VENV_NAME/bin/activate"
echo ""
echo "⚠️  IMPORTANT SECURITY NOTES:"
echo "- Never commit your HF_TOKEN to version control"
echo "- Use environment variables for sensitive tokens"
echo "- Consider using .env files for local development"
echo ""
echo "Useful links:"
echo "- Hugging Face Tokens: https://huggingface.co/settings/tokens"
echo "- Meta Llama 3 Model: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
echo "- CIC IoT 2023 Dataset: https://www.unb.ca/cic/datasets/iotdataset-2023.html"