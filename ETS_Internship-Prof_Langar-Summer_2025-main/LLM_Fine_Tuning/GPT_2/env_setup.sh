#!/bin/bash
# =============================================================================
# Environment Setup Script for GPT-2 Fine-tuning
# =============================================================================
#
# This script sets up the Python environment and installs all required 
# dependencies for fine-tuning GPT-2 on the CIC IoT 2023 dataset.
#
# USAGE:
#   bash env_setup.sh
#
# REQUIREMENTS:
#   - Access to Compute Canada/Digital Research Alliance modules
#   - Internet connection for pip installations
#   - Sufficient disk space for packages (~2-3 GB)
#
# DIFFERENCES FROM LLAMA-3 SETUP:
#   - No PEFT/LoRA dependencies (using standard fine-tuning)
#   - No BitsAndBytes (no quantization needed for GPT-2)
#   - Lighter requirements overall
#
# =============================================================================

set -e  # Exit on any error

echo "Setting up environment for GPT-2 fine-tuning..."

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
VENV_NAME="gpt2env"
echo "Creating virtual environment: $VENV_NAME"

if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment '$VENV_NAME' already exists. Removing..."
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
pip install --no-index pandas scikit-learn

# Install transformers and related libraries
echo "Installing Hugging Face libraries..."
pip install transformers>=4.36.0
pip install datasets
pip install tokenizers
pip install accelerate

# Install visualization libraries
echo "Installing visualization libraries..."
pip install matplotlib
pip install seaborn

# Install additional utilities
echo "Installing additional utilities..."
pip install evaluate
pip install tqdm

# Optional: Install experiment tracking tools
echo "Installing experiment tracking tools (optional)..."
pip install wandb
pip install tensorboard

# Install development tools (optional but recommended)
echo "Installing development tools..."
pip install ipython
pip install jupyter

# Verify installations
echo "Verifying critical installations..."

echo "Checking Python version:"
python --version

echo "Checking PyTorch installation:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null || echo "CUDA version info not available"

# Check CUDA device if available
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('No CUDA devices available')
"

echo "Checking Transformers installation:"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo "Checking other core libraries:"
python -c "
try:
    import pandas, datasets, accelerate, matplotlib
    print('Core libraries imported successfully')
    print(f'Pandas version: {pandas.__version__}')
    print(f'Datasets version: {datasets.__version__}')
    print(f'Matplotlib version: {matplotlib.__version__}')
except Exception as e:
    print(f'Error importing libraries: {e}')
"

# Test GPT-2 model loading
echo "Testing GPT-2 model access..."
python -c "
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print('Testing GPT-2 tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    print('GPT-2 tokenizer loaded successfully')
    print('Note: Full model loading test skipped to save time and memory')
    print('GPT-2 models are publicly available and should work without issues')
except Exception as e:
    print(f'⚠️  Error testing GPT-2 access: {e}')
"

# Create requirements.txt for reference
echo "Creating requirements.txt file..."
pip freeze > requirements.txt

# Display final information
echo ""
echo "Environment setup completed successfully!"
echo ""
echo "NEXT STEPS:"
echo "1. (Optional) Set your Hugging Face token for faster downloads:"
echo "   export HF_TOKEN=your_token_here"
echo "2. Prepare your dataset CSV with 'text' and 'label' columns"
echo "3. Update paths in test_gpt2.sh if needed"
echo "4. Submit your job: sbatch test_gpt2.sh"
echo ""
echo "Virtual environment location: $(pwd)/$VENV_NAME"
echo "Requirements saved to: $(pwd)/requirements.txt"
echo ""
echo "To activate this environment in the future:"
echo "source $(pwd)/$VENV_NAME/bin/activate"
echo ""
echo "⚠️  IMPORTANT SECURITY NOTES:"
echo "- Hugging Face tokens are optional for GPT-2 (publicly available)"
echo "- Never commit any tokens to version control"
echo "- Use environment variables for sensitive information"
echo ""
echo "GPT-2 vs Llama-3 DIFFERENCES:"
echo "- GPT-2 is publicly available (no special access needed)"
echo "- Standard fine-tuning instead of LoRA (no PEFT library needed)"
echo "- No quantization required (lighter memory usage)"
echo "- Faster training due to smaller model size"
echo ""
echo "Useful links:"
echo "- Hugging Face Tokens (optional): https://huggingface.co/settings/tokens"
echo "- GPT-2 Large Model: https://huggingface.co/gpt2-large"
echo "- CIC IoT 2023 Dataset: https://www.unb.ca/cic/datasets/iotdataset-2023.html"
echo "- Transformers Documentation: https://huggingface.co/docs/transformers/"