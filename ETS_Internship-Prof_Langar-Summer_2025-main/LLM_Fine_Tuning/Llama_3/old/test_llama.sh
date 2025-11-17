#!/bin/bash
#SBATCH --account=def-rlangar_gpu
#SBATCH --job-name=Llama_3_CIC_IOT
#SBATCH --output=Llama_3_CIC_IOT.out
#SBATCH --error=Llama_3_CIC_IOT.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=12:00:00

# Email notifications - UPDATE WITH YOUR EMAIL
#SBATCH --mail-user=your_email@example.com
#SBATCH --mail-type=ALL

# =============================================================================
# SLURM Job Script for Llama-3 Fine-tuning on CIC IoT 2023 Dataset
# =============================================================================
# 
# INSTRUCTIONS FOR NEW USERS:
# 1. Update the email address above with your actual email
# 2. Set your Hugging Face token as an environment variable:
#    export HF_TOKEN=your_hugging_face_token_here
# 3. Make sure your dataset (cic_iot_ddos_clean.csv) is in the same directory
# 4. Ensure you have access to Meta-Llama-3-8B-Instruct model on HuggingFace
# 5. Update the virtual environment path below to match your setup
#
# BEFORE RUNNING:
# - Verify you have completed the environment setup (see env_setup.sh)
# - Check that your dataset is properly formatted with 'text' and 'label' columns
# - Ensure you have sufficient GPU memory (this script uses V100 with 32GB memory)
#
# =============================================================================

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set!"
    echo "Please set it with: export HF_TOKEN=your_hugging_face_token"
    exit 1
fi

echo "Starting Llama-3 fine-tuning job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Activate virtual environment
# UPDATE THIS PATH TO MATCH YOUR VIRTUAL ENVIRONMENT LOCATION
VENV_PATH="/home/sonngnn/projects/def-rlangar/sonngnn/masterenv"

echo "Activating virtual environment: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Verify Python and CUDA availability
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Print current directory and check for required files
echo "Current directory: $(pwd)"
echo "Dataset file check:"
if [ -f "CICIoT2023_attacks_benign_CTGAN_preprocessed.csv" ]; then
    echo "✓ Dataset file found: CICIoT2023_attacks_benign_CTGAN_preprocessed.csv"
    echo "Dataset info: $(wc -l CICIoT2023_attacks_benign_CTGAN_preprocessed.csv)"
else
    echo "⚠️  WARNING: Dataset file 'CICIoT2023_attacks_benign_CTGAN_preprocessed.csv' not found in current directory"
    echo "Please ensure your dataset is in the correct location"
fi

# Check if training script exists
if [ -f "Llama_3_CIC_IOT_2023.py" ]; then
    echo "✓ Training script found"
else
    echo "ERROR: Training script 'Llama_3_CIC_IOT_2023.py' not found"
    exit 1
fi

# Set additional environment variables for optimal performance
export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism warnings
export CUDA_LAUNCH_BLOCKING=1        # For better error reporting

# Start training with unbuffered output
echo "Starting training..."
python -u Llama_3_CIC_IOT_2023.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
    exit 1
fi

echo "Job completed at: $(date)"