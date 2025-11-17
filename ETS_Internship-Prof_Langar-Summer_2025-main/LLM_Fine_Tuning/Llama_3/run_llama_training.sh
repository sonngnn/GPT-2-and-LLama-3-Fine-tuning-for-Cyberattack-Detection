#!/bin/bash
#SBATCH --account=def-rlangar
#SBATCH --job-name=llama3_network_classification
#SBATCH --output=logs/llama3_out_%j.txt
#SBATCH --error=logs/llama3_err_%j.err
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=mail@gmail.com       
#SBATCH --mail-type=ALL

echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "================================"

# Define project path
PROJECT_PATH="/project/6084087/username/LLMs"
echo "Project path: $PROJECT_PATH"

# Load required modules
echo "Loading modules..."
module load gcc
module load arrow/21.0.0   # IMPORTANT for pyarrow
module load cuda/12.6

# Activate virtual environment
echo "Activating virtual environment..."
source /project/6084087/username/username_venv2/bin/activate

# Set cache directories to use the pre-downloaded models
export TRANSFORMERS_CACHE="/project/6084087/username/.cache/transformers"
export HF_HOME="/project/6084087/username/.cache/huggingface"
export HF_DATASETS_CACHE="/project/6084087/username/.cache/datasets"

# Create local cache directories in temp for faster access
TEMP_CACHE="$SLURM_TMPDIR/cache"
mkdir -p $TEMP_CACHE

echo "Copying cached models to local temp directory for faster access..."
if [ -d "/project/6084087/username/.cache/transformers" ]; then
    cp -r /project/6084087/username/.cache/transformers $TEMP_CACHE/
    export TRANSFORMERS_CACHE="$TEMP_CACHE/transformers"
    echo "[SUCCESS] Model cache copied to local temp"
    ls -la $TEMP_CACHE/transformers/
else
    echo "[WARNING] No cached models found at /project/6084087/username/.cache/transformers"
    echo "Using original cache location"
fi

# Copy project to temporary directory for faster I/O
echo "Copying project from $PROJECT_PATH to temporary directory..."
cd $SLURM_TMPDIR
cp -r $PROJECT_PATH/* .

# Verify files are copied correctly
echo "Verifying copied files..."
ls -la
if [ ! -f "train.sh" ]; then
    echo "ERROR: train.sh not found in $PROJECT_PATH!"
    echo "Please verify your project path and ensure train.sh exists."
    exit 1
fi

# IMPORTANT: Copy HuggingFace token to temporary directory
echo "Copying HuggingFace authentication..."
mkdir -p .cache/huggingface
if [ -f "/project/6084087/username/.cache/huggingface/token" ]; then
    cp /project/6084087/username/.cache/huggingface/token .cache/huggingface/
    echo "[SUCCESS] HF token copied to SLURM_TMPDIR"
else
    echo "[FAILED] HF token not found at /project/6084087/username/.cache/huggingface/token"
    # Try home directory
    if [ -f "$HOME/.cache/huggingface/token" ]; then
        cp $HOME/.cache/huggingface/token .cache/huggingface/
        echo "[SUCCESS] HF token copied from home directory"
    else
        echo "[FAILED] No HF token found in home directory either"
        exit 1
    fi
fi

# Update HF_HOME to point to local cache
export HF_HOME=$SLURM_TMPDIR/.cache/huggingface
echo "HF_HOME updated to: $HF_HOME"

# Charger token HuggingFace depuis le fichier et l'exporter
if [ -f "$HF_HOME/token" ]; then
    export HF_TOKEN=$(cat $HF_HOME/token)
    export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
    echo "HF_TOKEN exporté avec succès"
fi

# Display system information
echo "=== System Information ==="
nvidia-smi
echo "CUDA Version: $(nvcc --version | grep release)"
echo "Python Version: $(python --version)"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
echo "Available GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU Memory: $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")' 2>/dev/null || echo 'N/A')"
echo "================================"

# Test cache availability
echo "=== Testing Model Cache ==="
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
if [ -d "$TRANSFORMERS_CACHE" ]; then
    echo "[SUCCESS] Transformers cache directory exists"
    echo "Cache contents:"
    find $TRANSFORMERS_CACHE -name "*llama*" -type d | head -5
else
    echo "[WARNING] Transformers cache not found"
fi

# Test model loading from cache
echo "Testing model loading from cache..."
python -c "
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    cache_dir = os.environ.get('TRANSFORMERS_CACHE')
    print(f'Using cache directory: {cache_dir}')
    
    # Test tokenizer loading
    print('Testing tokenizer loading from cache...')
    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-3.2-3B-Instruct',
        cache_dir=cache_dir,
        local_files_only=True,
        trust_remote_code=True
    )
    print('[SUCCESS] Tokenizer loaded from cache!')
    
    # Test that we can at least access the model config
    print('Testing model config access...')
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        'meta-llama/Llama-3.2-3B-Instruct',
        cache_dir=cache_dir,
        local_files_only=True,
        trust_remote_code=True
    )
    print('[SUCCESS] Model config accessible from cache!')
    print(f'Model type: {config.model_type}')
    
except Exception as e:
    print(f'[FAILED] Cache test failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" || {
    echo "[FAILED] Model cache test failed!"
    echo "Please ensure the model was downloaded correctly"
    exit 1
}

# Check HuggingFace authentication
echo "Checking HuggingFace authentication..."
echo "Testing HF_TOKEN content (first 5 chars): ${HF_TOKEN:0:5}..."
# python -c "from huggingface_hub import whoami; print(whoami())"
# hf auth whoami || {
#     echo "ERROR: HuggingFace authentication failed!"
#     echo "Please run 'huggingface-cli login' before submitting this job."
#     echo "Steps to fix:"
#     echo "1. huggingface-cli logout"
#     echo "2. huggingface-cli login"
#     echo "3. Verify access at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct"
#     exit 1
# }

# Test model access specifically
# echo "Testing LLaMA-3.2 model access..."
# python simple_load_test_llama3.py > test_output.txt 2>&1

# if grep -q "\[SUCCESS\] LLaMA-3.2-3B-Instruct access confirmed!" test_output.txt; then
#     echo "Model access test PASSED"
# else
#     echo "Model access test FAILED"
#     cat test_output.txt  # Montre l'erreur
#     exit 1
# fi

# Verify required files exist
if [ ! -f "data_preprocessing.py" ]; then
    echo "ERROR: data_preprocessing.py not found!"
    exit 1
fi

# Check if CSV path is configured
CSV_CHECK=$(grep -n "csv_path.*=" data_preprocessing.py | head -1)
echo "Dataset configuration: $CSV_CHECK"

# Make train.sh executable
chmod +x train.sh

# Modify llama_finetuning.py to force local_files_only mode
echo "=== Modifying training script for offline mode ==="
if [ -f "llama_finetuning.py" ]; then
    # Create a backup
    cp llama_finetuning.py llama_finetuning_original.py
    
    # Modify to force offline mode
    python -c "
import re

with open('llama_finetuning.py', 'r') as f:
    content = f.read()

# Add local_files_only=True to tokenizer loading
content = re.sub(
    r'(AutoTokenizer\.from_pretrained\(\s*self\.model_name,)',
    r'\1\n            local_files_only=True,',
    content
)

# Add local_files_only=True to model loading
content = re.sub(
    r'(AutoModelForCausalLM\.from_pretrained\(\s*self\.model_name,)',
    r'\1\n            local_files_only=True,',
    content
)

with open('llama_finetuning.py', 'w') as f:
    f.write(content)

print('[SUCCESS] Training script modified for offline mode')
"
    echo "Training script prepared for offline execution"
else
    echo "[WARNING] llama_finetuning.py not found"
fi

# Run your existing training pipeline
echo "=== Starting Training Pipeline using train.sh ==="
./train.sh

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training pipeline completed successfully!"
    
    # Run additional evaluation if available
    if [ -f "evaluate_model.py" ]; then
        echo "=== Running Comprehensive Model Evaluation ==="
        python evaluate_model.py
        if [ $? -eq 0 ]; then
            echo "Model evaluation completed!"
        else
            echo "Model evaluation failed, but training was successful"
        fi
    fi
    
    # Copy results back to project directory
    echo "Copying results back to project directory..."
    cp -r llama-network-classifier $PROJECT_PATH/
    cp -r evaluation_results $PROJECT_PATH/ 2>/dev/null || echo "No evaluation results to copy"
    cp -r logs $PROJECT_PATH/ 2>/dev/null || echo "No additional logs to copy"
    cp -r data $PROJECT_PATH/ 2>/dev/null || echo "No data directory to copy"
    
    # Also copy to home directory for easy access
    echo "Copying results to home directory for easy access..."
    cp -r llama-network-classifier ~/
    cp -r evaluation_results ~/ 2>/dev/null || echo "No evaluation results to copy"
    
    # Display training summary
    echo "=== Training Summary ==="
    echo "Job ID: $SLURM_JOB_ID"
    echo "End Time: $(date)"
    echo "Model saved to:"
    echo "  - Project: $PROJECT_PATH/llama-network-classifier/"
    echo "  - Home: ~/llama-network-classifier/"
    echo "Evaluation results: $PROJECT_PATH/evaluation_results/"
    echo "Logs: $PROJECT_PATH/logs/"
    
    # Check final model size
    if [ -d "llama-network-classifier" ]; then
        MODEL_SIZE=$(du -sh llama-network-classifier | cut -f1)
        echo "Final model size: $MODEL_SIZE"
        echo "Model files:"
        ls -la llama-network-classifier/
    fi
    
    echo "=== Job Completed Successfully! ==="
else
    echo "Training pipeline failed!"
    echo "Check the error logs for details"
    exit 1
fi

echo "All done! Check your email for job completion notification."