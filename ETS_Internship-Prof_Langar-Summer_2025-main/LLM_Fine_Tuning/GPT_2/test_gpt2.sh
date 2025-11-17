#!/bin/bash
#SBATCH --account=def-rlangar_gpu
#SBATCH --job-name=GPT2_CIC_IOT
#SBATCH --output=GPT2_CIC_IOT.out
#SBATCH --error=GPT2_CIC_IOT.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=8:00:00

# Email notifications - UPDATE WITH YOUR EMAIL
#SBATCH --mail-user=your_email@example.com
#SBATCH --mail-type=ALL

# =============================================================================
# SLURM Job Script for GPT-2 Fine-tuning on CIC IoT 2023 Dataset
# =============================================================================
# 
# INSTRUCTIONS FOR NEW USERS:
# 1. Update the email address above with your actual email
# 2. Optionally set your Hugging Face token (not required for GPT-2):
#    export HF_TOKEN=your_hugging_face_token_here
# 3. Make sure your dataset (CICIoT2023_attacks_benign_CTGAN_preprocessed.csv) is in the same directory
# 4. Update the virtual environment path below to match your setup
#
# BEFORE RUNNING:
# - Verify you have completed the environment setup (see env_setup.sh)
# - Check that your dataset is properly formatted with 'text' and 'label' columns
# - Ensure you have sufficient GPU memory (this script uses V100 with 32GB memory)
# - GPT-2 training typically takes less time than Llama-3, hence the 8-hour time limit
#
# DATASET FORMAT:
# Your CSV should have columns: 'text', 'label'
# Example:
# text,label
# "flow_duration:1234 src_port:80...",benign
# "flow_duration:5678 src_port:12345...",ddos
#
# =============================================================================

echo "Starting GPT-2 fine-tuning job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Check if HF_TOKEN is set (optional for GPT-2)
if [ -z "$HF_TOKEN" ]; then
    echo "INFO: HF_TOKEN environment variable is not set"
    echo "This is fine for GPT-2 (publicly available model)"
    echo "Set HF_TOKEN for potentially faster downloads: export HF_TOKEN=your_token"
else
    echo "INFO: Using Hugging Face token for authentication"
fi

# Activate virtual environment
# UPDATE THIS PATH TO MATCH YOUR VIRTUAL ENVIRONMENT LOCATION
VENV_PATH="/home/sonngnn/projects/def-rlangar/sonngnn/gpt2env"

echo "Activating virtual environment: $VENV_PATH"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    echo "Please run env_setup.sh first or update the path"
    exit 1
fi

# Verify Python and dependencies
echo "=== Environment Verification ==="
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" != "True" ]; then
    echo "⚠️  WARNING: CUDA not available! Training will be very slow on CPU"
fi

# Print current directory and check for required files
echo "=== File Verification ==="
echo "Current directory: $(pwd)"
echo "Available files:"
ls -la *.csv *.py 2>/dev/null || echo "No CSV or Python files found"

echo "Dataset file check:"
DATASET_FILE="CICIoT2023_attacks_benign_CTGAN_preprocessed.csv"
if [ -f "$DATASET_FILE" ]; then
    echo "✓ Dataset file found: $DATASET_FILE"
    echo "Dataset size: $(wc -l < $DATASET_FILE) lines"
    echo "First few lines:"
    head -3 "$DATASET_FILE" | cat -n
else
    echo "⚠️  WARNING: Dataset file '$DATASET_FILE' not found in current directory"
    echo "Available CSV files:"
    ls -la *.csv 2>/dev/null || echo "No CSV files found"
    echo "Please ensure your dataset is in the correct location with the correct name"
fi

# Check if training script exists
SCRIPT_NAME="GPT2_CIC_IOT_2023.py"
if [ -f "$SCRIPT_NAME" ]; then
    echo "✓ Training script found: $SCRIPT_NAME"
else
    echo "ERROR: Training script '$SCRIPT_NAME' not found"
    echo "Available Python files:"
    ls -la *.py 2>/dev/null || echo "No Python files found"
    exit 1
fi

# Set additional environment variables for optimal performance
export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism warnings
export CUDA_LAUNCH_BLOCKING=1        # For better error reporting
export PYTHONUNBUFFERED=1           # Ensure immediate output
export NUM_PROC=8                    # Number of processes for dataset processing

# Create outputs directory if it doesn't exist
mkdir -p outputs

echo "=== Starting Training ==="
echo "Configuration:"
echo "- Model: GPT-2-Large"
echo "- Dataset: $DATASET_FILE"
echo "- Output directory: outputs/gpt2_ddos/"
echo "- GPU memory optimization: FP16 enabled"
echo "- Batch size: 4 per device with 16x gradient accumulation"
echo "- Effective batch size: 64"

# Start training with unbuffered output
echo "Launching training script..."
python -u "$SCRIPT_NAME"

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "End time: $(date)"
    
    # Display output directory contents
    echo "=== Training Results ==="
    if [ -d "outputs/gpt2_ddos" ]; then
        echo "Output directory contents:"
        find outputs/gpt2_ddos -name "run_*" -type d | head -1 | xargs ls -la
        
        # Show loss curve if available
        LOSS_CURVE=$(find outputs/gpt2_ddos -name "loss_curve.png" | head -1)
        if [ -f "$LOSS_CURVE" ]; then
            echo "Loss curve saved: $LOSS_CURVE"
        fi
        
        # Show training metrics if available
        METRICS_FILE=$(find outputs/gpt2_ddos -name "training_metrics.json" | head -1)
        if [ -f "$METRICS_FILE" ]; then
            echo "Training metrics:"
            cat "$METRICS_FILE"
        fi
    fi
else
    echo "Training failed with exit code $EXIT_CODE"
    echo "End time: $(date)"
    
    # Show recent log entries for debugging
    echo "=== Recent Log Entries ==="
    if [ -f "training.log" ]; then
        echo "Last 20 lines of training.log:"
        tail -20 training.log
    fi
    
    exit $EXIT_CODE
fi

echo "=== Job Summary ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Status: $([ $EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"