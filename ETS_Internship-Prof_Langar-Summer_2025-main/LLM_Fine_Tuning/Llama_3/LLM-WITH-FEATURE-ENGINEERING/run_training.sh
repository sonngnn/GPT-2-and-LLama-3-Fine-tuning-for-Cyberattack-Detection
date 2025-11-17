#!/bin/bash
#SBATCH --account=def-rlangar
#SBATCH --job-name=llama3_multifeature
#SBATCH --output=logs/multifeature_out_%j.txt
#SBATCH --error=logs/multifeature_err_%j.err
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=bamolitho@gmail.com
#SBATCH --mail-type=ALL

echo "=== SLURM Multi-Feature Pipeline Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "======================================="

# Pipeline configuration
PROJECT_PATH="/project/6084087/amomo/LLM-WITH-FEATURE-ENGINEERING"
FEATURES_LIST="12 16 20"  # Multiple features for parallel training
CSV_PATH="./data/CICIoT2023_attacks_benign_CTGAN_V2.csv"
MAX_CONCURRENT_TRAINING=2  

echo "Configuration:"
echo "  Project: $PROJECT_PATH"
echo "  Features: $FEATURES_LIST"
echo "  CSV: $CSV_PATH"
echo "  Max concurrent training: $MAX_CONCURRENT_TRAINING"

# Load modules
echo "Loading modules..."
module load gcc
module load arrow/21.0.0
module load cuda/12.6

# Activate virtual environment
echo "Activating virtual environment..."
source /project/6084087/amomo/amomo_venv2/bin/activate

# Cache configuration - DIRECT WORK (no copying)
export TRANSFORMERS_CACHE="/project/6084087/amomo/.cache/transformers"
export HF_HOME="/project/6084087/amomo/.cache/huggingface"
export HF_DATASETS_CACHE="/project/6084087/amomo/.cache/datasets"

echo "Cache configuration:"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "  HF_HOME: $HF_HOME"

# Change to working directory
cd $PROJECT_PATH
echo "Working directory: $(pwd)"

# HuggingFace token verification
if [ -f "/project/6084087/amomo/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat /project/6084087/amomo/.cache/huggingface/token)
    export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
    echo "HuggingFace token loaded"
else
    echo "WARNING: HuggingFace token not found"
fi

# Configure offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "Offline mode activated"

# System information
echo "=== System Information ==="
nvidia-smi
echo "CUDA Version: $(nvcc --version | grep release || echo 'N/A')"
echo "Python Version: $(python --version)"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)' || echo 'N/A')"

# GPU availability check
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || {
    echo "ERROR: Unable to verify CUDA"
    exit 1
}

# Model cache verification
echo "=== Model Cache Verification ==="
if [ -d "$TRANSFORMERS_CACHE" ]; then
    echo "Transformers cache found:"
    find $TRANSFORMERS_CACHE -name "*llama*" -type d | head -3
else
    echo "ERROR: Transformers cache not found: $TRANSFORMERS_CACHE"
    exit 1
fi

# CRITICAL FIX: Model loading test without local_files_only since we're in offline mode
echo "Testing model loading..."
python -c "
import os
from transformers import AutoTokenizer

try:
    cache_dir = os.environ.get('TRANSFORMERS_CACHE')
    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-3.2-3B-Instruct',
        cache_dir=cache_dir
    )
    print('[SUCCESS] Model loading test successful')
except Exception as e:
    print(f'[FAILED] Model loading test failed: {e}')
    exit(1)
" || {
    echo "ERROR: Model loading test failed"
    exit 1
}

# CSV file verification
if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: CSV file not found: $CSV_PATH"
    echo "Check the path in configuration"
    exit 1
fi

CSV_SIZE=$(du -sh "$CSV_PATH" | cut -f1)
echo "CSV file found: $CSV_PATH ($CSV_SIZE)"

# Required scripts verification
REQUIRED_SCRIPTS=(
    "orchestrate_pipeline.py"
    "data_preprocessing.py" 
    "feature_engineering.py"
    "llama_finetuning.py"
)

echo "Script verification..."
for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "ERROR: Missing script: $script"
        exit 1
    else
        echo "  [SUCCESS] $script"
    fi
done

# Create working directories
echo "Creating working directories..."
mkdir -p data outputs logs

# Optimization variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=true
export WANDB_DISABLED=true

echo "=== STARTING MULTI-FEATURE PIPELINE ==="
echo "Features to process: $FEATURES_LIST"
echo "Start: $(date)"

# Execute pipeline with error handling
python orchestrate_pipeline.py \
    --features_list $FEATURES_LIST \
    --csv_path "$CSV_PATH" \
    --max_concurrent_training $MAX_CONCURRENT_TRAINING \
    2>&1 | tee "logs/pipeline_execution_${SLURM_JOB_ID}.log"

# Success verification
PIPELINE_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=== PIPELINE RESULTS ==="
echo "End: $(date)"
echo "Exit code: $PIPELINE_EXIT_CODE"

if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY"
    
    # Results analysis
    echo ""
    echo "=== RESULTS ANALYSIS ==="
    
    # Dataset verification
    echo "Generated datasets:"
    DATASETS_OK=0
    for n_features in $FEATURES_LIST; do
        data_dir="data/data_${n_features}_features"
        if [ -d "$data_dir" ] && [ -f "$data_dir/dataset_metadata.json" ]; then
            dataset_size=$(du -sh "$data_dir" 2>/dev/null | cut -f1 || echo "?")
            sample_count=$(python -c "import json; print(json.load(open('$data_dir/dataset_metadata.json'))['total_samples'])" 2>/dev/null || echo "?")
            echo "  [SUCCESS] $n_features features: $data_dir ($dataset_size, $sample_count samples)"
            DATASETS_OK=$((DATASETS_OK + 1))
        else
            echo "  [FAILED] $n_features features: failed or incomplete"
        fi
    done
    
    # Model verification
    echo ""
    echo "Trained models:"
    MODELS_OK=0
    for n_features in $FEATURES_LIST; do
        model_dir="outputs/model_${n_features}_features"
        if [ -d "$model_dir" ]; then
            model_size=$(du -sh "$model_dir" 2>/dev/null | cut -f1 || echo "?")
            
            # Check essential model files
            if [ -f "$model_dir/adapter_model.bin" ] || [ -f "$model_dir/adapter_model.safetensors" ]; then
                echo "  [SUCCESS] $n_features features: $model_dir ($model_size)"
                MODELS_OK=$((MODELS_OK + 1))
            else
                echo "  [!] $n_features features: directory present but model incomplete"
            fi
        else
            echo "  [FAILED] $n_features features: no model"
        fi
    done
    
    # Statistical summary
    TOTAL_REQUESTED=$(echo $FEATURES_LIST | wc -w)
    echo ""
    echo "=== FINAL STATISTICS ==="
    echo "Requested configurations: $TOTAL_REQUESTED"
    echo "Generated datasets: $DATASETS_OK/$TOTAL_REQUESTED"
    echo "Trained models: $MODELS_OK/$TOTAL_REQUESTED"
    echo "Success rate: $((MODELS_OK * 100 / TOTAL_REQUESTED))%"
    
    # Usage information
    if [ $MODELS_OK -gt 0 ]; then
        echo ""
        echo "=== AVAILABLE MODELS ==="
        for n_features in $FEATURES_LIST; do
            model_dir="outputs/model_${n_features}_features"
            if [ -d "$model_dir" ] && ([ -f "$model_dir/adapter_model.bin" ] || [ -f "$model_dir/adapter_model.safetensors" ]); then
                echo "Model $n_features features:"
                echo "  Directory: $model_dir"
                echo "  Usage inference: Modify inference.py to point to this model"
                echo "  Usage evaluation: Modify evaluate_model.py to point to this model"
                echo ""
            fi
        done
    fi
    
else
    echo "[FAILED] PIPELINE FAILED (code: $PIPELINE_EXIT_CODE)"
    
    # Error analysis
    echo ""
    echo "=== ERROR ANALYSIS ==="
    echo "Detailed logs available in:"
    echo "  - logs/pipeline_execution_${SLURM_JOB_ID}.log"
    
    # Display last error logs
    for n_features in $FEATURES_LIST; do
        error_log="logs/${n_features}_features"
        if [ -d "$error_log" ]; then
            echo ""
            echo "--- Errors $n_features features ---"
            for log_file in "$error_log"/*.log; do
                if [ -f "$log_file" ] && [ -s "$log_file" ]; then
                    echo "File: $(basename "$log_file")"
                    echo "Last lines:"
                    tail -n 5 "$log_file" 2>/dev/null | sed 's/^/    /'
                fi
            done
        fi
    done
fi

# Cleanup and finalization
echo ""
echo "=== CLEANUP ==="

# Important logs backup
LOG_ARCHIVE="logs/job_${SLURM_JOB_ID}_complete.tar.gz"
tar -czf "$LOG_ARCHIVE" logs/ 2>/dev/null && echo "Logs archived: $LOG_ARCHIVE"

# GPU memory cleanup
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU cache cleaned')
" 2>/dev/null || true

echo "=== JOB COMPLETED ==="
echo "Job ID: $SLURM_JOB_ID"
echo "End: $(date)"

exit $PIPELINE_EXIT_CODE

# =============================================================================
# train.sh - Updated for single model training
# =============================================================================

#!/bin/bash

# LLaMA-3 Network Traffic Classification Training Script (Single Model)
# This script runs training for a single model configuration

echo "=== LLaMA-3 Network Traffic Classification Training Pipeline ==="

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null

# Environment variables for better performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p llama-network-classifier
mkdir -p logs

# Configuration - adjust as needed
N_FEATURES=8  # Change this to the desired number of features
DATA_DIR="data/data_${N_FEATURES}_features"

# Step 1: Data Preprocessing
echo "=== Step 1: Data Preprocessing ==="
if [ -d "$DATA_DIR" ] && [ -f "$DATA_DIR/dataset_metadata.json" ]; then
    echo "Data for $N_FEATURES features already exists - skipping preprocessing"
else
    echo "Converting CSV data to instruction format for $N_FEATURES features..."
    python data_preprocessing.py --n_features $N_FEATURES
    
    # Check if data preprocessing was successful
    if [ ! -f "$DATA_DIR/train.json" ]; then
        echo "Error: Data preprocessing failed. Please check your CSV file path."
        exit 1
    fi
    
    echo "Data preprocessing completed successfully!"
fi

# Step 2: Fine-tuning
echo "=== Step 2: Starting Fine-tuning ==="
echo "This may take several hours depending on your hardware..."

python llama_finetuning.py --n_features $N_FEATURES

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "=== Training completed successfully! ==="
    echo "Model saved to: ./llama-network-classifier-${N_FEATURES}features"
    
    # Step 3: Test inference (if inference script exists)
    if [ -f "inference.py" ]; then
        echo "=== Step 3: Testing Inference ==="
        python inference.py
    else
        echo "Inference script not found - skipping inference test"
    fi
    
    echo "=== Pipeline completed successfully! ==="
    echo "You can now use the fine-tuned model for network traffic classification."
else
    echo "Error: Training failed. Please check the logs for details."
    exit 1
fi