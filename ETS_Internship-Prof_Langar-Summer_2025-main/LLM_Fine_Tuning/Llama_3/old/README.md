# Llama-3 Fine-tuning for Cybersecurity Traffic Classification

This project fine-tunes Meta's Llama-3-8B-Instruct model for network traffic classification and cyber-attack detection using the CIC IoT 2023 dataset. The implementation uses LoRA (Low-Rank Adaptation) with 4-bit quantization for memory-efficient training.

## Project Overview

- **Model**: Meta-Llama-3-8B-Instruct
- **Dataset**: CIC IoT 2023 (network traffic with DDoS, DoS, Mirai, Recon attacks)
- **Technique**: LoRA fine-tuning with 4-bit quantization
- **Hardware**: Optimized for V100 GPUs (32GB memory)
- **Framework**: PyTorch + Hugging Face Transformers

## Prerequisites

### 1. Hugging Face Access
- Create a [Hugging Face account](https://huggingface.co/)
- Request access to [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- Generate an access token from [HF Settings](https://huggingface.co/settings/tokens)

### 2. Dataset Preparation
Prepare your CSV dataset with the following columns:
- `text`: Network flow features/data
- `label`: Traffic type (e.g., "benign", "ddos", "dos", "mirai", "recon")

Example CSV structure:
```csv
text,label
"flow_duration:1234 src_port:80 dst_port:443 protocol:TCP...",benign
"flow_duration:5678 src_port:12345 dst_port:80 protocol:TCP...",ddos
```

### 3. Computing Environment
- CUDA-capable GPU (recommended: V100 with 32GB+ memory)
- Python 3.11+
- Compute Canada/Digital Research Alliance access (for the provided scripts)

## Quick Start

### Step 1: Environment Setup
```bash
# Run the environment setup script
bash env_setup.sh

# Activate the virtual environment
source masterenv/bin/activate
```

### Step 2: Set Hugging Face Token
```bash
# NEVER commit tokens to git!
export HF_TOKEN=your_hugging_face_token_here
```

### Step 3: Prepare Dataset
Place your dataset file as `CICIoT2023_attacks_benign_CTGAN_preprocessed.csv` in the project directory, or update the path in the configuration.

### Step 4: Configure Training
Edit the `CONFIG` dictionary in `Llama_3_CIC_IOT_2023.py` to adjust:
- Batch sizes (based on your GPU memory)
- Learning rate and training epochs
- LoRA parameters
- Dataset path

### Step 5: Submit Training Job
```bash
# Update email and paths in test_llama.sh
# Then submit to SLURM
sbatch test_llama.sh

# Or run directly
python Llama_3_CIC_IOT_2023.py
```

## File Structure

```
project/
├── Llama_3_CIC_IOT_2023.py    # Main training script
├── test_llama.sh              # SLURM job submission script
├── env_setup.sh               # Environment setup script
├── CICIoT2023_attacks_benign_CTGAN_preprocessed.csv     # Your dataset (not included)
├── requirements.txt           # Python dependencies (generated)
└── outputs/                   # Training outputs
    └── llama3_ddos_lora/
        └── run_YYYYMMDD_HHMMSS/
            ├── adapter_model.safetensors
            ├── training_metrics.json
            └── logs/
```