# LLaMA-3.2 Network Traffic Classification with LoRA

This repository contains a complete pipeline for fine-tuning LLaMA-3.2-3B-Instruct on network traffic classification tasks using LoRA (Low-Rank Adaptation). The model can classify network flows into categories like DDoS, Mirai, XSS, Normal traffic, and other attack types.

## Features

- **Complete Pipeline**: From data preprocessing to model inference
- **Memory Efficient**: Uses 4-bit quantization and LoRA for efficient training
- **GPU Optimized**: Supports CUDA with automatic device mapping
- **Easy to Use**: Simple scripts for training and inference
- **Evaluation Metrics**: Built-in accuracy and classification report generation
- **Compute Canada Ready**: Optimized for HPC environments

---

# Installation and Fine-Tuning Guide for LLaMA 3.2-3B-Instruct on Compute Canada

This is a complete step-by-step guide to download **meta-llama/Llama-3.2-3B-Instruct** from HuggingFace and install it on Compute Canada for fine-tuning. All commands and tips are included to avoid VRAM and dependency issues.

---

## 1️⃣ Prepare Your HuggingFace Account

1. Create an account on [HuggingFace](https://huggingface.co/).
2. Accept the **LLaMA 3.x** license (Meta) if not already done.
3. Generate an **API access token**:
   - Profile → Settings → Access Tokens → New token (type: read)
   - Copy this token for Compute Canada

---

## 2️⃣ Set Up Compute Canada Environment

```bash
# Load a recent Python module
module load python/3.11

# Create a virtual environment
python -m venv llama_env # If you already have one, no need to create another
source username_venv2/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

## 3️⃣ Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # or GPU if available
pip install --no-index -r requirements.txt # Install them one by one if this cause some issues
```

>  Tip: Use `--no-cache-dir` on CC if storage is limited.

---

## 4️⃣ HuggingFace Authentication on Compute Canada

```bash
# Store your token as an environment variable
export HF_TOKEN="your_huggingface_token"

# Verify access
huggingface-cli login # You will have to enter your token
hf auth whoami
```

> [!] Note: `huggingface-cli login` is recommended to save the token on Compute Canada.

---

This LLaMA 3 model is gated on Hugging Face, which means you cannot access it without explicit permission. You may get the 403 Client Error and GatedRepoError. Even using use_auth_token=True won't help unless your Hugging Face account has been authorized.

Here's what you need to do:

Go to the model page: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

Click "Request Access". You'll need a Hugging Face account if you don't have one.

Once you're approved, you'll get access, and you can use your token in the script

---

## 5️⃣ Download the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.2-3B-Instruct"
token = "your_token"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

# Modèle optimisé pour Compute Canada
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",              # distribue sur GPU(s) automatiquement
    torch_dtype=torch.float16,      # FP16 pour économiser la VRAM
    low_cpu_mem_usage=True,         # réduit l’usage CPU
    offload_folder="offload",       # décharge certaines couches sur disque si nécessaire
    use_auth_token=token
)

```

> 💡 Tip: `device_map="auto"` and `low_cpu_mem_usage=True` allow loading the model even with 12GB VRAM.

---

```python
## 6️⃣ Verify the Model
inputs = tokenizer("Hello world!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

If it generates text without errors → everything is OK.

> 💡 Note : Steps 5 & 6 can be done using the script load_test_llama3.py (Just run sbatch load_test_llama3.sh)
---

## 7️⃣ Using This Pipeline

After completing the above setup, you can use this classification pipeline:

1. **Update your dataset path** in `data_preprocessing.py`:
```python
csv_path = "path/to/your/network_traffic_dataset.csv"
```

2. **Run the complete pipeline**:
```bash
chmod +x run_llama_training.sh
sbatch run_llama_training.sh
```

**OR run individual steps:**
```bash
python data_preprocessing.py  # Convert CSV to instruction format
python llama_finetuning.py   # Fine-tune with LoRA
python inference.py          # Test inference
python evaluate_model.py     # Comprehensive evaluation
```

---

## Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 8GB VRAM (RTX 3080/4080 or V100 recommended)
  - LLaMA-3.2-3B: ~6-8GB VRAM
  - LLaMA-3.2-7B: ~12-16GB VRAM
- **RAM**: At least 16GB system RAM
- **Storage**: 20GB+ free space for model weights and data

### Software Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- PyTorch 2.0+
- HuggingFace account with LLaMA access


## Dataset Format

The CSV dataset should contain the following columns:

### Required Feature Columns:
- `flow_duration`, `Header_Length`, `Protocol Type`, `Duration`, `Rate`
- `Srate`, `Drate`, `fin_flag_number`, `syn_flag_number`, `rst_flag_number`
- `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`
- `ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count`
- `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`, `TCP`, `UDP`
- `DHCP`, `ARP`, `ICMP`, `IPv`, `LLC`, `Tot sum`, `Min`, `Max`, `AVG`
- `Std`, `Tot size`, `IAT`, `Number`, `Magnitue`, `Radius`, `Covariance`
- `Variance`, `Weight`

### Required Label Column:
- `label`: Contains the traffic class (e.g., "DDoS", "Mirai", "XSS", "Normal")

### Example CSV structure:
```csv
flow_duration,Header_Length,Protocol Type,Duration,Rate,...,label
1000.5,20,6,500.2,1024.0,...,Normal
2000.1,40,17,800.5,2048.0,...,DDoS
...
```
---

## Detailed Usage

### Data Preprocessing

The `data_preprocessing.py` script:
- Loads your CSV dataset
- Handles missing values
- Normalizes numerical features
- Converts tabular data to instruction-following format
- Splits data into train/validation/test sets

**Output files:**
- `./data/train.json`
- `./data/validation.json`
- `./data/test.json`

### Fine-tuning Configuration

The `llama_finetuning.py` script uses the following optimized settings:

**LoRA Configuration:**
- Rank (r): 16
- Alpha: 32
- Dropout: 0.1
- Target modules: All attention and MLP layers

**Training Arguments:**
- Epochs: 3
- Batch size: 4 (per device)
- Gradient accumulation: 4 steps
- Learning rate: 2e-4
- Mixed precision: FP16

### Memory Optimization

The pipeline includes several memory optimization techniques:
- **4-bit quantization** using BitsAndBytesConfig
- **LoRA** for parameter-efficient fine-tuning
- **Gradient accumulation** to simulate larger batch sizes
- **Mixed precision training** (FP16)

---

## Inference

### Single Flow Prediction

```python
from inference import NetworkTrafficPredictor

# Initialize predictor
predictor = NetworkTrafficPredictor()
predictor.load_model()

# Define flow features
flow_features = {
    'flow_duration': 1000.5,
    'Header_Length': 20,
    'Protocol Type': 6,
    # ... other features
}

# Make prediction
prediction = predictor.predict_single_flow(flow_features)
print(f"Predicted class: {prediction}")
```

### Batch Prediction from CSV

```python
# Predict for entire CSV file
df_with_predictions = predictor.predict_from_csv(
    "test_data.csv", 
    "predictions.csv"
)
```
---

## Troubleshooting

### Common Issues

1. **Authentication Errors (403/GatedRepoError):**
   - Ensure you've been approved for LLaMA access on HuggingFace
   - Run `huggingface-cli login` and verify with `huggingface-cli whoami`
   - Check that your token has "read" permissions

2. **CUDA Out of Memory:**
   - Reduce batch size in `llama_finetuning.py`
   - Increase gradient accumulation steps
   - Use smaller LoRA rank
   - Switch to LLaMA-3.2-3B instead of 7B/11B

3. **Model Loading Issues:**
   - Check internet connection for model downloads
   - Verify cache directories are writable
   - Ensure sufficient disk space for model weights

4. **Data Format Errors:**
   - Ensure CSV has all required columns
   - Check for missing values in critical columns
   - Verify label format consistency

### Performance Optimization

1. **For Faster Training:**
   - Use V100 or A100 GPUs on Compute Canada
   - Increase batch size if memory allows
   - Set appropriate cache directories
   - Use CPU offloading for larger models

2. **For Better Accuracy:**
   - Increase training epochs
   - Tune LoRA hyperparameters (try r=32, alpha=64)
   - Balance your dataset
   - Use class weights for imbalanced data

3. **Memory Optimization:**
   - Enable gradient checkpointing
   - Use smaller sequence lengths
   - Reduce LoRA rank if needed

## Project Structure

```
llama-network-classification/
├── setup.py                 # Automated setup script
├── data_preprocessing.py     # Data conversion and preprocessing
├── llama_finetuning.py      # Main fine-tuning script
├── inference.py             # Model inference and prediction
├── evaluate_model.py        # Comprehensive model evaluation
├── config.py                # Configuration management
├── train.sh                 # Automated training pipeline
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── data/                    # Generated data files
│   ├── train.json
│   ├── validation.json
│   ├── test.json
│   └── CICIoT2023_attacks_benign_CTGAN_V2.csv      # The dataset
│
├── llama-network-classifier/ # Saved model directory
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files
│
├── evaluation_results/      # Evaluation outputs
│   ├── confusion_matrix.png
│   ├── class_performance.png
│   ├── inference_times.png
│   ├── evaluation_report.txt
│   └── detailed_results.json
│
└── logs/                    # Training logs
```

## Customization

### Modifying LoRA Parameters

In `llama_finetuning.py`, adjust the LoRA configuration:

```python
lora_config = LoraConfig(
    r=32,           # Increase for more parameters
    lora_alpha=64,  # Increase proportionally with r
    lora_dropout=0.05,  # Decrease for less regularization
    # ... other parameters
)
```

### Changing Training Parameters

Modify the `TrainingArguments` in `llama_finetuning.py`:

```python
training_args = TrainingArguments(
    num_train_epochs=5,        # More epochs
    learning_rate=1e-4,        # Lower learning rate
    per_device_train_batch_size=8,  # Larger batch size
    # ... other parameters
)
```

## Acknowledgments

- Meta AI for the LLaMA model family
- HuggingFace for the transformers library
- Microsoft for the LoRA implementation
- The open-source community for various tools and libraries
