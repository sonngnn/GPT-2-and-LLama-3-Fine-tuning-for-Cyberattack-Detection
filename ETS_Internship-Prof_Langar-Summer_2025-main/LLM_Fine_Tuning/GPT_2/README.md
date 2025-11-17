# GPT-2 Fine-tuning for Cybersecurity Traffic Classification

This project fine-tunes OpenAI's GPT-2-Large model for network traffic classification and cyber-attack detection using the CIC IoT 2023 dataset. The implementation uses standard fine-tuning with FP16 precision for memory-efficient training.

## Project Overview

- **Model**: GPT-2-Large (774M parameters)
- **Dataset**: CIC IoT 2023 (network traffic with DDoS, DoS, Mirai, Recon attacks)
- **Technique**: Standard fine-tuning with FP16 precision
- **Hardware**: Optimized for V100 GPUs (16GB+ memory)
- **Framework**: PyTorch + Hugging Face Transformers

## GPT-2 vs Llama-3 Comparison

| Feature | GPT-2 | Llama-3 |
|---------|-------|---------|
| **Model Size** | 774M parameters | 8B parameters |
| **Access** | Publicly available | Requires HF approval |
| **Training Method** | Standard fine-tuning | LoRA + 4-bit quantization |
| **Memory Usage** | Lower (~16GB GPU) | Higher (~32GB GPU) |
| **Training Speed** | Faster | Slower |
| **Performance** | Good for smaller tasks | Better for complex tasks |
| **Setup Complexity** | Simpler | More complex |

## Prerequisites

### 1. Hugging Face Access (Optional)
- GPT-2 is publicly available, no special access required
- Optional: Create a [Hugging Face account](https://huggingface.co/) for faster downloads
- Optional: Generate a token from [HF Settings](https://huggingface.co/settings/tokens)

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
- CUDA-capable GPU (recommended: V100 with 16GB+ memory)
- Python 3.11+
- Compute Canada/Digital Research Alliance access (for the provided scripts)

## Quick Start

### Step 1: Environment Setup
```bash
# Run the environment setup script
bash env_setup.sh

# Activate the virtual environment
source gpt2env/bin/activate
```

### Step 2: (Optional) Set Hugging Face Token
```bash
# Optional for GPT-2 - can improve download speeds
export HF_TOKEN=your_hugging_face_token_here
```

### Step 3: Prepare Dataset
Place your dataset file as `CICIoT2023_attacks_benign_CTGAN_preprocessed.csv` in the project directory, or update the path in the configuration.

### Step 4: Configure Training
Edit the `CONFIG` dictionary in `GPT2_CIC_IOT_2023.py` to adjust:
- Batch sizes (based on your GPU memory)
- Learning rate and training epochs
- Dataset path and output directory

### Step 5: Submit Training Job
```bash
# Update email and paths in test_gpt2.sh
# Then submit to SLURM
sbatch test_gpt2.sh

# Or run directly
python GPT2_CIC_IOT_2023.py
```

## File Structure

```
project/
├── GPT2_CIC_IOT_2023.py             # Main training script
├── test_gpt2.sh                          # SLURM job submission script
├── env_setup.sh                          # Environment setup script
├── CICIoT2023_attacks_benign_CTGAN_preprocessed.csv  # Your dataset (not included)
├── requirements.txt                      # Python dependencies (generated)
├── training.log                          # Training logs
└── outputs/                              # Training outputs
    └── gpt2_ddos/
        └── run_YYYYMMDD_HHMMSS/
            ├── pytorch_model.bin         # Fine-tuned model
            ├── tokenizer.json            # Tokenizer files
            ├── training_metrics.json     # Training metrics
            ├── loss_curve.png           # Training progress visualization
            └── logs/                    # TensorBoard logs
```

## ⚙️ Configuration Options

### Training Parameters
```python
CONFIG = {
    "base_model": "gpt2-large",              # GPT-2 variant
    "dataset_path": "your_dataset.csv",      # Path to your dataset
    "output_dir": "outputs/gpt2_ddos",       # Output directory
    "val_split": 0.20,                       # Validation split
    
    # Training settings
    "epochs": 3,                             # Number of training epochs
    "per_device_train_batch_size": 4,        # Batch size per GPU
    "gradient_accumulation_steps": 16,       # Effective batch size = 4 * 16 = 64
    "learning_rate": 2e-4,                   # Learning rate
    "max_length": 512,                       # Maximum sequence length
}
```

### Hardware Requirements
- **Minimum**: CUDA GPU with 12GB+ memory
- **Recommended**: V100 with 16GB+ memory
- **CPU**: 8+ cores for data processing
- **RAM**: 32GB+ system memory
- **Storage**: 10GB+ free space for model and outputs

## Training Process

1. **Data Loading**: CSV file is loaded and validated
2. **Preprocessing**: Text is formatted with instruction prompts
3. **Tokenization**: Text is converted to GPT-2 tokens
4. **Training**: Standard fine-tuning with FP16 precision
5. **Validation**: Regular evaluation on held-out data
6. **Visualization**: Loss curves and metrics are generated
7. **Saving**: Model, tokenizer, and metrics are saved

## Troubleshooting

### Common Issues

**Out of Memory Error**
```bash
# Reduce batch size in CONFIG
"per_device_train_batch_size": 2,  # Reduce from 4 to 2
```

**Dataset Loading Error**
- Ensure CSV has 'text' and 'label' columns
- Check for missing values or encoding issues
- Verify file path is correct

**CUDA Not Available**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
# Ensure proper module loading in SLURM script
```

**Slow Training**
- Verify GPU is being used
- Check batch size and gradient accumulation
- Monitor GPU utilization with `nvidia-smi`

### Performance Optimization

**Memory Optimization**
- Use FP16 precision (enabled by default)
- Enable gradient checkpointing (enabled by default)
- Reduce sequence length if needed
- Use gradient accumulation for larger effective batch sizes

**Speed Optimization**
- Increase batch size if memory allows
- Use multiple GPUs if available
- Optimize data loading with more workers

## Expected Results

**Training Time** (V100, default settings):
- ~2-4 hours for 3 epochs on typical dataset
- Depends on dataset size and sequence length

**Memory Usage**:
- ~12-16GB GPU memory with default settings
- Scales with batch size and sequence length

**Performance Metrics**:
- Training loss should decrease steadily
- Validation loss should follow training loss
- Monitor for overfitting if gap increases

## 🔍 Model Evaluation

After training, evaluate your model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model
model_path = "outputs/gpt2_ddos/run_YYYYMMDD_HHMMSS"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Test with new data
prompt = "You are a network security expert. Analyze the following network flow..."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Contributing

Feel free to:
- Report issues or bugs
- Suggest improvements
- Add new features
- Improve documentation

## License

This project is provided for educational and research purposes. Please ensure compliance with:
- OpenAI's usage policies for GPT-2
- CIC IoT 2023 dataset license
- Your institution's research policies

## Citation

If you use this code in your research, please consider citing:
- The original GPT-2 paper by OpenAI
- The CIC IoT 2023 dataset creators
- This implementation (if helpful for your work)

## Support

For help and questions:
1. Check this README first
2. Look at the training logs for error messages
3. Verify your environment setup
4. Check GPU memory and CUDA availability
5. Consult Hugging Face Transformers documentation