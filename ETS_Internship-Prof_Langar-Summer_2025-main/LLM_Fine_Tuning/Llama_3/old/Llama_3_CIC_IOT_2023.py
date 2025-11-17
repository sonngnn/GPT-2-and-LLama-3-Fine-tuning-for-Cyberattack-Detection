#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Llama_3_CIC_IOT_2023.py

Fine-tune Meta-Llama-3-8B-Instruct for 5G traffic flow classification / cyber-attack detection
using LoRA + 4-bit quantization.

PREREQUISITES:
1. Ensure you have a Hugging Face account and access to Meta-Llama-3-8B-Instruct model
2. Set your HF_TOKEN environment variable: export HF_TOKEN=your_token_here
3. Prepare your dataset in CSV format with 'text' and 'label' columns
4. Install required dependencies (see env_setup.sh)

USAGE:
1. Set your HF token: export HF_TOKEN=your_hugging_face_token
2. Place your dataset CSV file in the same directory (or update dataset_path in CONFIG)
3. Run: python Llama_3_CIC_IOT_2023.py

Key features:
• Secure Hugging Face authentication via environment variable
• Dynamic dtype selection (BF16 if available, otherwise FP16 – necessary on V100)
• LoRA fine-tuning with 4-bit quantization for memory efficiency
• Automatic train/validation split with stratification
"""

# ───────────────────────────────────────────────────────────────
# Imports
# ───────────────────────────────────────────────────────────────
import os
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ───────────────────────────────────────────────────────────────
# Logging setup
# ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────
# Configuration parameters
# ───────────────────────────────────────────────────────────────
CONFIG = {
    # Model and data paths
    "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "dataset_path": "CICIoT2023_attacks_benign_CTGAN_preprocessed.csv",  # Update this path to your dataset
    "output_dir": "outputs/llama3_ddos_lora",
    "val_split": 0.20,  # 20% for validation
    
    # LoRA configuration
    "lora_r": 16,           # Rank of adaptation
    "lora_alpha": 32,       # LoRA scaling parameter
    "lora_dropout": 0.1,    # Dropout probability for LoRA layers
    
    # Training parameters
    "epochs": 3,
    "per_device_train_batch_size": 1,    # Adjust based on your GPU memory
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 16,   # Effective batch size = batch_size * grad_accum_steps
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "logging_steps": 10,
    "eval_steps": 100,
    "save_steps": 100,
    "max_length": 512,                   # Maximum sequence length
    
    # 4-bit quantization settings
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",       # Normal Float 4-bit quantization
}

SEED = 42  # For reproducibility

# ───────────────────────────────────────────────────────────────
# Utility functions
# ───────────────────────────────────────────────────────────────
def get_hf_token():
    """
    Get Hugging Face token from environment variable.
    
    Returns:
        str: Hugging Face token
        
    Raises:
        ValueError: If HF_TOKEN environment variable is not set
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            "Please set it with: export HF_TOKEN=your_hugging_face_token"
        )
    return token

def load_and_prepare_dataset(path: str, val_split: float):
    """
    Load and prepare dataset from CSV file.
    
    Args:
        path (str): Path to CSV file
        val_split (float): Fraction of data to use for validation
        
    Returns:
        datasets.DatasetDict: Train/test split dataset
        
    Raises:
        ValueError: If required columns are missing or file not found
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    
    # Validate required columns
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "Dataset CSV must contain 'text' and 'label' columns. "
            f"Found columns: {list(df.columns)}"
        )
    
    # Clean data
    initial_rows = len(df)
    df = df.dropna(subset=["text", "label"])
    logger.info(f"Cleaned dataset: {initial_rows} -> {len(df)} rows")
    
    # Create dataset and split
    dataset = Dataset.from_pandas(df[["text", "label"]])
    return dataset.train_test_split(
        test_size=val_split, 
        stratify_by_column="label", 
        seed=SEED
    )

def format_instruction(example):
    """
    Format training examples using Llama-3 chat template.
    
    Args:
        example (dict): Single example with 'text' and 'label' keys
        
    Returns:
        dict: Formatted example with chat template
    """
    instruction = (
        "You are a network security expert. "
        "Analyze the following network flow and classify the traffic type.\n\n"
        f"Network Flow: {example['text']}\n\n"
        "What type of network traffic or attack is this?"
    )
    response = str(example["label"])
    
    # Llama-3 Instruct format
    formatted_text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant specialized in network security analysis."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{response}<|eot_id|>"
    )
    
    return {"text": formatted_text}

def tokenize_function(examples, tokenizer, max_length):
    """
    Tokenize examples for causal language modeling.
    
    Args:
        examples (dict): Batch of examples
        tokenizer: HuggingFace tokenizer
        max_length (int): Maximum sequence length
        
    Returns:
        dict: Tokenized examples with input_ids and labels
    """
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,  # DataCollatorForLanguageModeling handles padding
        max_length=max_length,
        add_special_tokens=False,  # Already added in format_instruction
    )
    # For causal LM, labels are the same as input_ids
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# ───────────────────────────────────────────────────────────────
# Model and tokenizer setup
# ───────────────────────────────────────────────────────────────
def setup_model_and_tokenizer():
    """
    Initialize model and tokenizer with quantization and LoRA.
    
    Returns:
        tuple: (model, tokenizer, compute_dtype, fp16_flag, bf16_flag)
    """
    hf_token = get_hf_token()
    
    # Determine optimal dtype based on GPU capabilities
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        fp16_flag, bf16_flag = False, True
        logger.info("Using BF16 precision")
    else:
        compute_dtype = torch.float16
        fp16_flag, bf16_flag = True, False
        logger.info("Using FP16 precision")

    # 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG["load_in_4bit"],
        bnb_4bit_use_double_quant=CONFIG["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=CONFIG["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model"],
        token=hf_token,
        use_fast=True,
        trust_remote_code=True,
    )
    
    # Set padding token if not available
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with quantization
    logger.info("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        token=hf_token,
        device_map="auto",
        torch_dtype=compute_dtype,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # Configure LoRA
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer, compute_dtype, fp16_flag, bf16_flag

# ───────────────────────────────────────────────────────────────
# Main training function
# ───────────────────────────────────────────────────────────────
def main():
    """Main training loop."""
    # Set seed for reproducibility
    torch.manual_seed(SEED)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting fine-tuning at {timestamp}")

    # Load and prepare dataset
    logger.info("Loading and preparing dataset...")
    dataset_split = load_and_prepare_dataset(CONFIG["dataset_path"], CONFIG["val_split"])
    logger.info(f"Train samples: {len(dataset_split['train'])}, "
                f"Validation samples: {len(dataset_split['test'])}")
    
    # Format instructions
    logger.info("Formatting instructions...")
    dataset_split = dataset_split.map(format_instruction, num_proc=4)

    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    model, tokenizer, compute_dtype, fp16_flag, bf16_flag = setup_model_and_tokenizer()

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset_split.map(
        lambda x: tokenize_function(x, tokenizer, CONFIG["max_length"]),
        batched=True,
        remove_columns=dataset_split["train"].column_names,
        num_proc=4,
        desc="Tokenizing",
    )
    
    # Log example tokenized length
    example_length = len(tokenized_dataset["train"][0]["input_ids"])
    logger.info(f"Example tokenized length: {example_length}")

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8  # For tensor core efficiency
    )

    # Setup output directory
    output_dir = Path(CONFIG["output_dir"]) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        logging_steps=CONFIG["logging_steps"],
        eval_strategy="steps",  # Fixed: was evaluation_strategy
        eval_steps=CONFIG["eval_steps"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16_flag,
        bf16=bf16_flag,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        report_to=[],  # Disable wandb/tensorboard logging
        logging_dir=str(output_dir / "logs"),
        seed=SEED,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save model and tokenizer
    logger.info("Saving model and tokenizer...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics_file = output_dir / "training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    logger.info(f"Training completed successfully!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Training metrics saved to: {metrics_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise