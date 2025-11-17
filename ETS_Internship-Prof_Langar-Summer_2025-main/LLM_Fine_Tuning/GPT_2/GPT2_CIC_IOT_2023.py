#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPT2_CIC_IOT_2023.py — v3

Fine-tune GPT-2-Large for 5G traffic flow classification / cyber-attack detection
using standard fine-tuning with FP16 precision.

PREREQUISITES:
1. Ensure you have a Hugging Face account (GPT-2 is publicly available)
2. Optionally set your HF_TOKEN environment variable for better download speeds
3. Prepare your dataset in CSV format with 'text' and 'label' columns
4. Install required dependencies (see env_setup.sh)

USAGE:
1. Optionally set your HF token: export HF_TOKEN=your_hugging_face_token
2. Place your dataset CSV file in the same directory (or update dataset_path in CONFIG)
3. Run: python GPT2_CIC_IOT_2023.py

Key features:
• Secure Hugging Face authentication via environment variable (optional for GPT-2)
• FP16 precision training for memory efficiency on V100 GPUs
• Automatic train/validation split with stratification
• Training loss visualization and metrics logging
• Fixed ClassLabel encoding for proper stratification

Fixes in v3:
• Removed hardcoded tokens for security
• Added comprehensive error handling and logging
• Fixed tokenize_function bug (missing return statement)
• Added environment variable handling for HF token
• Improved documentation and user guidance
"""

# ────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────
import json
import logging
import os
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, ClassLabel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ────────────────────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Configuration parameters
# ────────────────────────────────────────────────────────────────
CONFIG = {
    # Model and data paths
    "base_model": "gpt2-large",  # GPT-2 Large (774M parameters)
    "dataset_path": "CICIoT2023_attacks_benign_CTGAN_preprocessed.csv",  # Update this path to your dataset
    "output_dir": "outputs/gpt2_ddos",
    "val_split": 0.20,  # 20% for validation
    
    # Training parameters
    "epochs": 3,
    "per_device_train_batch_size": 4,    # Adjust based on your GPU memory
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 16,   # Effective batch size = batch_size * grad_accum_steps
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "logging_steps": 50,
    "eval_steps": 1000,
    "save_steps": 1000,
    "max_length": 512,                   # Maximum sequence length
}

# Parallelism limit for dataset processing
NUM_PROC = int(os.getenv("NUM_PROC", "8"))  # Default to 8 processes

SEED = 42  # For reproducibility
torch.manual_seed(SEED)

# ────────────────────────────────────────────────────────────────
# Utility functions
# ────────────────────────────────────────────────────────────────
def get_hf_token():
    """
    Get Hugging Face token from environment variable.
    
    Returns:
        str or None: Hugging Face token if available, None otherwise
        
    Note:
        GPT-2 is publicly available, so HF token is optional but can improve download speeds.
    """
    token = os.getenv("HF_TOKEN")
    if token:
        logger.info("Using Hugging Face token for authentication")
    else:
        logger.info("No HF_TOKEN found - using public access (this is fine for GPT-2)")
    return token

def load_and_prepare_dataset(path: str, val_split: float):
    """
    Load and prepare dataset from CSV file with stratified split.
    
    Args:
        path (str): Path to CSV file
        val_split (float): Fraction of data to use for validation
        
    Returns:
        datasets.DatasetDict: Train/test split dataset with ClassLabel encoding
        
    Raises:
        FileNotFoundError: If dataset file is not found
        ValueError: If required columns are missing
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    
    # Validate required columns
    missing_cols = {"text", "label"} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset CSV must contain 'text' and 'label' columns. "
            f"Missing columns: {missing_cols}. Found columns: {list(df.columns)}"
        )
    
    # Clean data
    initial_rows = len(df)
    df = df.dropna(subset=["text", "label"])
    logger.info(f"Cleaned dataset: {initial_rows} -> {len(df)} rows")
    
    # Create dataset
    dataset = Dataset.from_pandas(df[["text", "label"]])
    
    # Encode label as ClassLabel to enable stratification
    # This fixes the bug where stratification would fail on string labels
    if not isinstance(dataset.features["label"], ClassLabel):
        logger.info("Encoding labels as ClassLabel for stratification...")
        dataset = dataset.class_encode_column("label")
    
    # Create stratified split
    logger.info(f"Creating stratified split: {1-val_split:.0%} train, {val_split:.0%} validation")
    return dataset.train_test_split(
        test_size=val_split,
        seed=SEED,
        stratify_by_column="label",
    )

def format_instruction(example):
    """
    Format training examples for GPT-2 instruction following.
    
    Args:
        example (dict): Single example with 'text' and 'label' keys
        
    Returns:
        dict: Formatted example with instruction prompt
    """
    prompt = (
        "You are a network security expert. "
        "Analyze the following network flow and classify the traffic type.\n\n"
        f"Network Flow: {example['text']}\n\n"
        "Answer:"
    )
    # Format as instruction-response pair for causal language modeling
    formatted_text = f"{prompt} {example['label']} "
    return {"text": formatted_text}

def tokenize_function(examples, tokenizer, max_length):
    """
    Tokenize examples for causal language modeling.
    
    Args:
        examples (dict): Batch of examples
        tokenizer: HuggingFace tokenizer
        max_length (int): Maximum sequence length
        
    Returns:
        dict: Tokenized examples with input_ids
    """
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,  # DataCollatorForLanguageModeling handles padding
        max_length=max_length,
        add_special_tokens=False,  # We'll let the model handle special tokens
    )
    return tokens

# ────────────────────────────────────────────────────────────────
# Model and tokenizer setup
# ────────────────────────────────────────────────────────────────
def setup_model_and_tokenizer():
    """
    Initialize GPT-2 model and tokenizer with FP16 precision.
    
    Returns:
        tuple: (model, tokenizer, compute_dtype)
    """
    hf_token = get_hf_token()
    
    # Use FP16 for V100 compatibility and memory efficiency
    compute_dtype = torch.float16
    logger.info("Using FP16 precision for training")

    # Load tokenizer
    logger.info("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model"], 
        use_fast=True,
        token=hf_token  # Optional for GPT-2
    )
    
    # Configure tokenizer for proper padding and special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_prefix_space = True  # Important for GPT-2

    # Load model
    logger.info("Loading GPT-2 model...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=compute_dtype,
        device_map="auto",  # Automatically distribute across available GPUs
        token=hf_token  # Optional for GPT-2
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model, tokenizer, compute_dtype

# ────────────────────────────────────────────────────────────────
# Main training function
# ────────────────────────────────────────────────────────────────
def main():
    """Main training loop."""
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Starting GPT-2 fine-tuning run {timestamp}")

    # Load and prepare dataset
    logger.info("Loading and preparing dataset...")
    dataset_split = load_and_prepare_dataset(CONFIG["dataset_path"], CONFIG["val_split"])
    logger.info(f"Train samples: {len(dataset_split['train'])}, "
                f"Validation samples: {len(dataset_split['test'])}")
    
    # Format instructions
    logger.info("Formatting instructions...")
    dataset_split = dataset_split.map(
        format_instruction,
        num_proc=NUM_PROC,
        keep_in_memory=False,  # Write Arrow cache to disk to save memory
        desc="Formatting instructions"
    )

    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    model, tokenizer, compute_dtype = setup_model_and_tokenizer()

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset_split.map(
        lambda x: tokenize_function(x, tokenizer, CONFIG["max_length"]),
        batched=True,
        remove_columns=dataset_split["train"].column_names,
        num_proc=NUM_PROC,
        keep_in_memory=False,
        desc="Tokenization",
    )
    
    # Log example tokenized length
    example_length = len(tokenized_dataset["train"][0]["input_ids"])
    logger.info(f"🔎 Example tokenized length: {example_length} tokens")

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
        evaluation_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # Use FP16 for memory efficiency
        bf16=False,
        optim="adamw_torch",  # Standard AdamW optimizer
        report_to=["tensorboard"],  # Enable TensorBoard logging
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

    # Generate and save loss curves
    logger.info("Generating loss curves...")
    try:
        history = pd.DataFrame(trainer.state.log_history)
        train_loss = history.dropna(subset=["loss"])[["step", "loss"]]
        eval_loss = history.dropna(subset=["eval_loss"])[["step", "eval_loss"]]

        plt.figure(figsize=(10, 6))
        plt.plot(train_loss["step"], train_loss["loss"], label="Training loss", linewidth=2)
        plt.plot(eval_loss["step"], eval_loss["eval_loss"], label="Validation loss", linewidth=2)
        plt.xlabel("Global step")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.title("GPT-2 CIC-IoT Dataset - Training Progress")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        loss_curve_path = output_dir / "loss_curve.png"
        plt.savefig(loss_curve_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"Loss curve saved to: {loss_curve_path}")
    except Exception as e:
        logger.warning(f"Failed to generate loss curve: {e}")

    # Save training metrics
    metrics_file = output_dir / "training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    # Final summary
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Training metrics saved to: {metrics_file}")
    logger.info(f"Final training loss: {train_result.training_loss:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise