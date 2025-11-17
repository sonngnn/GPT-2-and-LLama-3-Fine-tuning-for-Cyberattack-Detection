# llama_finetuning.py

import os
import torch
import json
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset
import numpy as np

# ===== OFFLINE CONFIGURATION =====
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
transformers.utils.logging.set_verbosity_error()

print("Offline mode activated - no internet connections will be attempted")

class LLaMANetworkClassifier:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", output_dir="./llama-network-classifier", cache_dir=None):
        """
        Initialize LLaMA fine-tuning for network traffic classification
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # FIXED: Better cache handling like v3
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            transformers_cache = os.environ.get('TRANSFORMERS_CACHE')
            model_cache_path = os.path.join(transformers_cache, "models--meta-llama--Llama-3.2-3B-Instruct") if transformers_cache else None
            
            if transformers_cache and os.path.exists(model_cache_path):
                self.cache_dir = transformers_cache
                print(f"Found model in TRANSFORMERS_CACHE: {model_cache_path}")
            else:
                self.cache_dir = os.environ.get('HF_HOME', None)
        
        if self.cache_dir:
            print(f"Using cache: {self.cache_dir}")
        else:
            print("No cache directory specified")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        
    def load_model_and_tokenizer(self):
        """
        Load LLaMA model and tokenizer WITHOUT quantization for multi-training stability
        """
        print("Loading tokenizer in offline mode...")
        
        try:
            # FIXED: Simplified tokenizer loading like v3
            tokenizer_kwargs = {
                "cache_dir": self.cache_dir,
                "padding_side": "left",
                "use_fast": True,
            }
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **tokenizer_kwargs
            )
            
            print("[SUCCESS] Tokenizer loaded successfully from cache")
            
        except Exception as e:
            print(f"[ERROR] Error loading tokenizer: {e}")
            print("Check that the model is properly cached in:", self.cache_dir)
            raise
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print("[SUCCESS] Padding token configured")
        
        print("Loading model WITHOUT quantization in offline mode...")
        
        try:
            # FIXED: No quantization for multi-training stability
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "cache_dir": self.cache_dir,
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            print("[SUCCESS] Model loaded successfully from cache (without quantization)")
            
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            print("Check that the model is properly cached in:", self.cache_dir)
            raise
        
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        
        print("[SUCCESS] Model and tokenizer loaded successfully in offline mode!")
        
    def setup_lora(self):
        """
        Setup LoRA configuration with conservative settings for multi-training
        """
        print("Setting up LoRA configuration...")
        
        # FIXED: More conservative LoRA config like v3
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # Reduced from 16 for stability
            lora_alpha=16,  # Reduced from 32 for stability
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj"  # Fewer modules like v3
            ],
            bias="none"
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
    def get_tokenized_data_dir(self, n_features, data_dir):
        """Generate tokenized data directory based on features count"""
        return f"./tokenized_data_{n_features}_features"

    def save_tokenized_data(self, n_features, data_dir):
        """Save tokenized datasets with feature-specific naming"""
        save_dir = self.get_tokenized_data_dir(n_features, data_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        self.train_dataset.save_to_disk(os.path.join(save_dir, "train_tokenized"))
        self.val_dataset.save_to_disk(os.path.join(save_dir, "val_tokenized"))
        
        metadata = {
            "n_features": n_features,
            "data_source": data_dir,
            "tokenizer_name": self.model_name,
            "max_length": 1024,
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset)
        }
        
        with open(os.path.join(save_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Tokenized data for {n_features} features saved to {save_dir}")

    def load_tokenized_data(self, n_features, data_dir):
        """Load pre-tokenized datasets for specific feature count"""
        from datasets import load_from_disk
        
        save_dir = self.get_tokenized_data_dir(n_features, data_dir)
        
        metadata_file = os.path.join(save_dir, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if metadata.get("n_features") != n_features:
                raise ValueError(f"Feature count mismatch: expected {n_features}, found {metadata.get('n_features')}")
            
            print(f"Loading tokenized data: {metadata}")
        
        self.train_dataset = load_from_disk(os.path.join(save_dir, "train_tokenized"))
        self.val_dataset = load_from_disk(os.path.join(save_dir, "val_tokenized"))
        
        print(f"Loaded tokenized data for {n_features} features from {save_dir}")
        
    def load_and_prepare_data(self, data_dir="./data", max_train_samples=None, max_val_samples=None):
        """
        Load and prepare data for training with size limits
        """
        print("Loading training data...")
        
        train_file = os.path.join(data_dir, "train.json")
        val_file = os.path.join(data_dir, "validation.json")
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not os.path.exists(val_file):
            raise FileNotFoundError(f"Validation file not found: {val_file}")
        
        def load_json_data(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        train_data = load_json_data(train_file)
        val_data = load_json_data(val_file)
        
        # Apply dataset size limits if specified
        if max_train_samples is not None and len(train_data) > max_train_samples:
            print(f"Limiting training data to {max_train_samples} samples (was {len(train_data)})")
            train_data = train_data[:max_train_samples]
        
        if max_val_samples is not None and len(val_data) > max_val_samples:
            print(f"Limiting validation data to {max_val_samples} samples (was {len(val_data)})")
            val_data = val_data[:max_val_samples]
        
        print(f"[SUCCESS] Loaded {len(train_data)} training samples")
        print(f"[SUCCESS] Loaded {len(val_data)} validation samples")
        
        self.train_dataset = Dataset.from_list(train_data)
        self.val_dataset = Dataset.from_list(val_data)
        
        # Tokenize datasets
        print("Tokenizing training data...")
        self.train_dataset = self.train_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=1000,
            remove_columns=self.train_dataset.column_names,
            desc="Tokenizing train data"
        )
        
        print("Tokenizing validation data...")
        self.val_dataset = self.val_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=1000,
            remove_columns=self.val_dataset.column_names,
            desc="Tokenizing val data"
        )
        
        print("[SUCCESS] Data preparation completed!")
        
    def tokenize_function(self, examples):
        """
        Tokenize the instruction-formatted examples
        """
        prompts = []
        for i in range(len(examples['instruction'])):
            prompt = f"""### Instruction:
{examples['instruction'][i]}

### Input:
{examples['input'][i]}

### Response:
{examples['output'][i]}{self.tokenizer.eos_token}"""
            prompts.append(prompt)
        
        model_inputs = self.tokenizer(
            prompts,
            max_length=1024,
            padding=False,
            truncation=True,
            return_tensors=None
        )
        
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    def setup_trainer(self, eval_steps=2000, early_stopping_patience=3, early_stopping_threshold=0.001, use_early_stopping=True):
        """
        Setup the HuggingFace Trainer with stable configuration
        """
        print("Setting up trainer...")
        
        # FIXED: Add pre-trainer diagnostic like v3
        print("=== PRE-TRAINER DIAGNOSTIC ===")
        try:
            print("Testing model forward pass...")
            sample = self.train_dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Input IDs shape: {len(sample['input_ids'])}")
            print("Sample processed successfully")
            
            print("Testing model state...")
            print(f"Model training mode: {self.model.training}")
            print(f"Model device: {next(self.model.parameters()).device}")
            print("Model state OK")
            
        except Exception as e:
            print(f"[ERROR] Pre-trainer diagnostic failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,  # Increased from 1 for full training
            per_device_train_batch_size=1,  
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,
            warmup_steps=100,
            learning_rate=2e-4,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=50,
            logging_strategy="steps",
            logging_first_step=True,
            eval_steps=eval_steps,
            save_steps=eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["tensorboard"],
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            optim="adamw_torch",
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            max_grad_norm=1.0,
            save_total_limit=3,  # Increased from 2
            load_best_model_at_end=True,
            gradient_checkpointing=True,
            bf16=True if torch.cuda.is_available() else False,
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # FIXED: Add early stopping like v3
        callbacks = []
        if use_early_stopping:
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold
            )
            callbacks.append(early_stopping_callback)
            print("Early stopping callback added")
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        print("[SUCCESS] Trainer setup completed!")
    
    def train(self, resume_from_checkpoint=None):
        """
        Start the fine-tuning process with enhanced error handling
        """
        print("Starting fine-tuning in offline mode...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
        try:
            # FIXED: Add initial evaluation like v3
            print("Performing initial evaluation...")
            eval_result = self.trainer.evaluate()
            print(f"Initial eval loss: {eval_result.get('eval_loss', 'N/A')}")
            
            print("Starting training process...")
            train_result = self.trainer.train(
                resume_from_checkpoint=resume_from_checkpoint,
                ignore_keys_for_eval=["hidden_states", "attentions"]
            )
            
            print(f"[SUCCESS] Training completed successfully!")
            if hasattr(train_result, 'training_loss'):
                print(f"Final training loss: {train_result.training_loss:.4f}")
            
            print("Saving fine-tuned model...")
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            print(f"[SUCCESS] Model saved to {self.output_dir}")
            
        except Exception as e:
            print(f"[ERROR] Training failed with error: {str(e)}")
            
            # Enhanced error debugging
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
            
            # GPU memory debug info
            if torch.cuda.is_available():
                print(f"\nGPU memory at error: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
            
            # Emergency save
            try:
                self.trainer.save_model(f"{self.output_dir}_emergency_save")
                print("Emergency save completed")
            except:
                print("Emergency save failed")
            
            raise

def main(n_features=8):
    """
    Main function with feature-aware tokenized data management
    """
    cache_dir = os.environ.get('TRANSFORMERS_CACHE')
    if cache_dir:
        print(f"Cache detected automatically: {cache_dir}")
    else:
        print("No automatic cache detected, using default cache")
    
    data_dir = f"./data/data_{n_features}_features"
    
    classifier = LLaMANetworkClassifier(
        model_name="meta-llama/Llama-3.2-3B-Instruct",  
        output_dir=f"./llama-network-classifier-{n_features}features",
        cache_dir=cache_dir
    )
    
    try:
        classifier.load_model_and_tokenizer()
        classifier.setup_lora()
        
        # Feature-specific tokenized data management
        tokenized_data_dir = f"./tokenized_data_{n_features}_features"
        
        if os.path.exists(tokenized_data_dir):
            print(f"Loading pre-tokenized data for {n_features} features...")
            classifier.load_tokenized_data(n_features, data_dir)
        else:
            print(f"Tokenizing data for {n_features} features for the first time...")
            classifier.load_and_prepare_data(
                data_dir=data_dir,
                max_train_samples=None,  # Full dataset for production
                max_val_samples=None     # Full dataset for production
            )
            classifier.save_tokenized_data(n_features, data_dir)
        
        classifier.setup_trainer(use_early_stopping=True)
        
        # Check for existing checkpoints
        checkpoint_dir = None
        if os.path.exists(classifier.output_dir):
            checkpoints = [d for d in os.listdir(classifier.output_dir) if d.startswith('checkpoint-')]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
                checkpoint_dir = os.path.join(classifier.output_dir, latest_checkpoint)
                print(f"Found existing checkpoint: {checkpoint_dir}")
                print("Resuming training from latest checkpoint...")
            else:
                print("No existing checkpoints found. Starting fresh training...")
        
        classifier.train(resume_from_checkpoint=checkpoint_dir)
        
        print("[SUCCESS] Fine-tuning completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Error during fine-tuning: {e}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_features", type=int, default=8, help="Number of features")
    args = parser.parse_args()
    
    main(n_features=args.n_features)