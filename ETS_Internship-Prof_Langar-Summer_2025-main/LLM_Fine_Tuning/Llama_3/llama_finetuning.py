import os
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import numpy as np

class LLaMANetworkClassifier:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", output_dir="./llama-network-classifier"):
        """
        Initialize LLaMA fine-tuning for network traffic classification
        
        Args:
            model_name (str): HuggingFace model identifier
            output_dir (str): Directory to save the fine-tuned model
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        
    def load_model_and_tokenizer(self):
        """
        Load LLaMA model and tokenizer with 8-bit quantization
        """
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print("Loading model with 8-bit quantization...")
        
        # Configure 8-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        
        print("Model and tokenizer loaded successfully!")
        
    def setup_lora(self):
        """
        Setup LoRA configuration for efficient fine-tuning
        """
        print("Setting up LoRA configuration...")
        
        # LoRA configuration optimized for LLaMA-3
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Rank - balance between efficiency and performance
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.1,  # Dropout for LoRA layers
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],  # Target all attention and MLP layers
            bias="none"  # Don't adapt bias parameters
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
    def load_and_prepare_data(self, data_dir="./data"):
        """
        Load and prepare data for training
        
        Args:
            data_dir (str): Directory containing the JSON data files
        """
        print("Loading training data...")
        
        # Load JSON data
        def load_json_data(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        train_data = load_json_data(os.path.join(data_dir, "train.json"))
        val_data = load_json_data(os.path.join(data_dir, "validation.json"))
        
        print(f"Loaded {len(train_data)} training samples")
        print(f"Loaded {len(val_data)} validation samples")
        
        # Convert to HuggingFace datasets
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
        
        print("Data preparation completed!")
        
    def tokenize_function(self, examples):
        """
        Tokenize the instruction-formatted examples
        
        Args:
            examples (dict): Batch of examples
            
        Returns:
            dict: Tokenized examples
        """
        # Format the examples as instruction-following prompts
        prompts = []
        for i in range(len(examples['instruction'])):
            prompt = f"""### Instruction:
{examples['instruction'][i]}

### Input:
{examples['input'][i]}

### Response:
{examples['output'][i]}{self.tokenizer.eos_token}"""
            prompts.append(prompt)
        
        # Tokenize
        model_inputs = self.tokenizer(
            prompts,
            max_length=1024,
            padding=False,
            truncation=True,
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    def setup_trainer(self):
        """
        Setup the HuggingFace Trainer with optimized training arguments
        """
        print("Setting up trainer...")
        
        # Training arguments optimized for LoRA fine-tuning
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,  
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,  # Effective batch size = 32
            warmup_steps=100,
            learning_rate=2e-4,  # Higher learning rate for LoRA
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=50,
            eval_steps=500,
            save_steps=500,
            eval_strategy="steps",
            save_strategy="steps",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="tensorboard",
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            optim="adamw_torch",
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            max_grad_norm=1.0,  # Gradients stability
            save_total_limit=2,  # Save only the 2 last checkpoints
            load_best_model_at_end=True,  # Save memory
            gradient_checkpointing=True,
            bf16=True if torch.cuda.is_available() else False,
        )
        
        # Data collator for sequence-to-sequence tasks
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )
        
        print("Trainer setup completed!")
    
    def train(self):
        """
        Start the fine-tuning process
        """
        print("Starting fine-tuning...")
        print("This may take several hours depending on your hardware and dataset size.")
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Start training
            self.trainer.train()
            
            # Save the final model
            print("Saving fine-tuned model...")
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            print(f"Model saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise

def main():
    """
    Main function to run the complete fine-tuning pipeline
    """
    # Initialize the classifier
    classifier = LLaMANetworkClassifier(
        model_name="meta-llama/Llama-3.2-3B-Instruct",  
        output_dir="./llama-network-classifier"
    )
    
    try:
        # Load model and tokenizer
        classifier.load_model_and_tokenizer()
        
        # Setup LoRA
        classifier.setup_lora()
        
        # Load and prepare data
        classifier.load_and_prepare_data()
        
        # Setup trainer
        classifier.setup_trainer()
        
        # Train the model
        classifier.train()
        
        print("Fine-tuning completed successfully!")
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        raise

if __name__ == "__main__":
    main()