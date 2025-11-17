#!/usr/bin/env python3
"""
Orchestration Pipeline for Multi-Feature LLaMA Training 
This script orchestrates dataset generation with different numbers of features
and their parallel training with intelligent resource management.

Usage:
    python orchestrate_pipeline.py --features_list 8 12 16 --max_concurrent_training 2
"""

import os
import sys
import argparse
import json
import time
import psutil
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import threading

@dataclass
class TaskConfig:
    """Configuration for pipeline tasks"""
    n_features: int
    data_dir: str
    output_dir: str
    log_dir: str
    status_file: str
    task_type: str  # 'preprocessing' or 'training'
    csv_path: str = "./data/CICIoT2023_attacks_benign_CTGAN_V2.csv"

@dataclass
class ResourceConfig:
    """System resource configuration"""
    total_memory_mb: int
    available_memory_mb: int
    max_concurrent_preprocessing: int
    max_concurrent_training: int
    gpu_available: bool

class SmartResourceManager:
    """Intelligent system resource manager"""
    
    def __init__(self, max_concurrent_training: int = 2):
        self.max_concurrent_training = max_concurrent_training
        self.update_system_info()
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def update_system_info(self):
        """Update system information"""
        memory = psutil.virtual_memory()
        self.total_memory_mb = int(memory.total / (1024 * 1024))
        self.available_memory_mb = int(memory.available / (1024 * 1024))
        
        # GPU detection
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.logger.info(f"GPU detected: {torch.cuda.get_device_name()}")
        except:
            self.gpu_available = False
    
    def get_resource_config(self) -> ResourceConfig:
        """Calculate optimal resource configuration"""
        self.update_system_info()
        
        # Calculate max parallel preprocessing tasks
        # Preprocessing is less memory intensive
        preprocessing_memory_per_task = 2000  # MB
        max_preprocessing = max(1, min(8, self.available_memory_mb // preprocessing_memory_per_task))
        
        # For training, be more conservative with GPU memory
        # Each training consumes ~15-20GB GPU on A100
        gpu_memory_per_training = 20000  # MB conservative estimate
        max_training_by_gpu = 1 if self.available_memory_mb < 40000 else 2
        
        final_max_training = min(self.max_concurrent_training, max_training_by_gpu)
        
        return ResourceConfig(
            total_memory_mb=self.total_memory_mb,
            available_memory_mb=self.available_memory_mb,
            max_concurrent_preprocessing=max_preprocessing,
            max_concurrent_training=final_max_training,
            gpu_available=self.gpu_available
        )

class TaskExecutor:
    """Execute preprocessing and training tasks"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.setup_logging()
    
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
    
    def create_directories(self, config: TaskConfig):
        """Create necessary directories"""
        for dir_path in [config.data_dir, config.output_dir, config.log_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def write_status(self, config: TaskConfig, status: str, message: str = ""):
        """Write task status"""
        status_data = {
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'n_features': config.n_features,
            'task_type': config.task_type
        }
        
        try:
            os.makedirs(os.path.dirname(config.status_file), exist_ok=True)
            with open(config.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error writing status: {e}")
    
    def check_data_exists(self, config: TaskConfig) -> bool:
        """Check if data already exists"""
        required_files = ['train.json', 'validation.json', 'dataset_metadata.json']
        
        if not os.path.exists(config.data_dir):
            return False
            
        for file in required_files:
            if not os.path.exists(os.path.join(config.data_dir, file)):
                return False
        
        # Verify metadata validity
        try:
            with open(os.path.join(config.data_dir, 'dataset_metadata.json'), 'r') as f:
                metadata = json.load(f)
                if metadata.get('n_features') == config.n_features:
                    return True
        except:
            pass
        
        return False
    
    def execute_preprocessing(self, config: TaskConfig) -> bool:
        """Execute preprocessing with feature engineering"""
        self.logger.info(f"Starting preprocessing for {config.n_features} features")
        
        # Check if data already exists
        if self.check_data_exists(config):
            self.logger.info(f"Data already exists for {config.n_features} features - skipped")
            self.write_status(config, "SKIPPED", "Data already exists")
            return True
        
        temp_dir = f"{config.data_dir}.tmp"
        
        try:
            # Clean temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            # Preprocessing command
            cmd = [
                sys.executable, "data_preprocessing.py",
                "--n_features", str(config.n_features),
                "--csv_path", self.csv_path,
                "--out_dir", temp_dir
            ]
            
            log_file = os.path.join(config.log_dir, f"preprocessing_{config.n_features}features.log")
            
            self.logger.info(f"Executing: {' '.join(cmd)}")
            
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=7200  # 2 hours timeout
                )
            
            if process.returncode == 0:
                # CRITICAL FIX: Atomic move to final directory
                if os.path.exists(config.data_dir):
                    shutil.rmtree(config.data_dir)
                shutil.move(temp_dir, config.data_dir)
                
                self.write_status(config, "SUCCESS", "Preprocessing completed")
                self.logger.info(f"Preprocessing successful for {config.n_features} features")
                return True
            else:
                self.write_status(config, "FAILED", f"Exit code: {process.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            self.write_status(config, "FAILED", "Timeout")
            self.logger.error(f"Preprocessing timeout {config.n_features} features")
            return False
        except Exception as e:
            self.write_status(config, "FAILED", f"Error: {str(e)}")
            self.logger.error(f"Preprocessing error {config.n_features}: {e}")
            return False
        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def execute_training(self, config: TaskConfig) -> bool:
        """Execute model training with improved memory management"""
        self.logger.info(f"Starting training for {config.n_features} features")
        
        # CRITICAL FIX: Verify data exists in correct directory
        if not self.check_data_exists(config):
            self.write_status(config, "FAILED", "Training data not found")
            self.logger.error(f"Missing data for training {config.n_features} features")
            return False
        
        try:
            # Training script using updated llama_finetuning.py
            training_script = f"""
import sys
import os
import torch

# Offline configuration
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# CRITICAL: Clear GPU memory before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU memory cleared. Available: {{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}} GB")

sys.path.append('.')

# Use the updated llama_finetuning.py
from llama_finetuning import LLaMANetworkClassifier

def main():
    # Get cache
    cache_dir = os.environ.get('TRANSFORMERS_CACHE')
    
    # Initialize classifier
    classifier = LLaMANetworkClassifier(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        output_dir="{config.output_dir}",
        cache_dir=cache_dir
    )
    
    try:
        print("=== Loading model and tokenizer ===")
        classifier.load_model_and_tokenizer()
        
        print("=== Setting up LoRA ===")
        classifier.setup_lora()
        
        print("=== Loading data ===")
        # Check if tokenized data exists
        tokenized_data_dir = f"./tokenized_data_{config.n_features}_features"
        
        if os.path.exists(tokenized_data_dir):
            print(f"Loading pre-tokenized data...")
            classifier.load_tokenized_data({config.n_features}, "{config.data_dir}")
        else:
            print(f"Tokenizing data for the first time...")
            classifier.load_and_prepare_data(
                data_dir="{config.data_dir}",
                max_train_samples=None,  # Full dataset for production
                max_val_samples=None     # Full dataset for production
            )
            classifier.save_tokenized_data({config.n_features}, "{config.data_dir}")
        
        print("=== Setting up trainer ===")
        classifier.setup_trainer(
            eval_steps=2000,
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        )
        
        print("=== Starting training ===")
        # Check for existing checkpoints
        checkpoint_dir = None
        if os.path.exists(classifier.output_dir):
            checkpoints = [d for d in os.listdir(classifier.output_dir) if d.startswith('checkpoint-')]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
                checkpoint_dir = os.path.join(classifier.output_dir, latest_checkpoint)
                print(f"Resuming from checkpoint: {{checkpoint_dir}}")
        
        classifier.train(resume_from_checkpoint=checkpoint_dir)
        
        print("Training completed successfully!")
        
        # Final GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
    except Exception as e:
        print(f"Training error: {{e}}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
            
            # Write temporary script
            script_file = f"/tmp/train_{config.n_features}features_{os.getpid()}.py"
            with open(script_file, 'w') as f:
                f.write(training_script)
            
            log_file = os.path.join(config.log_dir, f"training_{config.n_features}features.log")
            
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    [sys.executable, script_file],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=25200  # 7 hours
                )
            
            # Cleanup temporary script
            os.remove(script_file)
            
            if process.returncode == 0:
                self.write_status(config, "SUCCESS", "Training completed")
                self.logger.info(f"Training successful for {config.n_features} features")
                return True
            else:
                self.write_status(config, "FAILED", f"Exit code: {process.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            self.write_status(config, "FAILED", "Training timeout")
            return False
        except Exception as e:
            self.write_status(config, "FAILED", f"Training error: {str(e)}")
            return False

class PipelineOrchestrator:
    """Main pipeline orchestrator"""
    
    def __init__(self, args):
        self.args = args
        self.resource_manager = SmartResourceManager(args.max_concurrent_training)
        self.task_executor = TaskExecutor(args.csv_path)
        self.logger = logging.getLogger(__name__)
        
        # Create base directories
        for dir_name in ["data", "outputs", "logs"]:
            os.makedirs(dir_name, exist_ok=True)
    
    def create_task_configs(self, features_list: List[int], task_type: str) -> List[TaskConfig]:
        """Create task configurations"""
        configs = []
        
        for n_features in features_list:
            config = TaskConfig(
                n_features=n_features,
                data_dir=f"data/data_{n_features}_features",  # No .tmp for training
                output_dir=f"outputs/model_{n_features}_features",
                log_dir=f"logs/{n_features}_features",
                status_file=f"logs/{n_features}_features/{task_type}_status.json",
                task_type=task_type,
                csv_path=self.args.csv_path
            )
            
            self.task_executor.create_directories(config)
            configs.append(config)
        
        return configs
    
    def run_preprocessing_phase(self) -> Dict[int, bool]:
        """Execute preprocessing phase in parallel"""
        self.logger.info("=== PREPROCESSING PHASE ===")
        
        configs = self.create_task_configs(self.args.features_list, "preprocessing")
        resource_config = self.resource_manager.get_resource_config()
        
        print(f"Parallel preprocessing:")
        print(f"  - Concurrent tasks: {resource_config.max_concurrent_preprocessing}")
        print(f"  - Features to process: {self.args.features_list}")
        
        results = {}
        
        if self.args.dry_run:
            for config in configs:
                print(f"  [DRY RUN] Preprocessing {config.n_features} features")
                results[config.n_features] = True
            return results
        
        # Parallel preprocessing execution
        with ProcessPoolExecutor(max_workers=resource_config.max_concurrent_preprocessing) as executor:
            future_to_config = {
                executor.submit(self.task_executor.execute_preprocessing, config): config
                for config in configs
            }
            
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    success = future.result()
                    results[config.n_features] = success
                    status = "[SUCCESS] COMPLETED" if success else "[FAILED] FAILED"
                    print(f"Preprocessing {config.n_features} features: {status}")
                except Exception as e:
                    results[config.n_features] = False
                    print(f"Preprocessing {config.n_features} features: [FAILED] EXCEPTION - {e}")
        
        successful = [n for n, success in results.items() if success]
        failed = [n for n, success in results.items() if not success]
        
        print(f"\nPreprocessing results:")
        print(f"  [SUCCESS] Successful: {successful}")
        print(f"  [FAILED] Failed: {failed}")
        
        return results
    
    def run_training_phase(self, successful_features: List[int]) -> Dict[int, bool]:
        """Execute training phase in parallel (limited to 2)"""
        if not successful_features:
            print("No successful preprocessing - training skipped")
            return {}
        
        self.logger.info("=== TRAINING PHASE ===")
        
        configs = self.create_task_configs(successful_features, "training")
        
        print(f"Parallel training:")
        print(f"  - Concurrent tasks: {self.args.max_concurrent_training}")
        print(f"  - Features to train: {successful_features}")
        
        results = {}
        
        if self.args.dry_run:
            for config in configs:
                print(f"  [DRY RUN] Training {config.n_features} features")
                results[config.n_features] = True
            return results
        
        # Parallel training execution (limited to max_concurrent_training)
        with ProcessPoolExecutor(max_workers=self.args.max_concurrent_training) as executor:
            future_to_config = {
                executor.submit(self.task_executor.execute_training, config): config
                for config in configs
            }
            
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    success = future.result()
                    results[config.n_features] = success
                    status = "[SUCCESS] COMPLETED" if success else "[FAILED] FAILED"
                    print(f"Training {config.n_features} features: {status}")
                except Exception as e:
                    results[config.n_features] = False
                    print(f"Training {config.n_features} features: [FAILED] EXCEPTION - {e}")
        
        successful = [n for n, success in results.items() if success]
        failed = [n for n, success in results.items() if not success]
        
        print(f"\nTraining results:")
        print(f"  [SUCCESS] Successful: {successful}")
        print(f"  [FAILED] Failed: {failed}")
        
        return results
    
    def generate_report(self, prep_results: Dict[int, bool], train_results: Dict[int, bool]):
        """Generate final report"""
        print("\n" + "="*60)
        print("FINAL PIPELINE REPORT")
        print("="*60)
        
        print(f"Requested features: {self.args.features_list}")
        print(f"Source CSV: {self.args.csv_path}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        print(f"\n{'Feature':<8} {'Preprocessing':<15} {'Training':<15} {'Status'}")
        print("-" * 55)
        
        for n_features in self.args.features_list:
            prep_ok = prep_results.get(n_features, False)
            train_ok = train_results.get(n_features, False)
            
            prep_status = "[SUCCESS] OK" if prep_ok else "[FAILED] KO"
            train_status = "[SUCCESS] OK" if train_ok else ("[FAILED] KO" if prep_ok else " SKIP")
            
            if prep_ok and train_ok:
                final_status = " COMPLETE"
            elif prep_ok:
                final_status = "[!] PARTIAL"
            else:
                final_status = "[FAILED] FAILED"
            
            print(f"{n_features:<8} {prep_status:<15} {train_status:<15} {final_status}")
        
        # Statistics
        total = len(self.args.features_list)
        prep_success = sum(prep_results.values())
        train_success = sum(train_results.values())
        
        print(f"\nStatistics:")
        print(f"  Total requested: {total}")
        print(f"  Preprocessing successful: {prep_success}/{total}")
        print(f"  Training successful: {train_success}/{total}")
        print(f"  Complete pipeline: {train_success}/{total}")
        
        # Available models
        completed = [n for n in self.args.features_list if train_results.get(n, False)]
        if completed:
            print(f"\nTrained models available:")
            for n in completed:
                print(f"  - outputs/model_{n}_features/")
        
        print("="*60)
    
    def run_pipeline(self):
        """Execute complete pipeline"""
        print("Multi-Feature LLaMA Pipeline with Feature Engineering")
        print(f"Features: {self.args.features_list}")
        print(f"CSV: {self.args.csv_path}")
        
        if self.args.dry_run:
            print("DRY RUN MODE - Simulation only")
        
        # Phase 1: Parallel preprocessing
        prep_results = self.run_preprocessing_phase()
        
        # Phase 2: Parallel training (2 maximum)
        successful_features = [n for n, ok in prep_results.items() if ok]
        train_results = {}
        
        if successful_features and not self.args.preprocessing_only:
            train_results = self.run_training_phase(successful_features)
        elif self.args.preprocessing_only:
            print("Preprocessing only mode - training skipped")
        
        # Final report
        self.generate_report(prep_results, train_results)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-Feature LLaMA Pipeline with Feature Engineering")
    
    parser.add_argument("--features_list", type=int, nargs='+', required=True,
                        help="List of feature numbers (ex: 8 12 16)")
    
    parser.add_argument("--csv_path", type=str, 
                        default="./data/CICIoT2023_attacks_benign_CTGAN_V2.csv",
                        help="Path to CSV file")
    
    parser.add_argument("--max_concurrent_training", type=int, default=2,
                        help="Max number of concurrent training processes (default: 2)")
    
    parser.add_argument("--dry_run", action="store_true",
                        help="Simulation without actual execution")
    
    parser.add_argument("--preprocessing_only", action="store_true",
                        help="Preprocessing only, no training")
    
    return parser.parse_args()

def main():
    try:
        args = parse_arguments()
        
        # Validation
        if not os.path.exists(args.csv_path):
            print(f"[FAILED] CSV file not found: {args.csv_path}")
            sys.exit(1)
        
        if any(n <= 0 for n in args.features_list):
            print("[FAILED] Feature numbers must be positive")
            sys.exit(1)
        
        # Execution
        orchestrator = PipelineOrchestrator(args)
        orchestrator.run_pipeline()
        
    except KeyboardInterrupt:
        print("\n[FAILED] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[FAILED] Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()