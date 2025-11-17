# evaluate_model.py

import torch
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, base_model_name="meta-llama/Llama-3.2-3B-Instruct", peft_model_path="./checkpoint-2500/"):
        """
        Initialize model evaluator
        
        Args:
            base_model_name (str): Base LLaMA model identifier
            peft_model_path (str): Path to fine-tuned LoRA weights
        """
        self.base_model_name = base_model_name
        self.peft_model_path = peft_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        print("Loading model and tokenizer...")

        # Vérifier l'état GPU avant de commencer
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated")
            print(f"GPU Memory: {torch.cuda.memory_reserved()/1024**3:.1f}GB reserved")

        # Nettoyer la mémoire GPU d'abord
        torch.cuda.empty_cache()
        
        # Load tokenizer - CORRECTED with offline mode
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            padding_side="left",
            local_files_only=True,  
            token=True              
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model - CORRECTED with offline mode
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,  
            token=True              
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(
            self.base_model,
            self.peft_model_path,
            torch_dtype=torch.bfloat16
        )
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def predict_single_flow(self, flow_features):
        """
        Predict the class of a single network flow
        
        Args:
            flow_features (dict): Dictionary containing flow features
            
        Returns:
            str: Predicted class
        """
        # Format features as text
        features_text = []
        for key, value in flow_features.items():
            features_text.append(f"{key}: {value}")
        
        features_str = ", ".join(features_text)
        
        # Create prompt with EXACT same instruction as training
        prompt = f"""### Instruction:
Classify the following network traffic flow based on its characteristics. The possible classes are: Benign, BrowserHijacking, DoS-UDP_Flood, Mirai-greeth_flood, XSS, Backdoor_Malware, Uploading_Attack, Mirai-udpplain, MITM-ArpSpoofing, DNS_Spoofing, DoS-TCP_Flood, DictionaryBruteForce, Recon-OSScan, DoS-HTTP_Flood, SqlInjection, CommandInjection, Recon-PortScan, DDoS, Mirai-greip_flood, VulnerabilityScan, Recon-HostDiscovery, Recon-PingSweep, DoS-SYN_Flood.

### Input:
Network flow features: {features_str}

### Response:
"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode prediction
        generated_text = self.tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):], 
            skip_special_tokens=True
        )
        
        # Extract predicted class (first word)
        predicted_class = generated_text.strip().split()[0] if generated_text.strip() else "Unknown"
        
        return predicted_class
        
    def predict_batch(self, test_data, batch_size=8):
        """
        Make predictions on test data
        
        Args:
            test_data (list): List of test examples
            batch_size (int): Batch size for processing
            
        Returns:
            tuple: (predictions, true_labels, inference_times)
        """
        predictions = []
        true_labels = []
        inference_times = []
        
        print(f"Making predictions on {len(test_data)} samples...")
        
        for i in tqdm(range(0, len(test_data), batch_size)):
            batch = test_data[i:i+batch_size]
            
            for example in batch:
                start_time = time.time()
                
                # Parse features from the example input
                features_str = example['input'].replace('Network flow features: ', '')
                feature_pairs = features_str.split(', ')
                
                flow_features = {}
                for pair in feature_pairs:
                    if ': ' in pair:
                        key, value = pair.split(': ', 1)
                        try:
                            flow_features[key] = float(value)
                        except ValueError:
                            flow_features[key] = value

                # Use the single flow prediction method (consistent with inference.py)
                predicted_class = self.predict_single_flow(flow_features)
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                predictions.append(predicted_class)
                true_labels.append(example['output'])
                inference_times.append(inference_time)
        
        return predictions, true_labels, inference_times
    
    def predict_from_csv(self, csv_path, output_path=None):
        """
        Predict classes for flows in a CSV file
        
        Args:
            csv_path (str): Path to CSV file containing flow data
            output_path (str): Path to save predictions (optional)
            
        Returns:
            pd.DataFrame: DataFrame with original data and predictions
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Define feature columns (same as in preprocessing)
        feature_columns = [
            'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate',
            'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
            'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
            'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count',
            'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP',
            'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG',
            'Std', 'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance',
            'Variance', 'Weight'
        ]
        
        # Convert to list of dictionaries
        flows_list = []
        for _, row in df.iterrows():
            flow_features = {}
            for col in feature_columns:
                if col in df.columns:
                    flow_features[col] = row[col]
            flows_list.append(flow_features)
        
        # Make predictions
        predictions = []
        for i, flow in enumerate(flows_list):
            prediction = self.predict_single_flow(flow)
            predictions.append(prediction)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(flows_list)} flows")
        
        # Add predictions to DataFrame
        df['predicted_class'] = predictions
        
        # Save if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        
        return df
    
    def calculate_metrics(self, true_labels, predictions):
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            true_labels (list): True class labels
            predictions (list): Predicted class labels
            
        Returns:
            dict: Dictionary containing all metrics
        """
        # Get unique labels
        unique_labels = sorted(list(set(true_labels + predictions)))
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            true_labels, predictions, labels=unique_labels, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'support_total': sum(support_per_class),
            'unique_labels': unique_labels,
            'precision_per_class': dict(zip(unique_labels, precision_per_class)),
            'recall_per_class': dict(zip(unique_labels, recall_per_class)),
            'f1_per_class': dict(zip(unique_labels, f1_per_class)),
            'support_per_class': dict(zip(unique_labels, support_per_class)),
            'confusion_matrix': cm
        }
        
        return metrics
    
    def plot_confusion_matrix(self, confusion_matrix, labels, save_path="confusion_matrix.png"):
        """
        Plot and save confusion matrix
        
        Args:
            confusion_matrix (np.ndarray): Confusion matrix
            labels (list): Class labels
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(15, 12))
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Labels', fontsize=14)
        plt.ylabel('True Labels', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to avoid display issues on cluster
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_class_performance(self, metrics, save_path="class_performance.png"):
        """
        Plot per-class performance metrics
        
        Args:
            metrics (dict): Metrics dictionary from calculate_metrics
            save_path (str): Path to save the plot
        """
        labels = metrics['unique_labels']
        precision_scores = [metrics['precision_per_class'][label] for label in labels]
        recall_scores = [metrics['recall_per_class'][label] for label in labels]
        f1_scores = [metrics['f1_per_class'][label] for label in labels]
        
        x = np.arange(len(labels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        bars1 = ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes', fontsize=14)
        ax.set_ylabel('Scores', fontsize=14)
        ax.set_title('Per-Class Performance Metrics', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to avoid display issues on cluster
        print(f"Class performance plot saved to {save_path}")
    
    def plot_inference_time_distribution(self, inference_times, save_path="inference_times.png"):
        """
        Plot distribution of inference times
        
        Args:
            inference_times (list): List of inference times
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(inference_times, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Inference Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Inference Times')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(inference_times)
        plt.ylabel('Inference Time (seconds)')
        plt.title('Inference Time Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to avoid display issues on cluster
        print(f"Inference time distribution saved to {save_path}")
    
    def generate_evaluation_report(self, metrics, inference_times, save_path="evaluation_report.txt"):
        """
        Generate a comprehensive evaluation report
        
        Args:
            metrics (dict): Metrics dictionary
            inference_times (list): List of inference times
            save_path (str): Path to save the report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("LLAMA NETWORK TRAFFIC CLASSIFICATION - EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall performance
        report_lines.append("OVERALL PERFORMANCE:")
        report_lines.append("-" * 40)
        report_lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
        report_lines.append(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
        report_lines.append(f"Weighted Recall: {metrics['recall_weighted']:.4f}")
        report_lines.append(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
        report_lines.append(f"Total Samples: {metrics['support_total']}")
        report_lines.append("")
        
        # Per-class performance
        report_lines.append("PER-CLASS PERFORMANCE:")
        report_lines.append("-" * 40)
        report_lines.append(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        report_lines.append("-" * 70)
        
        for label in metrics['unique_labels']:
            precision = metrics['precision_per_class'][label]
            recall = metrics['recall_per_class'][label]
            f1 = metrics['f1_per_class'][label]
            support = metrics['support_per_class'][label]
            
            report_lines.append(f"{label:<20} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
        
        report_lines.append("")
        
        # Inference performance
        report_lines.append("INFERENCE PERFORMANCE:")
        report_lines.append("-" * 40)
        report_lines.append(f"Average Inference Time: {np.mean(inference_times):.4f} seconds")
        report_lines.append(f"Median Inference Time: {np.median(inference_times):.4f} seconds")
        report_lines.append(f"Min Inference Time: {np.min(inference_times):.4f} seconds")
        report_lines.append(f"Max Inference Time: {np.max(inference_times):.4f} seconds")
        report_lines.append(f"Standard Deviation: {np.std(inference_times):.4f} seconds")
        report_lines.append(f"Throughput: {len(inference_times)/sum(inference_times):.2f} samples/second")
        report_lines.append("")
        
        # Model information
        report_lines.append("MODEL INFORMATION:")
        report_lines.append("-" * 40)
        report_lines.append(f"Base Model: {self.base_model_name}")
        report_lines.append(f"Fine-tuned Model Path: {self.peft_model_path}")
        report_lines.append(f"Device: {self.device}")
        report_lines.append("")
        
        # Class distribution
        report_lines.append("CLASS DISTRIBUTION:")
        report_lines.append("-" * 40)
        total_samples = sum(metrics['support_per_class'].values())
        for label in metrics['unique_labels']:
            count = metrics['support_per_class'][label]
            percentage = (count / total_samples) * 100
            report_lines.append(f"{label}: {count} samples ({percentage:.1f}%)")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Save report
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Print report
        print('\n'.join(report_lines))
        print(f"\nEvaluation report saved to {save_path}")
    
    def test_single_sample(self):
        """Test on a single sample first"""
        print("\n=== Testing Single Sample ===")
        
        # Sample flow features
        sample_flow = {
            'flow_duration': -0.018,
            'Header_Length': -0.427,
            'Protocol Type': 6.5,
            'Duration': -0.845,
            'Rate': -0.073,
            'Srate': -0.069,
            'Drate': -0.069,
            'fin_flag_number': -0.5,
            'syn_flag_number': 0.5,
            'rst_flag_number': -0.5,
            'psh_flag_number': 0.5,
            'ack_flag_number': 0.5,
            'ece_flag_number': -0.5,
            'cwr_flag_number': -0.5,
            'ack_count': -0.2,
            'syn_count': 0.1,
            'fin_count': -0.1,
            'urg_count': -0.05,
            'rst_count': -0.05,
            'HTTP': 0.8,
            'HTTPS': -0.2,
            'DNS': -0.1,
            'Telnet': -0.1,
            'SMTP': -0.1,
            'SSH': -0.1,
            'IRC': -0.1,
            'TCP': 0.9,
            'UDP': -0.3,
            'DHCP': -0.1,
            'ARP': -0.1,
            'ICMP': -0.1,
            'IPv': 0.8,
            'LLC': -0.1,
            'Tot sum': 0.3,
            'Min': -0.4,
            'Max': 0.6,
            'AVG': 0.1,
            'Std': -0.2,
            'Tot size': 0.4,
            'IAT': -0.8,
            'Number': 0.2,
            'Magnitue': 0.15,
            'Radius': 0.08,
            'Covariance': 0.07,
            'Variance': 0.08,
            'Weight': 0.1
        }
        
        start_time = time.time()
        prediction = self.predict_single_flow(sample_flow)
        end_time = time.time()
        
        print(f"Predicted class: {prediction}")
        print(f"Inference time: {end_time - start_time:.4f} seconds")
        print("Single sample test completed successfully!")
        
        return prediction
    
    def evaluate_model_comprehensive(self, test_data_path="./data/test.json", 
                                   output_dir="./evaluation_results-10000",
                                   max_samples=None):
        """
        Run comprehensive model evaluation
        
        Args:
            test_data_path (str): Path to test data
            output_dir (str): Directory to save evaluation results
            max_samples (int): Maximum number of samples to evaluate (None for all)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Starting comprehensive model evaluation...")
        
        # Test single sample first
        self.test_single_sample()
        
        # Load test data
        print("Loading test data...")
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        if max_samples:
            test_data = test_data[:max_samples]
            print(f"Using first {max_samples} samples for evaluation")
        
        print(f"Loaded {len(test_data)} test samples")
        
        # Make predictions
        predictions, true_labels, inference_times = self.predict_batch(test_data)
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = self.calculate_metrics(true_labels, predictions)
        
        # Generate plots (no display on cluster)
        print("Generating visualizations...")
        
        # Confusion matrix
        self.plot_confusion_matrix(
            metrics['confusion_matrix'], 
            metrics['unique_labels'],
            save_path=os.path.join(output_dir, "confusion_matrix.png")
        )
        
        # Class performance
        self.plot_class_performance(
            metrics,
            save_path=os.path.join(output_dir, "class_performance.png")
        )
        
        # Inference times
        self.plot_inference_time_distribution(
            inference_times,
            save_path=os.path.join(output_dir, "inference_times.png")
        )
        
        # Generate report
        self.generate_evaluation_report(
            metrics, 
            inference_times,
            save_path=os.path.join(output_dir, "evaluation_report.txt")
        )
        
        # Save detailed results
        # Save detailed results avec conversion numpy complète
        def convert_numpy_types(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        results = {
            'predictions': predictions,
            'true_labels': true_labels,
            'inference_times': inference_times,
            'metrics': convert_numpy_types({k: v for k, v in metrics.items() if k != 'confusion_matrix'}),
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }

        with open(os.path.join(output_dir, "detailed_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Show some prediction examples
        print(f"\n=== Sample Predictions ===")
        for i in range(min(10, len(predictions))):
            status = "[SUCCESS]" if predictions[i] == true_labels[i] else "[FAILED]"
            print(f"  {status} Predicted: {predictions[i]:<20} True: {true_labels[i]}")
        
        print(f"\nEvaluation completed! Results saved to {output_dir}")
        
        return metrics, inference_times

def main():
    """
    Main function to run model evaluation
    """
    # Initialize evaluator with correct checkpoint path
    evaluator = ModelEvaluator(peft_model_path="./checkpoint-2500/")
    
    # Load model
    evaluator.load_model()
    
    # Run comprehensive evaluation (limit to 10000 samples for testing)
    metrics, inference_times = evaluator.evaluate_model_comprehensive(max_samples=10000)
    # metrics, inference_times = evaluator.evaluate_model_comprehensive()

    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
    print(f"Average Inference Time: {np.mean(inference_times):.4f}s")
    print(f"Throughput: {len(inference_times)/sum(inference_times):.2f} samples/second")
    print("="*50)

if __name__ == "__main__":
    main()
