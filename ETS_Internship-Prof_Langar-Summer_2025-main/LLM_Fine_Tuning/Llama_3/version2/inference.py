# inference.py

import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np

class NetworkTrafficPredictor:
    def __init__(self, base_model_name="meta-llama/Llama-3.2-3B-Instruct", peft_model_path="./llama-network-classifier/"):
        """
        Initialize the network traffic predictor with fine-tuned LLaMA model
        
        Args:
            base_model_name (str): Base LLaMA model identifier
            peft_model_path (str): Path to the fine-tuned LoRA weights
        """
        self.base_model_name = base_model_name
        self.peft_model_path = peft_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
    def load_model(self):
        """
        Load the fine-tuned model and tokenizer
        """
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, 
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_auth_token=True  # Use stored HuggingFace token
        )
        
        print("Loading LoRA weights...")
        self.model = PeftModel.from_pretrained(
            self.base_model,
            self.peft_model_path,
            torch_dtype=torch.bfloat16
        )
        
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

        # # ============== DEBUG PARTIE 2 ==============
        # print(f"  Prompt length: {len(prompt)} caractères")
        # print(f"  Input tokens: {len(inputs['input_ids'][0])} tokens")
        # # ============== FIN DEBUG PARTIE 2 ==============
        
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

        # # ============== DEBUG PARTIE 3 ==============
        # print(f"  Generated text brut: '{generated_text}'")
        # # ============== FIN DEBUG PARTIE 3 ==============
        
        # Extract predicted class (first word)
        predicted_class = generated_text.strip().split()[0] if generated_text.strip() else "Unknown"
        
        return predicted_class
    
    def predict_batch(self, flows_list):
        """
        Predict classes for a batch of network flows
        
        Args:
            flows_list (list): List of dictionaries containing flow features
            
        Returns:
            list: List of predicted classes
        """
        predictions = []
        
        print(f"Predicting {len(flows_list)} flows...")
        
        for i, flow in enumerate(flows_list):
            prediction = self.predict_single_flow(flow)
            predictions.append(prediction)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(flows_list)} flows")
        
        return predictions
    
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
        predictions = self.predict_batch(flows_list)
        
        # Add predictions to DataFrame
        df['predicted_class'] = predictions
        
        # Save if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        
        return df

def main():
    """
    Example usage of the network traffic predictor
    """
    # Initialize predictor
    predictor = NetworkTrafficPredictor()
    
    # Load the fine-tuned model
    predictor.load_model()
    
    # Example 1: Predict a single flow
    print("\n=== Single Flow Prediction ===")
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
    
    prediction = predictor.predict_single_flow(sample_flow)
    print(f"Predicted class: {prediction}")
    
    # Example 2: Test on real test.json data
    print("\n=== Testing on test.json (100 samples) ===")
    try:
        import json
        with open('data/test.json', 'r') as f:
            test_data = json.load(f)
        
        print(f"Loaded {len(test_data)} test samples")
        
        # Test on first 100 samples
        test_samples = test_data[:100]
        predictions = []
        true_labels = []
        
        print("Making predictions...")
        for i, example in enumerate(test_samples):
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

            # # ============== DEBUG PARTIE 1 ==============
            # print(f"DEBUG - Exemple {i}:")
            # print(f"  Features parsées: {len(flow_features)} features")
            # # ============== FIN DEBUG PARTIE 1 ==============

            prediction = predictor.predict_single_flow(flow_features)
            predictions.append(prediction)
            true_labels.append(example['output'])
            
            if (i + 1) % 25 == 0:
                print(f"Processed {i + 1}/100 samples")
        
        # Calculate accuracy
        correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
        accuracy = correct / len(predictions)
        
        print(f"\n=== Results ===")
        print(f"Samples tested: {len(predictions)}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Correct predictions: {correct}/{len(predictions)}")
        
        # Show some examples
        print(f"\nFirst 10 predictions vs true labels:")
        for i in range(min(10, len(predictions))):
            status = "[SUCCESS]" if predictions[i] == true_labels[i] else "[FAILED]"
            print(f"  {status} Predicted: {predictions[i]:<20} True: {true_labels[i]}")
            
    except FileNotFoundError:
        print("[FAILED] test.json not found in data/ directory")
    except Exception as e:
        print(f"[FAILED] Error testing on test.json: {e}")
    
    # Example 3: Predict from CSV (commented but available)
    # print("\n=== Batch Prediction from CSV ===")
    # csv_path = "path_to_your_test_data.csv"
    # if os.path.exists(csv_path):
    #     df_with_predictions = predictor.predict_from_csv(csv_path, "predictions.csv")
    #     print(f"Predictions completed for {len(df_with_predictions)} flows")
    # else:
    #     print(f"CSV file not found: {csv_path}")

if __name__ == "__main__":
    main()