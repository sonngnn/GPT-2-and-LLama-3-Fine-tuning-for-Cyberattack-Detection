import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import os

class NetworkTrafficDataProcessor:
    def __init__(self, csv_path):
        """
        Initialize the data processor for network traffic classification
        
        Args:
            csv_path (str): Path to the CSV file containing network traffic data
        """
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.feature_columns = [
            'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate',
            'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
            'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
            'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count',
            'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP',
            'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG',
            'Std', 'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance',
            'Variance', 'Weight'
        ]
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the network traffic data
        
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        print("Loading data from CSV...")
        df = pd.read_csv(self.csv_path)
        
        # Handle missing values
        print("Handling missing values...")
        df = df.fillna(0)
        
        # Ensure we have all required columns
        missing_cols = [col for col in self.feature_columns + ['label'] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
        
        # Normalize numerical features (excluding binary flags and protocol indicators)
        numerical_cols = [
            'flow_duration', 'Header_Length', 'Duration', 'Rate', 'Srate', 'Drate',
            'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count',
            'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
            'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
        ]
        
        # Only normalize existing numerical columns
        existing_numerical_cols = [col for col in numerical_cols if col in df.columns]
        df[existing_numerical_cols] = self.scaler.fit_transform(df[existing_numerical_cols])
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def convert_to_instruction_format(self, df, output_dir="./data"):
        """
        Convert tabular data to instruction-following format for LLaMA
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            output_dir (str): Directory to save the converted data
        
        Returns:
            list: List of instruction-formatted examples
        """
        print("Converting data to instruction format...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        instructions = []
        
        for idx, row in df.iterrows():
            # Create feature description
            features_text = []
            for col in self.feature_columns:
                if col in row:
                    features_text.append(f"{col}: {row[col]}")
            
            features_str = ", ".join(features_text)

            # Create instruction-following format with ALL real classes
            instruction = {
                "instruction": "Classify the following network traffic flow based on its characteristics. The possible classes are: Benign, BrowserHijacking, DoS-UDP_Flood, Mirai-greeth_flood, XSS, Backdoor_Malware, Uploading_Attack, Mirai-udpplain, MITM-ArpSpoofing, DNS_Spoofing, DoS-TCP_Flood, DictionaryBruteForce, Recon-OSScan, DoS-HTTP_Flood, SqlInjection, CommandInjection, Recon-PortScan, DDoS, Mirai-greip_flood, VulnerabilityScan, Recon-HostDiscovery, Recon-PingSweep, DoS-SYN_Flood.",
                "input": f"Network flow features: {features_str}",
                "output": str(row['label'])
            }
            
            instructions.append(instruction)
            
            # Progress indicator
            if (idx + 1) % 10000 == 0:
                print(f"Processed {idx + 1} samples...")
        
        print(f"Converted {len(instructions)} samples to instruction format")
        
        # Split data
        train_data, temp_data = train_test_split(instructions, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        # Save data
        self.save_json_data(train_data, os.path.join(output_dir, "train.json"))
        self.save_json_data(val_data, os.path.join(output_dir, "validation.json"))
        self.save_json_data(test_data, os.path.join(output_dir, "test.json"))
        
        print(f"Data split completed:")
        print(f"  Train: {len(train_data)} samples")
        print(f"  Validation: {len(val_data)} samples")
        print(f"  Test: {len(test_data)} samples")
        
        return train_data, val_data, test_data
    
    def save_json_data(self, data, filepath):
        """
        Save data to JSON file
        
        Args:
            data (list): List of examples
            filepath (str): Path to save the file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} samples to {filepath}")

def main():
    """
    Main function to run data preprocessing
    """
    # Update this path to your CSV file
    csv_path = "./data/CICIoT2023_attacks_benign_CTGAN_V2.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Please update the csv_path variable with the correct path to your dataset")
        return
    
    # Initialize processor
    processor = NetworkTrafficDataProcessor(csv_path)
    
    # Load and preprocess data
    df = processor.load_and_preprocess_data()
    
    # Convert to instruction format
    train_data, val_data, test_data = processor.convert_to_instruction_format(df)
    
    print("Data preprocessing completed successfully!")
    print("Files created:")
    print("  - ./data/train.json")
    print("  - ./data/validation.json") 
    print("  - ./data/test.json")

if __name__ == "__main__":
    main()