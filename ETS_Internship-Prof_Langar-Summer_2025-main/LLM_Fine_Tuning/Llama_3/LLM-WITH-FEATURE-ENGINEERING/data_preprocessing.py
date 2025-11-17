#!/usr/bin/env python3
"""
Network Traffic Data Preprocessing Module

This script processes network traffic data and converts it to instruction format for LLaMA training.
It supports multiple feature engineering configurations and generates feature-specific datasets.

Key Features:
- Feature engineering integration with configurable feature counts
- Atomic dataset generation (temp directory -> final directory)
- Comprehensive logging and error handling
- CLI interface for orchestrated execution

Usage:
    python data_preprocessing.py --n_features 8 --out_dir data/data_8_features
    python data_preprocessing.py --n_features 12 --csv_path custom_data.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import os
import argparse
import sys
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional

# Import feature engineering module with error handling
try:
    from feature_engineering import CICIoTFeatureEngineer
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Feature engineering module not available: {e}")
    print("Falling back to original feature processing")
    FEATURE_ENGINEERING_AVAILABLE = False

class NetworkTrafficDataProcessor:
    """
    Enhanced data processor for network traffic classification with feature engineering integration
    
    This processor integrates with the feature engineering pipeline to generate datasets
    with specific numbers of features for comparative model training.
    """
    
    def __init__(self, csv_path: str, n_features: int = None):
        """
        Initialize the data processor for network traffic classification
        
        Args:
            csv_path (str): Path to the CSV file containing network traffic data
            n_features (int, optional): Target number of features for feature engineering
        """
        self.csv_path = csv_path
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.logger = self.setup_logging()
        
        # Feature engineering configuration
        if n_features and FEATURE_ENGINEERING_AVAILABLE:
            self.feature_config = self.create_feature_config(n_features)
            self.feature_engineer = CICIoTFeatureEngineer(self.feature_config)
            self.logger.info(f"Feature engineering configured for {n_features} features")
        else:
            self.feature_engineer = None
            if n_features:
                self.logger.warning(f"Feature engineering requested ({n_features} features) but not available - using original features")
        
        # Original feature columns (backup for non-engineered processing)
        self.original_feature_columns = [
            'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate',
            'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
            'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
            'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count',
            'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP',
            'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG',
            'Std', 'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance',
            'Variance', 'Weight'
        ]
    
    def setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def create_feature_config(self, n_features: int):
        """Create feature engineering configuration based on target feature count"""
        
        # Configuration class for feature engineering
        class FeatureConfig:
            def __init__(self, n_features):
                self.n_features_final = n_features
                self.n_features_statistical = min(n_features + 10, 50)
                self.variance_threshold = 0.01
                self.seed = 42
        
        return FeatureConfig(n_features)
    
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess the network traffic data with optional feature engineering
        
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        self.logger.info(f"Starting dataset processing for {self.n_features or 'all'} features")
        self.logger.info(f"Loading data from CSV: {self.csv_path}")
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Handle missing values
        self.logger.info("Handling missing values...")
        df = df.fillna(0)
        
        # Check for required label column
        if 'label' not in df.columns:
            raise ValueError("Dataset must contain a 'label' column")
        
        # Apply feature engineering if configured
        if self.feature_engineer and self.n_features:
            self.logger.info(f"Applying feature engineering for {self.n_features} features...")
            try:
                # Use feature engineering pipeline
                X_engineered, y = self.feature_engineer.fit_transform(df)
                
                # Create new dataframe with engineered features
                feature_names = self.feature_engineer.selected_features
                df_engineered = pd.DataFrame(X_engineered, columns=feature_names)
                df_engineered['label'] = y
                
                self.logger.info(f"Feature engineering completed: {len(feature_names)} features selected")
                self.logger.info(f"Selected features: {feature_names}")
                
                # Verify we have exactly the requested number of features
                if len(feature_names) != self.n_features:
                    raise ValueError(f"Feature engineering returned {len(feature_names)} features but {self.n_features} were requested")
                
                # Store feature analysis
                if hasattr(self.feature_engineer, 'get_feature_importance_report'):
                    importance_report = self.feature_engineer.get_feature_importance_report()
                    self.logger.info("Feature importance report generated")
                
                return df_engineered
                
            except Exception as e:
                self.logger.error(f"Feature engineering failed: {e}")
                self.logger.warning("Falling back to original feature processing...")
                return self.process_original_features(df)
        else:
            # Use original feature processing
            return self.process_original_features(df)
    
    def process_original_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data using original feature set without engineering"""
        self.logger.info("Processing with original feature set...")
        
        # Ensure we have required columns
        available_features = [col for col in self.original_feature_columns if col in df.columns]
        missing_cols = [col for col in self.original_feature_columns if col not in df.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing columns: {missing_cols}")
        
        # If n_features is specified, select top features
        if self.n_features and self.n_features < len(available_features):
            self.logger.info(f"Selecting top {self.n_features} features from {len(available_features)} available")
            # Use the first n_features from available features (you could implement more sophisticated selection here)
            selected_features = available_features[:self.n_features]
        else:
            selected_features = available_features
        
        self.logger.info(f"Using {len(selected_features)} features")
        
        # Normalize numerical features
        numerical_cols = [
            'flow_duration', 'Header_Length', 'Duration', 'Rate', 'Srate', 'Drate',
            'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count',
            'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
            'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
        ]
        
        # Only normalize existing numerical columns that are in selected features
        existing_numerical_cols = [col for col in numerical_cols if col in selected_features]
        if existing_numerical_cols:
            df[existing_numerical_cols] = self.scaler.fit_transform(df[existing_numerical_cols])
        
        # Keep only selected features and label
        final_columns = selected_features + ['label']
        df_processed = df[final_columns].copy()
        
        self.logger.info(f"Data processed successfully. Shape: {df_processed.shape}")
        self.logger.info(f"Label distribution:\n{df_processed['label'].value_counts()}")
        
        return df_processed
    
    def convert_to_instruction_format(self, df: pd.DataFrame, output_dir: str = "./data") -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Convert tabular data to instruction-following format for LLaMA
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            output_dir (str): Directory to save the converted data
        
        Returns:
            Tuple[List[Dict], List[Dict], List[Dict]]: Train, validation, and test data
        """
        self.logger.info("Converting data to instruction format...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get feature columns (all except label)
        feature_columns = [col for col in df.columns if col != 'label']
        
        instructions = []
        
        self.logger.info(f"Processing {len(df)} samples with {len(feature_columns)} features...")
        
        for idx, row in df.iterrows():
            # Create feature description
            features_text = []
            for col in feature_columns:
                if col in row:
                    features_text.append(f"{col}: {row[col]}")
            
            features_str = ", ".join(features_text)

            # Create instruction-following format with all real classes
            instruction = {
                "instruction": "Classify the following network traffic flow based on its characteristics. The possible classes are: Benign, BrowserHijacking, DoS-UDP_Flood, Mirai-greeth_flood, XSS, Backdoor_Malware, Uploading_Attack, Mirai-udpplain, MITM-ArpSpoofing, DNS_Spoofing, DoS-TCP_Flood, DictionaryBruteForce, Recon-OSScan, DoS-HTTP_Flood, SqlInjection, CommandInjection, Recon-PortScan, DDoS, Mirai-greip_flood, VulnerabilityScan, Recon-HostDiscovery, Recon-PingSweep, DoS-SYN_Flood.",
                "input": f"Network flow features: {features_str}",
                "output": str(row['label'])
            }
            
            instructions.append(instruction)
            
            # Progress indicator
            if (idx + 1) % 10000 == 0:
                self.logger.info(f"Processed {idx + 1} samples...")
        
        self.logger.info(f"Converted {len(instructions)} samples to instruction format")
        
        # Split data with stratification to maintain class distribution
        train_data, temp_data = train_test_split(
            instructions, 
            test_size=0.3, 
            random_state=42,
            stratify=[item['output'] for item in instructions]
        )
        val_data, test_data = train_test_split(
            temp_data, 
            test_size=0.5, 
            random_state=42,
            stratify=[item['output'] for item in temp_data]
        )
        
        # Save data atomically
        self.save_json_data(train_data, os.path.join(output_dir, "train.json"))
        self.save_json_data(val_data, os.path.join(output_dir, "validation.json"))
        self.save_json_data(test_data, os.path.join(output_dir, "test.json"))
        
        # Save dataset metadata
        metadata = {
            "total_samples": len(instructions),
            "train_samples": len(train_data),
            "validation_samples": len(val_data),
            "test_samples": len(test_data),
            "n_features": len(feature_columns),
            "feature_names": feature_columns,
            "class_distribution": df['label'].value_counts().to_dict(),
            "csv_source": self.csv_path,
            "feature_engineering_applied": self.n_features is not None and self.feature_engineer is not None
        }
        
        metadata_path = os.path.join(output_dir, "dataset_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Dataset metadata saved to {metadata_path}")
        
        self.logger.info(f"Data split completed:")
        self.logger.info(f"  Train: {len(train_data)} samples ({len(train_data)/len(instructions)*100:.1f}%)")
        self.logger.info(f"  Validation: {len(val_data)} samples ({len(val_data)/len(instructions)*100:.1f}%)")
        self.logger.info(f"  Test: {len(test_data)} samples ({len(test_data)/len(instructions)*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def save_json_data(self, data: List[Dict], filepath: str) -> None:
        """
        Save data to JSON file with error handling
        
        Args:
            data (List[Dict]): List of examples
            filepath (str): Path to save the file
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved {len(data)} samples to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save data to {filepath}: {e}")
            raise
    
    def process_dataset(self, output_dir: str) -> bool:
        """
        Complete dataset processing pipeline
        
        Args:
            output_dir (str): Output directory for processed data
            
        Returns:
            bool: Success status
        """
        try:
            # Load and preprocess data
            df = self.load_and_preprocess_data()
            
            # Convert to instruction format
            train_data, val_data, test_data = self.convert_to_instruction_format(df, output_dir)
            
            self.logger.info("Dataset processing completed successfully!")
            self.logger.info("Files created:")
            self.logger.info(f"  - {output_dir}/train.json")
            self.logger.info(f"  - {output_dir}/validation.json")
            self.logger.info(f"  - {output_dir}/test.json")
            self.logger.info(f"  - {output_dir}/dataset_metadata.json")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset processing failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Process network traffic data with configurable feature engineering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with 8 features using feature engineering
  python data_preprocessing.py --n_features 8 --out_dir data/data_8_features
  
  # Process with custom CSV path
  python data_preprocessing.py --n_features 12 --csv_path custom_data.csv --out_dir data/data_12_features
  
  # Process without feature engineering (use all original features)
  python data_preprocessing.py --out_dir data/data_all_features
        """
    )
    
    parser.add_argument(
        "--n_features",
        type=int,
        help="Target number of features for feature engineering (optional)"
    )
    
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./data/CICIoT2023_attacks_benign_CTGAN_V2.csv",
        help="Path to input CSV file (default: ./data/CICIoT2023_attacks_benign_CTGAN_V2.csv)"
    )
    
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./data",
        help="Output directory for processed data (default: ./data)"
    )
    
    return parser.parse_args()

def main():
    """
    Main function to run data preprocessing with CLI support
    """
    try:
        args = parse_arguments()
        
        # Validate arguments
        if args.n_features is not None and args.n_features <= 0:
            print("Error: n_features must be a positive integer")
            sys.exit(1)
        
        if not os.path.exists(args.csv_path):
            print(f"Error: CSV file not found at {args.csv_path}")
            print("Please update the csv_path argument with the correct path to your dataset")
            sys.exit(1)
        
        # Initialize processor
        processor = NetworkTrafficDataProcessor(args.csv_path, args.n_features)
        
        # Process dataset
        success = processor.process_dataset(args.out_dir)
        
        if success:
            print("Data preprocessing completed successfully!")
            sys.exit(0)
        else:
            print("Data preprocessing failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()