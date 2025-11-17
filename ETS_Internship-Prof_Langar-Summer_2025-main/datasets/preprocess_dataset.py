#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_dataset.py

Convert the CIC-IoT-2023 numerical dataset into text format suitable for LLM training.
This script transforms the 47 numerical features into natural language descriptions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def numerical_to_text(row, feature_columns):
    """Convert a row of numerical features to natural language description."""
    
    text_parts = []
    
    # 1. Basic flow characteristics
    flow_duration = row.get('flow_duration', 0)
    duration = row.get('Duration', 0)
    header_len = row.get('Header_Length', 0)
    
    text_parts.append(f"Network flow with duration {flow_duration:.4f} seconds")
    if duration != flow_duration and duration > 0:
        text_parts.append(f"total duration {duration:.4f}")
    if header_len > 0:
        text_parts.append(f"header length {int(header_len)} bytes")
    
    # 2. Protocol identification
    protocol_type = row.get('Protocol Type', 0)
    if protocol_type == 6:
        text_parts.append("using TCP protocol")
    elif protocol_type == 17:
        text_parts.append("using UDP protocol")
    elif protocol_type == 1:
        text_parts.append("using ICMP protocol")
    elif protocol_type > 0:
        text_parts.append(f"using protocol {int(protocol_type)}")
    
    # 3. Rate information
    rate = row.get('Rate', 0)
    srate = row.get('Srate', 0)
    drate = row.get('Drate', 0)
    
    if rate > 0:
        text_parts.append(f"overall rate {rate:.4f}")
    if srate > 0:
        text_parts.append(f"source rate {srate:.4f}")
    if drate > 0:
        text_parts.append(f"destination rate {drate:.4f}")
    
    # 4. TCP Flags (numbers)
    flag_info = []
    tcp_flags = ['fin_flag_number', 'syn_flag_number', 'rst_flag_number', 
                 'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number']
    flag_names = ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'ECE', 'CWR']
    
    for flag_col, flag_name in zip(tcp_flags, flag_names):
        flag_count = row.get(flag_col, 0)
        if flag_count > 0:
            flag_info.append(f"{int(flag_count)} {flag_name}")
    
    if flag_info:
        text_parts.append(f"TCP flags: {', '.join(flag_info)}")
    
    # 5. Flag counts
    count_info = []
    count_flags = ['ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count']
    count_names = ['ACK', 'SYN', 'FIN', 'URG', 'RST']
    
    for count_col, count_name in zip(count_flags, count_names):
        count_val = row.get(count_col, 0)
        if count_val > 0:
            count_info.append(f"{int(count_val)} {count_name} count")
    
    if count_info:
        text_parts.append(f"Flag counts: {', '.join(count_info)}")
    
    # 6. Service/Protocol indicators (binary features)
    services = ['HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 
                'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']
    active_services = []
    
    for service in services:
        if row.get(service, 0) > 0:
            active_services.append(service)
    
    if active_services:
        text_parts.append(f"Services: {', '.join(active_services)}")
    
    # 7. Statistical features
    stats_info = []
    tot_sum = row.get('Tot sum', 0)
    tot_size = row.get('Tot size', 0)
    min_val = row.get('Min', 0)
    max_val = row.get('Max', 0)
    avg_val = row.get('AVG', 0)
    std_val = row.get('Std', 0)
    
    if tot_sum > 0:
        stats_info.append(f"total sum {tot_sum:.4f}")
    if tot_size > 0:
        stats_info.append(f"total size {tot_size:.4f}")
    if min_val != 0:
        stats_info.append(f"minimum {min_val:.4f}")
    if max_val > 0:
        stats_info.append(f"maximum {max_val:.4f}")
    if avg_val > 0:
        stats_info.append(f"average {avg_val:.4f}")
    if std_val > 0:
        stats_info.append(f"standard deviation {std_val:.4f}")
    
    if stats_info:
        text_parts.append(f"Statistics: {', '.join(stats_info)}")
    
    # 8. Advanced metrics
    advanced_info = []
    iat = row.get('IAT', 0)
    number = row.get('Number', 0)
    magnitude = row.get('Magnitue', 0)  # Note: original spelling from dataset
    radius = row.get('Radius', 0)
    covariance = row.get('Covariance', 0)
    variance = row.get('Variance', 0)
    weight = row.get('Weight', 0)
    
    if iat > 0:
        advanced_info.append(f"inter-arrival time {iat:.4f}")
    if number > 0:
        advanced_info.append(f"number {number:.4f}")
    if magnitude > 0:
        advanced_info.append(f"magnitude {magnitude:.4f}")
    if radius > 0:
        advanced_info.append(f"radius {radius:.4f}")
    if covariance != 0:
        advanced_info.append(f"covariance {covariance:.4f}")
    if variance > 0:
        advanced_info.append(f"variance {variance:.4f}")
    if weight > 0:
        advanced_info.append(f"weight {weight:.4f}")
    
    if advanced_info:
        text_parts.append(f"Advanced metrics: {', '.join(advanced_info)}")
    
    # Join all parts with periods, limit length to prevent token overflow
    full_text = ". ".join(text_parts)
    
    # Truncate if too long (approximately 400 characters to stay under token limits)
    if len(full_text) > 400:
        full_text = full_text[:400] + "..."
    
    return full_text

def preprocess_dataset(input_path, output_path, sample_size=None):
    """
    Convert CIC-IoT-2023 dataset from numerical to text format.
    
    Args:
        input_path: Path to the original CSV file
        output_path: Path to save the processed CSV file
        sample_size: Optional - number of samples to process (for testing)
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Validate column structure
    validate_columns(df)
    
    # Get the label column (should be 'label' as the last column)
    label_col = 'label'
    if label_col not in df.columns:
        # Fallback to last column if 'label' not found
        label_col = df.columns[-1]
        print(f"⚠️  'label' column not found, using '{label_col}' as label column")
    
    feature_columns = [col for col in df.columns if col != label_col]
    
    print(f"Label column: {label_col}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Class distribution:\n{df[label_col].value_counts()}")
    
    # Clean data - remove any rows with missing labels
    df = df.dropna(subset=[label_col])
    print(f"After removing missing labels: {df.shape}")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} rows...")
        # Stratified sampling to maintain class distribution
        df = df.groupby(label_col, group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(df)))), 
                              random_state=42)
        ).reset_index(drop=True)
        print(f"After sampling: {df.shape}")
        print(f"Sampled class distribution:\n{df[label_col].value_counts()}")
    
    # Convert to text format
    print("Converting numerical features to text...")
    text_data = []
    
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"Processed {idx}/{len(df)} rows...")
        
        text_description = numerical_to_text(row, feature_columns)
        text_data.append({
            'text': text_description,
            'label': row[label_col]
        })
    
    # Create output dataframe
    output_df = pd.DataFrame(text_data)
    
    # Clean up labels - simplify for binary or multi-class
    print("Cleaning labels...")
    output_df['label'] = output_df['label'].astype(str).str.strip()
    
    # Create binary classification labels (Benign vs Attack)
    binary_labels = output_df['label'].apply(
    lambda x: 'Benign' if x.lower().startswith('benign') else 'Attack'
)
    output_df['binary_label'] = binary_labels
    
    print(f"Final dataset shape: {output_df.shape}")
    print(f"Binary class distribution:\n{binary_labels.value_counts()}")
    print(f"Multi-class distribution:\n{output_df['label'].value_counts()}")
    
    # Save processed dataset
    output_df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")
    
    # Show sample of processed data
    print("\n" + "="*80)
    print("SAMPLE OF PROCESSED DATA:")
    print("="*80)
    for i in range(min(3, len(output_df))):
        print(f"\nSample {i+1}:")
        print(f"Text: {output_df.iloc[i]['text']}")
        print(f"Original Label: {output_df.iloc[i]['label']}")
        print(f"Binary Label: {output_df.iloc[i]['binary_label']}")
        print("-" * 50)
    
    return output_df

def validate_columns(df):
    """Validate that the dataset has the expected 47 columns."""
    expected_columns = [
        'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate',
        'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
        'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
        'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count',
        'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP',
        'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG',
        'Std', 'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance',
        'Variance', 'Weight', 'label'
    ]
    
    missing_cols = set(expected_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_columns)
    
    if missing_cols:
        print(f"Missing expected columns: {missing_cols}")
    if extra_cols:
        print(f"Extra columns found: {extra_cols}")
    
    print(f"Found {len(df.columns)} columns, expected 47")
    return len(missing_cols) == 0

if __name__ == "__main__":
    # Adjust these paths according to your setup
    input_file = "CICIoT2023_attacks_benign_CTGAN.csv"  # Your original dataset
    output_file = "CICIoT2023_attacks_benign_CTGAN_preprocessed.csv"   # Output for GPT-2 training
    
    # For testing, use a smaller sample first (remove sample_size for full dataset)
    preprocess_dataset(input_file, output_file, sample_size=None)