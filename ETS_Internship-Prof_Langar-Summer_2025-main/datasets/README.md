# CIC-IoT-2023 Dataset Processing Scripts

This repository contains two Python scripts for processing the CIC-IoT-2023 dataset for machine learning tasks, specifically for converting numerical network flow data into formats suitable for different ML approaches.

## Scripts Overview

### 1. `build_attacks_benign_ctgan.py`
**Purpose**: Creates a balanced dataset from the raw CIC-IoT-2023 data using CTGAN (Conditional Tabular GAN) for synthetic data generation.

**What it does**:
- Processes multiple CSV files (`part-*.csv`) from the CIC-IoT-2023 dataset
- Balances attack classes by capping each at ~54,545 rows
- Uses CTGAN to synthesize additional samples for minority attack classes
- Creates a final balanced dataset of ~2,000,000 rows
- Canonicalizes labels (e.g., merges DDoS variants, normalizes "BenignTraffic" to "Benign")

**Output**: `CICIoT2023_attacks_benign_CTGAN.csv` - A balanced numerical dataset ready for traditional ML training.

### 2. `preprocess_dataset.py`
**Purpose**: Converts numerical features into natural language descriptions for LLM training.

**What it does**:
- Takes the balanced numerical dataset (from script 1 or any CIC-IoT-2023 CSV)
- Transforms 47 numerical network flow features into human-readable text descriptions
- Creates both binary (Benign/Attack) and multi-class labels
- Generates text suitable for language model fine-tuning

**Output**: `CICIoT2023_attacks_benign_CTGAN_preprocessed.csv` - A text-based dataset with natural language descriptions of network flows.

## Requirements

### Dependencies
```bash
pip install pandas numpy torch sdv tqdm pathlib
```

### SDV Version Support
- **Preferred**: SDV 1.9+ with `single_table` API
- **Fallback**: Legacy SDV versions with `tabular.CTGAN`

## Usage

### Step 1: Balance the Dataset (Optional but Recommended)

If you have the raw CIC-IoT-2023 dataset split into multiple `part-*.csv` files:

```bash
python build_attacks_benign_ctgan.py
```

**Configuration**: Edit the script's user settings section to adjust:
- `TOTAL_TARGET`: Final dataset size (default: 2,000,000)
- `CAP_ATTACK`: Max rows per attack class (default: 54,545)
- `CTGAN_EPOCHS`: Training epochs for synthetic data generation (default: 80)
- `DATA_DIR`: Directory containing `part-*.csv` files

### Step 2: Convert to Text Format

```bash
python preprocess_dataset.py
```

**Configuration**: Edit the script to set:
- `input_file`: Path to your numerical CSV (e.g., output from step 1)
- `output_file`: Where to save the text-formatted dataset
- `sample_size`: Optional parameter to process only a subset (useful for testing)

## Input Format Requirements

- CSV files with a `label` column containing attack/benign classifications
- 47 numerical features as specified in the CIC-IoT-2023 dataset
- For script 1: Multiple `part-*.csv` files in the specified directory
- For script 2: Single CSV file with numerical features

## Output Formats

### Script 1 Output Structure
```csv
flow_duration,Header_Length,Protocol Type,...,label
0.0001,40,6,...,DDoS
0.0023,52,17,...,Benign
...
```

### Script 2 Output Structure
```csv
text,label,binary_label
"Network flow with duration 0.0001 seconds using TCP protocol...",DDoS,Attack
"Network flow with duration 0.0023 seconds using UDP protocol...",Benign,Benign
...
```

## Features

### Script 1 Features:
- **Memory-efficient streaming**: Processes large datasets in chunks
- **Intelligent balancing**: Uses CTGAN for minority classes, duplication fallback for very small classes
- **Robust CSV parsing**: Auto-detects delimiters and handles malformed rows
- **Reproducible**: Fixed random seeds for consistent results
- **Label canonicalization**: Merges similar attack variants

### Script 2 Features:
- **Rich text descriptions**: Converts 47 numerical features into natural language
- **Protocol identification**: Translates protocol numbers to names (TCP, UDP, ICMP)
- **Statistical summaries**: Includes flow statistics, TCP flags, and advanced metrics
- **Dual labeling**: Provides both original and binary (Attack/Benign) labels
- **Token-aware**: Truncates descriptions to prevent token overflow in LLMs

## Example Text Output

```
"Network flow with duration 0.0234 seconds using TCP protocol. TCP flags: 2 SYN, 1 ACK. Services: HTTP. Statistics: total size 1024.0000, average 512.0000. Advanced metrics: inter-arrival time 0.0012"
```

## Performance Notes

- Script 1 is CPU-intensive during CTGAN training but memory-efficient
- Script 2 is lightweight and processes rows quickly
- Both scripts include progress indicators via tqdm
- Processing time scales with dataset size and number of minority classes

## Troubleshooting

### Common Issues:
1. **Missing 'label' column**: Ensure your CSV has a column named 'label'
2. **Memory errors**: Reduce `CHUNKSIZE` in script 1 or use `sample_size` in script 2
3. **SDV import errors**: Check SDV version and install missing dependencies
4. **Empty output**: Verify input file paths and check for data format issues

### Expected Dataset Structure:
The scripts expect the standard CIC-IoT-2023 format with 47 numerical features plus a label column. Missing features will be handled gracefully but may affect output quality.