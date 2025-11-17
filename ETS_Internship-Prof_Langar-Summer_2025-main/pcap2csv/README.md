# PCAP to CSV Feature Extraction Toolkit

A comprehensive Python toolkit for extracting network traffic features from PCAP files and converting them to CSV format for machine learning applications. This toolkit specializes in IoT network traffic analysis and supports multi-protocol feature extraction.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Output Format](#output-format)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This toolkit processes network packet capture (PCAP) files to extract comprehensive features for network traffic analysis, intrusion detection, and IoT device classification. It supports various protocols and communication technologies including TCP, UDP, HTTP/HTTPS, WiFi, Zigbee, and Bluetooth.

The extracted features are organized into multiple categories:
- **Connectivity Features**: IP addresses, ports, protocols, timing
- **Flow Features**: Bidirectional traffic statistics, duration, rates
- **Protocol Features**: Layer 2-4 protocol identification and flags
- **Dynamic Features**: Statistical measures, geometric properties
- **Communication Features**: WiFi, Zigbee, Bluetooth characteristics

## Features

### Core Capabilities
- **Multi-protocol Support**: TCP, UDP, ICMP, ARP, HTTP/HTTPS, DNS, SSH, MQTT, CoAP
- **IoT Protocol Analysis**: WiFi, Zigbee, Bluetooth Low Energy (BLE)
- **Scalable Processing**: Automatic PCAP splitting and parallel processing
- **Flow-based Analysis**: Bidirectional traffic flow reconstruction
- **Statistical Features**: Real-time computation of statistical measures

### Supported Protocols
- **Layer 2**: Ethernet, ARP, RARP, DHCP, LLC
- **Layer 3**: IPv4, ICMP, IGMP
- **Layer 4**: TCP, UDP with flag analysis
- **Application Layer**: HTTP, HTTPS, DNS, SSH, SMTP, Telnet, IRC, MQTT, CoAP
- **IoT Protocols**: WiFi (802.11), Zigbee, Bluetooth

## Requirements

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install tcpdump libpcap-dev

# CentOS/RHEL
sudo yum install tcpdump libpcap-devel

# macOS
brew install tcpdump libpcap
```

### Python Dependencies
```
numpy>=1.19.0
pandas>=1.3.0
scapy>=2.4.5
dpkt>=1.9.8
scipy>=1.7.0
tqdm>=4.62.0
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd pcap2csv
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import dpkt, scapy, pandas; print('Dependencies installed successfully')"
```

## Usage

### Basic Usage

1. **Place your PCAP file** in the project directory and rename it to `filename.pcap`, or modify the `file` variable in `Generating_dataset.py`.

2. **Run the extraction**:
```bash
python Generating_dataset.py
```

3. **Output**: The processed dataset will be saved as `final_dataset.csv` in the project directory.

### Advanced Configuration

#### Modify Processing Parameters
```python
# In Generating_dataset.py
subfiles_size = 10      # MB - Size of PCAP splits
n_threads = 8           # Number of parallel threads
```

#### Custom Labels
```python
# In Feature_extraction.py
category = "DDOS"       # Attack category
attack_type = "SYN_FLOOD"  # Specific attack type
protocol = "TCP"        # Primary protocol
```

### Direct Feature Extraction

For processing without splitting:
```python
from Feature_extraction import Feature_extraction

fe = Feature_extraction()
success = fe.pcap_evaluation("input.pcap", "output_filename")
```

## Architecture

### Core Components

#### 1. Generating_dataset.py
- **Main orchestrator** for the entire pipeline
- Handles PCAP splitting using `tcpdump`
- Manages parallel processing and thread coordination
- Merges individual CSV outputs into final dataset

#### 2. Feature_extraction.py
- **Core feature extraction engine**
- Processes individual packets using dpkt and scapy
- Maintains flow state and statistical computations
- Outputs structured CSV with 49 feature columns

#### 3. Supporting Modules

**Supporting_functions.py**
- Utility functions for IP conversion, protocol identification
- Flow information calculation and flag processing
- Network statistics computation

**Connectivity_features.py**
- Basic connectivity features (IPs, ports, protocols)
- Timing features (TTL, jitter, inter-arrival time)
- Byte counting and flow statistics

**Dynamic_features.py**
- Statistical computations (mean, std, variance)
- Geometric features (magnitude, radius, correlation)
- Real-time sliding window calculations

**Layered_features.py**
- Protocol detection across OSI layers
- Port-based service identification
- Layer-specific feature extraction

**Communication_features.py**
- WiFi 802.11 frame analysis
- Zigbee network layer features
- Bluetooth communication characteristics

### Processing Pipeline

```
PCAP File → Split (tcpdump) → Parallel Processing → Feature Extraction → CSV Merge → Final Dataset
     ↓              ↓                ↓                    ↓              ↓
   Input      Multiple Small     Multi-threaded      Individual      Combined
   File         PCAP Files        Processing          CSV Files       Output
```

## Output Format

### Feature Categories (49 total features)

#### Flow Features
- `flow_duration`: Duration of the network flow
- `Header_Length`: Protocol header length
- `Rate`, `Srate`, `Drate`: Overall, source-to-destination, destination-to-source rates

#### Protocol Identification
- `TCP`, `UDP`, `HTTP`, `HTTPS`, `DNS`, `SSH`, `SMTP`, `IRC`
- `ARP`, `ICMP`, `IPv`, `DHCP`, `Telnet`
- `MQTT`, `CoAP` (IoT protocols)

#### TCP Flags
- `fin_flag_number`, `syn_flag_number`, `rst_flag_number`
- `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`
- `ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count`

#### Statistical Features
- `Tot sum`, `Min`, `Max`, `AVG`, `Std`: Packet size statistics
- `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight`: Geometric properties
- `IAT`: Inter-arrival time between packets

#### Temporal Features
- `Duration`: Total capture duration
- `cumulative_duration`: Cumulative time from start
- `remaining_time`: Remaining capture time

### Sample Output
```csv
flow_duration,Header_Length,Protocol Type,Duration,Rate,HTTP,HTTPS,TCP,UDP,label
0.0,20,6,1.234,15.3,1,0,1,0,DDOS-SYN_FLOOD-TCP
```

## Performance

### Optimization Features
- **Parallel Processing**: Multi-threaded PCAP processing
- **Memory Efficient**: Streaming processing with bounded memory usage
- **Configurable Splitting**: Adjustable PCAP chunk sizes
- **Flow Caching**: Efficient flow state management

### Benchmarks
- **Small Files** (<100MB): ~2-5 minutes
- **Medium Files** (100MB-1GB): ~10-30 minutes  
- **Large Files** (>1GB): ~1-3 hours (depends on thread count)

### Resource Requirements
- **Memory**: ~2-4GB RAM for processing
- **CPU**: Multi-core recommended (scales with thread count)
- **Storage**: 2-3x original PCAP size for temporary files

## Troubleshooting

### Common Issues

#### 1. "tcpdump not found"
```bash
# Install tcpdump
sudo apt-get install tcpdump  # Ubuntu/Debian
sudo yum install tcpdump      # CentOS/RHEL
```

#### 2. Memory Issues with Large Files
- Reduce `subfiles_size` parameter
- Decrease `n_threads` count
- Process files in smaller batches

#### 3. Empty or Incomplete Output
- Check PCAP file integrity
- Verify sufficient disk space
- Review error logs for protocol parsing issues

#### 4. Performance Issues
```python
# Optimize for large files
subfiles_size = 5     # Smaller chunks
n_threads = 4         # Fewer parallel threads
```

### Debug Mode
Enable detailed logging by modifying the print statements in `Feature_extraction.py`:
```python
# Add at the beginning of pcap_evaluation
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests before submitting

### Adding New Features
- **New Protocols**: Extend `Layered_features.py`
- **Additional Statistics**: Modify `Dynamic_features.py`
- **Flow Features**: Update flow processing in `Feature_extraction.py`

### Code Style
- Follow PEP 8 guidelines
- Add docstrings for new functions
- Include error handling for network parsing

## License


## Citation


## Acknowledgments

- Built using dpkt and scapy libraries
- Inspired by CIC IoT Dataset feature extraction methodologies
- Supports research in network security and IoT device classification