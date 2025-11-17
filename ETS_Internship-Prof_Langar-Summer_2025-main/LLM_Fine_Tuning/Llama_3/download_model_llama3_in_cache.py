# Install huggingface-hub if necessary
pip install huggingface-hub transformers torch

# Find your cache directory
python -c "
from transformers import file_utils
import os
cache_path = file_utils.default_cache_path
print(f'Detected cache path: {cache_path}')
print(f'HF_HOME variable: {os.environ.get(\"HF_HOME\", \"Not defined\")}')
print(f'TRANSFORMERS_CACHE variable: {os.environ.get(\"TRANSFORMERS_CACHE\", \"Not defined\")}')
"

# HuggingFace authentication (required for LLaMA)
# Get your token at: https://huggingface.co/settings/tokens
huggingface-cli login --token YOUR_HF_TOKEN

# Download the model (uses default cache or specify --cache-dir)
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct

# Verify download
huggingface-cli scan-cache | grep Llama-3.2-3B-Instruct

# Test loading
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
   'meta-llama/Llama-3.2-3B-Instruct',
   local_files_only=True
)
print('Model downloaded and accessible!')
"