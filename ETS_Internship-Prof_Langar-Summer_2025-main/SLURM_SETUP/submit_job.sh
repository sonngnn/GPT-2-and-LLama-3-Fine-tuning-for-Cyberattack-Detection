#!/bin/bash

#SBATCH --account=def-rlangar_gpu              # Account to charge compute time
#SBATCH --job-name=trans_SCAFFOLD_FedBN_v2     # Job name
#SBATCH --output=trans_SCAFFOLD_FedBN_v2.out   # Standard output file
#SBATCH --error=trans_SCAFFOLD_FedBN_v2.err    # Standard error file
#SBATCH --nodes=1                              # Number of nodes requested
#SBATCH --time=12:00:00                        # Time limit (HH:MM:SS)
#SBATCH --ntasks=1                             # Number of tasks
#SBATCH --cpus-per-task=2                      # Number of CPU cores per task
#SBATCH --mem=64G                              # Memory allocation
#SBATCH --gres=gpu:v100l:1                     # Request 1 V100 GPU (long version)
#SBATCH --mail-user=[base]@[full_domain_name]  # Email address for notifications. Replace [base]@[full_domain_name] with your real email (e.g., example@example.com) to receive job status updates.
#SBATCH --mail-type=ALL                        # Send email on job BEGIN, END, FAIL, etc.

# Optional: Activate your virtual environment if needed
# Uncomment and adjust the path if your environment is stored in your project
# source /project/def-rlangar/amomo/amomo_venv2/bin/activate

# Run the Python script (without buffering stdout/stderr)
python -u trans_SCAFFOLD_FedBN_v2.py  # SLURM job ID reference: 65044725

# Adapt job name, script name, output and error files name as needed.
# Do not uncomment the lines that begin with SBATCH
