#!/bin/bash

# This is a great minimal test script to validate that your SLURM setup is working properly with GPU access, 
# email notifications, and logging. You can safely use this template as a base for longer or more complex jobs later on.

#SBATCH --account=def-rlangar_gpu               # Account to charge compute time
#SBATCH --job-name=test_slurm                   # Job name
#SBATCH --output=test_slurm.out                 # Output file for standard output
#SBATCH --error=test_slurm.err                  # Output file for errors
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --time=00:10:00                         # Time limit (HH:MM:SS)
#SBATCH --ntasks=1                              # Number of tasks (usually 1 per script)
#SBATCH --cpus-per-task=1                       # Number of CPU cores per task
#SBATCH --mem=1G                                # Allocated memory
#SBATCH --gres=gpu:v100l:1                      # Request 1 V100 GPU
#SBATCH --mail-user=example@example.com         # Email for job notifications
#SBATCH --mail-type=ALL                         # Send email on job BEGIN, END, FAIL, etc.

# Display the hostname and start time
echo "Job started on $(hostname) at $(date)"
echo "Running simple test script..."

# Simulate a workload with a 10-second sleep
sleep 10

# Display the end time
echo "Job finished at $(date)"
