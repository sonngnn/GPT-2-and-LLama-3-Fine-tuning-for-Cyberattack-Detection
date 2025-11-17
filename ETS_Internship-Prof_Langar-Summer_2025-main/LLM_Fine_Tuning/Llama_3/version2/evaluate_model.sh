#!/bin/bash
#SBATCH --account=def-rlangar
#SBATCH --job-name=evaluate_model-2500-10000
#SBATCH --output=evaluate_model-2500-10000.out
#SBATCH --error=evaluate_model-2500-10000.err
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_3g.20gb:1 
#SBATCH --mail-user=bamolitho@gmail.com       
#SBATCH --mail-type=ALL

echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "================================"

# Load required modules
echo "Loading modules..."
module load gcc
module load arrow/21.0.0   # IMPORTANT for pyarrow
module load cuda/12.6

# Activate virtual environment
echo "Activating virtual environment..."
source /project/6084087/amomo/amomo_venv2/bin/activate

python evaluate_model.py
