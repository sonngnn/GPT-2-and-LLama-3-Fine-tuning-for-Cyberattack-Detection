#!/bin/bash

# ===========================================
# SLURM Useful Commands for Job Management
# ===========================================

# Submit a SLURM job (runs your batch .sh script)
# Example: sbatch my_script.sh
echo "Submit a job with sbatch:"
echo "sbatch my_script.sh"
echo ""

# List current jobs for your user
echo "List pending/running jobs:"
echo "squeue -u \$USER"
squeue -u $USER
echo ""

# Cancel a job by its JobID (replace 123456 with the actual ID)
echo "Cancel a job (example with dummy ID 123456):"
echo "scancel 123456"
echo ""

# Show detailed information about a specific job
echo "Show detailed info for a job:"
echo "scontrol show job JOB_ID"
echo ""

# View job history (including completed jobs)
echo "Display your job history:"
echo "sacct -u \$USER"
sacct -u $USER
echo ""

# More readable job history including resource usage
echo "Detailed job history with resource usage:"
echo "sacct -u \$USER --format=JobID,JobName,State,Elapsed,MaxRSS"
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS
echo ""

# View available resources (by node, queue, etc.)
echo "Check the state of compute nodes:"
echo "sinfo"
sinfo
echo ""

# Save the job description (useful for reproducing a job)
echo "Save job description (replace JOB_ID):"
echo "scontrol show job JOB_ID > job_info.txt"
echo ""

# View GPU and memory usage (if the sinfon tool is installed)
echo "sinfon: GPU and resource usage info (if available)"
echo "sinfon"
echo ""

# Check job scheduling priorities
# Use sprio -u $USER to see your priority in the queue.
echo "Check scheduling priority:"
echo "sprio -u \$USER"
sprio -u $USER
echo ""

# Reminder
echo "Remember to write your SLURM scripts in executable .sh files and submit them using 'sbatch'"

