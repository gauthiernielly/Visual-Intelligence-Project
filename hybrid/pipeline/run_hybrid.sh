#!/bin/bash
#SBATCH --job-name=hybrid_vlm
#SBATCH --nodes=1                   # 1 node per array task
#SBATCH --ntasks-per-node=1
#SBATCH --account=cs-503            # CRITICAL: Bills the compute to your class
#SBATCH --qos=cs-503
#SBATCH --cpus-per-task=4           # Give it some CPU cores for video decoding
#SBATCH --gres=gpu:1                # Request 1 GPU per task
#SBATCH --mem=40G                   # 40GB RAM to safely hold the VLM and video frames
#SBATCH --time=23:00:00             
#SBATCH --output=logs/hybrid_%A.out # Save logs with JobID and ArrayID

# Create the logs directory if it doesn't exist yet
mkdir -p logs

# Export the total chunk count so your Python script can read it
export NUM_CHUNKS=2

# Activate your conda environment (adjust 'base' if you use a different env name)
source ~/miniconda3/bin/activate nanovlm

# Run the pipeline!
python hybrid.py