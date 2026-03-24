#!/bin/bash --login
#SBATCH -p gpuA                   # Use the high-end GPU partition
#SBATCH -t 01:00:00               # 1 hour is more than enough for inference
#SBATCH -c 2                      # 2 CPU cores
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --mem=16G                 # 16GB RAM is plenty for inference
#SBATCH -J eval_deberta           # Job name
#SBATCH -o eval_output.%j.out     # Standard output log
#SBATCH -e eval_error.%j.err      # Standard error log

cd /scratch/$USER/comp34812_nlu

# Load your Anaconda module
module load apps/binapps/anaconda3/2022.10
source activate nlu_env

# Prevent internet connection hangs
export HF_HOME=/scratch/$USER/huggingface_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job started on $(date)"

# Run the inference script (unbuffered so we see the output live)
python -u eval.py

echo "Job finished on $(date)"