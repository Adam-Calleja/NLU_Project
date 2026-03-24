#!/bin/bash

# 1. Define your project directory in the high-speed scratch space
PROJECT_DIR="/scratch/$USER/comp34812_nlu"

# 2. Create the directory and navigate into it
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

echo "Created project directory at: $PROJECT_DIR"

# 3. Load the Anaconda module (adjust this if your CSF uses a different module name)
module load apps/binapps/anaconda3/2022.10  # Check your specific CSF docs for the exact version

# 4. Create and activate a virtual environment
echo "Creating virtual environment 'nlu_env'..."
conda create -n nlu_env python=3.9 -y
source activate nlu_env

# 5. Install the required libraries for the DeBERTa model
echo "Installing dependencies..."
pip install torch torchvision torchaudio
pip install transformers pandas scikit-learn sentencepiece

echo "Setup complete! Please upload your train.csv, dev.csv, test.csv, and train.py to $PROJECT_DIR"