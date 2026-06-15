#!/bin/bash

echo "Downloading torch dependencies..."

pip install tokenizers datasets tqdm json

echo "Dependencies installed successfully."

#-----------------------------------------------------------------------------#

echo "Downloading pre-trained data..."

python tokenizers_dir/DataPrep.py

echo "Pre-trained data downloaded successfully."


echo "Starting training process..."

python tokenizers_dir/Tokenizer_train.py

echo "Training process completed successfully."


echo "Running benchmarks..."

python tokenizers_dir/tokenizers_benchmark.py

echo "Benchmarks completed and results saved successfully."

#------------------------------------------------------------------------------------#
