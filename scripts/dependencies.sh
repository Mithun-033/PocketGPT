echo "Downloading required torch version..."

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "PyTorch version: $TORCH_VERSION"

CUDA_VERSION=$(python -c "import torch; print("cu"+str(int(float(torch.version.cuda)*10)))")
echo "CUDA version: $CUDA_VERSION"

echo "Uninstalling existing torch packages..."
pip uninstall -y torch torchvision torchaudio 

echo "Installing torch packages with specific CUDA version..."
pip install torch==2.11.00 --index-url https://download.pytorch.org/whl/$CUDA_VERSION
echo "Torch packages installed successfully."

#------------------------------------------------------------------------------------#

pip install datasets tokenizers torchinfo tqdm 
echo "All dependencies installed successfully."

#------------------------------------------------------------------------------------#
