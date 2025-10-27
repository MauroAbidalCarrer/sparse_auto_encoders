#!/bin/bash
# Sets up the remote machine 

set -e
scp ~/.vast_hf_token vast:~/
# Paths to your local credential files
NETRC_PATH="$HOME/.netrc"
AWS_ACCESS_KEY_PATH="$HOME/.aws_access_key"
AWS_SECRET_KEY_PATH="$HOME/.aws_secret_key"
# Copy credentials to the Vast instance (no prompt)
echo "Copying credentials to Vast..."
scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$NETRC_PATH" "$AWS_ACCESS_KEY_PATH" "$AWS_SECRET_KEY_PATH" vast:~/
# SSH into Vast and install/configure AWS CLI + copy dataset (no prompt)
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null vast <<'EOF'
set -e

# --- Make sure conda is available ---
# Try to detect where conda is installed (common locations)
if [ -d "/root/miniconda3" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
elif [ -d "$HOME/miniconda3" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "Conda not found. Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/miniconda3/etc/profile.d/conda.sh
    conda init bash
fi


# --- Check and install AWS CLI if missing ---
if ! command -v aws >/dev/null 2>&1; then
    echo "AWS CLI not found. Installing..."
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -oq awscliv2.zip
    yes | sudo ./aws/install
    echo "AWS CLI installation completed."
else
    echo "AWS CLI already installed. Skipping installation."
fi

echo "Verifying AWS CLI installation..."
aws --version

# --- Configure AWS CLI only if not already configured ---
if [ ! -f ~/.aws/credentials ]; then
    echo "Configuring AWS CLI credentials..."
    AWS_ACCESS_KEY_ID=$(cat ~/.aws_access_key)
    AWS_SECRET_ACCESS_KEY=$(cat ~/.aws_secret_key)

    mkdir -p ~/.aws
    cat > ~/.aws/credentials <<CREDENTIALS
[default]
aws_access_key_id = ${AWS_ACCESS_KEY_ID}
aws_secret_access_key = ${AWS_SECRET_ACCESS_KEY}
CREDENTIALS

    cat > ~/.aws/config <<CONFIG
[default]
region = eu-west-1
output = table
CONFIG

    echo "AWS CLI configured successfully."
else
    echo "AWS CLI credentials already configured. Skipping setup."
fi

# Clean up key files (optional for security)
rm -f ~/.aws_access_key ~/.aws_secret_key

echo "Cloning repo."
cd repos
git clone $WORKING_REPO
cd sparse_auto_encoders
git remote rename origin github

echo "Creating the conda environment.."
conda env create -yf environment.yml
conda activate SAE
echo "Authenticating hugging face credentials"
hf auth login --token $(cat ~/.vast_hf_token)

# Download dataset from S3
echo "Downloading dataset from S3..."
mkdir -p dataset
aws s3 cp s3://vastdataset/sae/ ./dataset --recursive
echo "S3 dataset download completed successfully."
EOF