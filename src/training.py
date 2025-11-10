import os
import warnings
from tqdm import tqdm
# from e
from functools import partial
from contextlib import nullcontext

# Ignore the wandb warning(this is a bad practice I know)
warnings.filterwarnings("ignore")

import wandb
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch import Tensor
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split

from models import SparseAutoencoder, TopK


# Torch setup
torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def can_use_bfloat16(device: torch.device) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        # get capability for current device index
        major, minor = torch.cuda.get_device_capability(device.index)
    except Exception:
        major, minor = torch.cuda.get_device_capability()
    # Ampere (sm_80) and newer support efficient bfloat16
    return major >= 8

USE_BFLOAT16 = can_use_bfloat16(device)
print("USE_BFLOAT16 autocast:", USE_BFLOAT16)
# autocast context factory (nullcontext if not available)
if USE_BFLOAT16:
    autocast_ctx = partial(torch.autocast, device_type=device.type, dtype=torch.bfloat16)
else:
    autocast_ctx = nullcontext

# -------- USER: paths to your data files --------
# ACTIVATIONS_PATH = "dataset/middle_layer_activations.pt"  # shape: (N, seq_len, hidden_dim)
OUT_DIR = "sae_output"
os.makedirs(OUT_DIR, exist_ok=True)

# -------- HYPERPARAMS --------
BATCH_SIZE = 512
print("batch size:", BATCH_SIZE)
LR = 1e-4
WEIGHT_DECAY = 1e-5
SAE_EPOCHS = 64
LAMBDA_L1 = 1e-4          # coefficient for L1 on latent
# -------- Utility: load activations and labels --------
N_ACTIVATIONS_DIMS = 1024
# activations: torch.Tensor = torch.load(ACTIVATIONS_PATH, weights_only=True)  # (N_TOKENS, EMBED_DIMS)
# activations = activations.float()

# -------- Create dataset and splits --------
class ResidualActivationsDataset(Dataset):
    def __init__(self, path_to_dataset: str):
        super().__init__()
        self.path_to_dataset = path_to_dataset

    def __len__(self):
        if hasattr(self, "length"):
            return self.length
        self.shard_lengths = []
        self.length = 0
        for filename in tqdm(os.listdir(self.path_to_dataset)[:2], desc="Computing the length of each shard"):
            path = os.path.join(self.path_to_dataset, filename)
            tensor = torch.load(path, weights_only=True)
            # tensor = torch.from_file(path)
            print("shard shape:", tensor.shape)
            self.shard_lengths.append(tensor.shape[0])

        self.length = sum(self.shard_lengths)
        return self.length

    def __getitem__(self, index: int):
        total_shard_length = 0
        shard_it = zip(os.listdir(self.path_to_dataset)[:2], self.shard_lengths)
        for shard_filename, shard_len in shard_it:
            file_path = os.path.join(self.path_to_dataset, shard_filename)
            if total_shard_length + shard_len > index:
                return torch.load(file_path, weights_only=True)[index - total_shard_length]
            total_shard_length += shard_len
        raise IndexError(f"Index {index} is out of range {self.length}")

dataset = ResidualActivationsDataset("dataset")
print("dataset len:", len(dataset))
n_train = int(len(dataset) * 0.8)
n_val = len(dataset) - n_train
train_ds, val_ds = random_split(
    dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42),
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

EXPANSION_RATIO = 16
N_LATENTS = N_ACTIVATIONS_DIMS * EXPANSION_RATIO
USE_NORMALIZATION = False
K = 256
sae = SparseAutoencoder(
    n_inputs=N_ACTIVATIONS_DIMS,
    n_latents=N_LATENTS,
    activation=TopK(256),
    tied=True,
    normalize=USE_NORMALIZATION,
)
sae = sae.to(device)
# sae = torch.compile(sae)
optimizer = torch.optim.AdamW(sae.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
mse_loss = nn.MSELoss(reduction="mean")

# Weights and Biases
LOG_COMMIT_FREQ = 100
wandb.init(
    project="sparse-auto-encoders",
    config={
        "weight_decay": WEIGHT_DECAY,
        "lr": LR,
        "BATCH_SIZE": BATCH_SIZE,
        "l1_loss_weight": LAMBDA_L1,
        "sae_training_epochs": SAE_EPOCHS,
        "using_bfloat16": USE_BFLOAT16,
        "sparse_activations_size": N_LATENTS,
        "normalization": USE_NORMALIZATION,
        "K": K,
    }
)

def train_sae(sae: nn.Module):
    step = 0
    eval_model(sae, step)
    for _ in range(1, SAE_EPOCHS + 1):
        sae.train()
        for (xb, ) in tqdm(train_loader):
            xb = xb.to(device)
            with autocast_ctx():
                _, latents, recons = sae(xb)
                reconstruction_loss = mse_loss(recons, xb)
                l0_loss = (latents > 0).float().mean()
                l1_loss = latents.abs().mean()
                loss = reconstruction_loss #+ LAMBDA_L1 * l1_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log(
                data={
                    "training/recon_loss": reconstruction_loss.item(),
                    "training/l0_loss": l0_loss.item(),
                    "training/loss_l1": l1_loss.item(),
                    "training/loss": loss.item(),
                },
                step=step,
                commit=step % LOG_COMMIT_FREQ == 0,
            )
            step += 1
        eval_model(sae, step)

def eval_model(sae: nn.Module, step: int):
    sae.eval()
    l0_loss = 0
    l1_loss = 0
    reconstruction_loss = 0
    with torch.no_grad():
        for (x, ) in val_loader:
            x = x.to(device)
            batch_weight = x.shape[0] / len(val_loader.dataset)
            _, latents, recons = sae(x)
            l1_loss += latents.abs().mean().item() * batch_weight
            l0_loss += (latents > 0).float().mean().item() * batch_weight
            reconstruction_loss += mse_loss(recons, x).item() * batch_weight
    wandb.log(
        data={
            "validation/l1_loss": l1_loss,
            "validation/l0_loss": l0_loss,
            "validation/reconstruction_loss": reconstruction_loss,
            "validation/loss": reconstruction_loss + l1_loss * LAMBDA_L1
        },
        step=step
    )

if __name__ == "__main__":
    train_sae(sae)